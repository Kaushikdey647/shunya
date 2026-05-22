"""Instrument OHLCV: Timescale-first with yfinance fallback and optional writeback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd
from fastapi import HTTPException

from api.schemas.models import (
    InstrumentOhlcvResponse,
    OhlcvBar,
    OhlcvProvenance,
)
from api.tunable_config import get_effective_tunables
from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
from shunya.data.market_data.context import MarketDataRouteContext
from shunya.data.market_data.decision import MarketRouteDecision
from shunya.data.market_data.errors import MarketRouteError
from shunya.data.market_data.resolve import resolve_market_route
from shunya.data.market_data.types import is_upstream_source_id
from shunya.data.market_router import try_timescale_then_live_ohlcv
from shunya.data.ohlcv_multindex import flatten_ohlcv_for_symbol as _flatten_ohlcv_for_symbol
from shunya.data.timeframes import bar_spec_is_intraday, default_bar_index_policy
from shunya.data.timescale.intervals import bar_spec_to_interval_key
from shunya.data.timescale.ohlcv_window import period_to_utc_bounds, yfinance_interval_to_bar_spec
from shunya.data.timescale.ohlcv_writeback import replace_ohlcv_range_sync
from shunya.data.yfinance_session import build_yfinance_session

_log = logging.getLogger(__name__)

MAX_OHLCV_ROWS = 5000


@dataclass
class PendingOhlcvWriteback:
    """Router schedules :func:`replace_ohlcv_range_sync` after creating a deferred ingestion row."""

    dsn: str
    symbol: str
    interval_key: str
    source: str
    start_inclusive: pd.Timestamp
    end_exclusive: pd.Timestamp
    ohlcv_df: pd.DataFrame


@dataclass
class InstrumentOhlcvResult:
    response: InstrumentOhlcvResponse
    pending_deferred_writeback: PendingOhlcvWriteback | None = None


def _validation_window(
    start_inclusive: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    *,
    intraday: bool,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if intraday:
        return start_inclusive, end_exclusive - pd.Timedelta(seconds=1)
    s = pd.Timestamp(start_inclusive)
    e = pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")
    return s.normalize(), e.normalize()


def _normalize_instrument_route(route: str) -> str:
    r = str(route).strip().lower()
    if r in ("auto", "best_effort", "timescale"):
        return r
    if is_upstream_source_id(r):
        return r
    raise HTTPException(status_code=400, detail="invalid market route")


def _provenance_from(
    *,
    read_path: str,
    upstream_source_id: str,
    decision: MarketRouteDecision,
    attempted: tuple[str, ...],
    partial_coverage: bool,
) -> OhlcvProvenance:
    return OhlcvProvenance(
        read_path=read_path,  # type: ignore[arg-type]
        upstream_source_id=upstream_source_id,
        route_rule_id=decision.rule_id,
        attempted_sources=list(attempted) if attempted else None,
        partial_coverage=partial_coverage if partial_coverage else None,
    )


def _dataframe_to_bars(df: pd.DataFrame, *, max_rows: int) -> list[OhlcvBar]:
    if df is None or df.empty:
        return []
    part = df.sort_index().tail(max_rows)
    bars: list[OhlcvBar] = []
    for ts, row in part.iterrows():
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC")
        else:
            t = t.tz_localize("UTC")
        t_iso = t.isoformat()
        try:
            vol = row["Volume"]
            bars.append(
                OhlcvBar(
                    time=t_iso,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(vol) if pd.notna(vol) else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return bars


def resolve_instrument_ohlcv_sync(
    symbol: str,
    interval: str,
    period: str,
    *,
    defer_storage: bool = False,
    route: str = "auto",
) -> InstrumentOhlcvResult:
    bar_spec = yfinance_interval_to_bar_spec(interval)
    interval_key = bar_spec_to_interval_key(bar_spec)
    start_inclusive, end_exclusive = period_to_utc_bounds(period)
    policy = default_bar_index_policy()
    intraday = bar_spec_is_intraday(bar_spec)
    val_start, val_end = _validation_window(start_inclusive, end_exclusive, intraday=intraday)
    cache_ttl_days = get_effective_tunables().market_data_cache_ttl_days
    session = build_yfinance_session()

    mode = _normalize_instrument_route(route)
    ctx = MarketDataRouteContext(symbols=(symbol,), bar_spec=bar_spec, dataset="ohlcv", demo_relaxed=False)
    try:
        decision = resolve_market_route(ctx, mode)
    except MarketRouteError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    dsn: str | None = None
    if os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL"):
        try:
            from shunya.data.timescale.dbutil import get_database_url

            dsn = get_database_url()
        except ValueError:
            dsn = None

    timescale_ok = False
    if dsn is not None:
        try:
            import psycopg

            with psycopg.connect(dsn, connect_timeout=5) as conn:
                conn.execute("SELECT 1")
            timescale_ok = True
        except Exception as exc:  # noqa: BLE001
            _log.info("timescale unavailable, using live fetch only: %s", exc)
            timescale_ok = False

    flat_df, outcome, read_path, upstream = try_timescale_then_live_ohlcv(
        symbol=symbol,
        bar_spec=bar_spec,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
        decision=decision,
        dsn=dsn,
        timescale_ok=timescale_ok,
        val_start=val_start,
        val_end=val_end,
        intraday=intraday,
        cache_ttl_days=int(max(1, round(cache_ttl_days))),
        bar_index_policy=policy,
        yfinance_session=session,
    )

    prov = _provenance_from(
        read_path=read_path,
        upstream_source_id=upstream,
        decision=decision,
        attempted=outcome.attempted,
        partial_coverage=outcome.partial_coverage,
    )

    if flat_df is None or flat_df.empty:
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=[],
                provenance=prov,
                storage_status="none",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=not timescale_ok,
            )
        )

    bars = _dataframe_to_bars(flat_df, max_rows=MAX_OHLCV_ROWS)
    write_source = outcome.satisfied_source or upstream

    if read_path == "timescale":
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                provenance=prov,
                storage_status="none",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=False,
            )
        )

    if not timescale_ok or dsn is None:
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                provenance=prov,
                storage_status="none",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=True,
            )
        )

    if defer_storage:
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                provenance=prov,
                storage_status="deferred",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=False,
            ),
            pending_deferred_writeback=PendingOhlcvWriteback(
                dsn=dsn,
                symbol=symbol,
                interval_key=interval_key,
                source=write_source,
                start_inclusive=start_inclusive,
                end_exclusive=end_exclusive,
                ohlcv_df=flat_df.copy(),
            ),
        )

    try:
        replace_ohlcv_range_sync(
            dsn,
            symbol=symbol,
            interval_key=interval_key,
            source=write_source,
            start_inclusive=start_inclusive,
            end_exclusive=end_exclusive,
            ohlcv_df=flat_df,
        )
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                provenance=prov,
                storage_status="ok",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=False,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                provenance=prov,
                storage_status="failed",
                storage_error=str(exc)[:2048],
                storage_job_id=None,
                storage_skipped=False,
            )
        )
