"""Instrument OHLCV: Timescale-first with yfinance fallback and optional writeback."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
import pandas as pd
from fastapi import HTTPException

from backtest_api.schemas.models import InstrumentOhlcvResponse, OhlcvBar
from shunya.data.providers import YFinanceMarketDataProvider
from shunya.data.timeframes import bar_spec_is_intraday, default_bar_index_policy
from shunya.data.yfinance_session import build_yfinance_session
from shunya.data.validation import validate_core_ohlcv_coverage
from shunya.data.timescale.intervals import bar_spec_to_interval_key
from shunya.data.timescale.ohlcv_window import period_to_utc_bounds, yfinance_interval_to_bar_spec
from shunya.data.timescale.ohlcv_writeback import replace_ohlcv_range_sync

_log = logging.getLogger(__name__)

# #region agent log
_AGENT_DEBUG_LOG = "/Users/kaushik.dey1/PythonProjects/shunya/.cursor/debug-612bbe.log"


def _agent_log(hypothesis_id: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "612bbe",
            "runId": "post-fix",
            "hypothesisId": hypothesis_id,
            "location": "instrument_ohlcv.resolve",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


# #endregion

OHLCV_SOURCE = "yfinance"
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


_OHLCV_FIELD_NAMES = frozenset({"Open", "High", "Low", "Close", "Volume", "Adj Close"})


def _flatten_ohlcv_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    yfinance ``download`` may return MultiIndex columns either as ``(Ticker, Field)``
    (``group_by='ticker'``) or ``(Field, Ticker)`` / ``('Price','Ticker')`` with field names
    on level 0 (``group_by='column'`` / library default). Downstream expects flat ``Open``/…
    columns.
    """
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df
    sym_u = symbol.upper()
    lev0_names = {str(x) for x in df.columns.get_level_values(0).unique()}
    if lev0_names & _OHLCV_FIELD_NAMES:
        for lev in (1, 0):
            if lev >= df.columns.nlevels:
                continue
            for raw in df.columns.get_level_values(lev).unique():
                if str(raw).upper() != sym_u:
                    continue
                out = df.xs(raw, axis=1, level=lev, drop_level=True)
                if isinstance(out.columns, pd.MultiIndex):
                    return df
                return out.copy()
        return df
    tickers = [str(t) for t in df.columns.get_level_values(0).unique()]
    for t in tickers:
        if t.upper() == sym_u:
            return df[t].copy()
    if len(tickers) == 1:
        return df[tickers[0]].copy()
    return df


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


def _fetch_yfinance_ohlcv(
    symbol: str,
    bar_spec,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    session: object | None,
) -> pd.DataFrame:
    policy = default_bar_index_policy()
    prov = YFinanceMarketDataProvider(session=session)
    try:
        return prov.download([symbol], start, end_exclusive, bar_spec=bar_spec, bar_index_policy=policy)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance download failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail="market data unavailable") from exc


def resolve_instrument_ohlcv_sync(
    symbol: str,
    interval: str,
    period: str,
    *,
    defer_storage: bool = False,
) -> InstrumentOhlcvResult:
    bar_spec = yfinance_interval_to_bar_spec(interval)
    interval_key = bar_spec_to_interval_key(bar_spec)
    start_inclusive, end_exclusive = period_to_utc_bounds(period)
    policy = default_bar_index_policy()
    intraday = bar_spec_is_intraday(bar_spec)
    val_start, val_end = _validation_window(start_inclusive, end_exclusive, intraday=intraday)

    session = build_yfinance_session()

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
            _log.info("timescale unavailable, using yfinance only: %s", exc)
            timescale_ok = False

    if timescale_ok and dsn is not None:
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider

            ts_prov = TimescaleMarketDataProvider(dsn=dsn, source=OHLCV_SOURCE)
            ts_df = ts_prov.download(
                [symbol],
                start_inclusive,
                end_exclusive,
                bar_spec=bar_spec,
                bar_index_policy=policy,
            )
            if ts_df is not None and not ts_df.empty:
                try:
                    validate_core_ohlcv_coverage(
                        ts_df,
                        ticker_list=[symbol],
                        start=val_start,
                        end=val_end,
                        bar_spec=bar_spec,
                        strict_provider_universe=True,
                        strict_ohlcv=True,
                        strict_empty=True,
                        strict_trading_grid=True,
                        bar_index_policy=policy,
                    )
                    bars = _dataframe_to_bars(
                        _flatten_ohlcv_for_symbol(ts_df, symbol), max_rows=MAX_OHLCV_ROWS
                    )
                    return InstrumentOhlcvResult(
                        response=InstrumentOhlcvResponse(
                            symbol=symbol,
                            interval=interval,
                            period=period,
                            bars=bars,
                            data_source="timescale",
                            storage_status="none",
                            storage_error=None,
                            storage_job_id=None,
                            storage_skipped=False,
                        )
                    )
                except ValueError as exc:
                    _log.info("timescale ohlcv incomplete for %s: %s", symbol, exc)
        except Exception as exc:  # noqa: BLE001
            _log.info("timescale read failed for %s: %s", symbol, exc)

    yf_df = _fetch_yfinance_ohlcv(symbol, bar_spec, start_inclusive, end_exclusive, session)
    if yf_df is None or yf_df.empty:
        # #region agent log
        _agent_log(
            "H4",
            "yfinance_empty_or_none",
            {"symbol": symbol, "is_none": yf_df is None, "empty": yf_df is None or bool(yf_df.empty)},
        )
        # #endregion
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=[],
                data_source="yfinance",
                storage_status="none",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=not timescale_ok,
            )
        )

    # #region agent log
    _mi = isinstance(yf_df.columns, pd.MultiIndex)
    _lv0 = (
        [str(x) for x in yf_df.columns.get_level_values(0).unique()][:24]
        if _mi
        else []
    )
    _agent_log(
        "H1",
        "post_yfinance_fetch",
        {
            "symbol": symbol,
            "shape": list(yf_df.shape),
            "empty": bool(yf_df.empty),
            "multiindex": _mi,
            "level0_unique": _lv0,
        },
    )
    # #endregion

    _flat_yf = _flatten_ohlcv_for_symbol(yf_df, symbol)
    # #region agent log
    _agent_log(
        "H1,H2",
        "post_flatten_yf",
        {
            "symbol": symbol,
            "shape": list(_flat_yf.shape),
            "cols": [str(c) for c in _flat_yf.columns][:24],
            "multiindex_after": isinstance(_flat_yf.columns, pd.MultiIndex),
        },
    )
    # #endregion
    bars = _dataframe_to_bars(_flat_yf, max_rows=MAX_OHLCV_ROWS)
    # #region agent log
    _agent_log("H2", "post_bars_yf", {"symbol": symbol, "n_bars": len(bars)})
    # #endregion

    if not timescale_ok or dsn is None:
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                data_source="yfinance",
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
                data_source="yfinance",
                storage_status="deferred",
                storage_error=None,
                storage_job_id=None,
                storage_skipped=False,
            ),
            pending_deferred_writeback=PendingOhlcvWriteback(
                dsn=dsn,
                symbol=symbol,
                interval_key=interval_key,
                source=OHLCV_SOURCE,
                start_inclusive=start_inclusive,
                end_exclusive=end_exclusive,
                ohlcv_df=_flat_yf.copy(),
            ),
        )

    try:
        _run_id, _n_up = replace_ohlcv_range_sync(
            dsn,
            symbol=symbol,
            interval_key=interval_key,
            source=OHLCV_SOURCE,
            start_inclusive=start_inclusive,
            end_exclusive=end_exclusive,
            ohlcv_df=_flat_yf,
        )
        # #region agent log
        _agent_log("H1,H3", "post_writeback", {"symbol": symbol, "rows_upserted": _n_up})
        # #endregion
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=symbol,
                interval=interval,
                period=period,
                bars=bars,
                data_source="yfinance",
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
                data_source="yfinance",
                storage_status="failed",
                storage_error=str(exc)[:2048],
                storage_job_id=None,
                storage_skipped=False,
            )
        )
