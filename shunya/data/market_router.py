"""Impure helpers: materialize OHLCV from a :class:`~shunya.data.market_data.decision.MarketRouteDecision`."""

# TODO(market-data-router): Prefer this module (+ shunya.data.market_data.resolve) over ad-hoc
# yfinance paths; remove duplicate TODOs once callers are migrated (see docs/internal/tech-debt-inventory.md).

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
from shunya.data.market_data.decision import BestEffortReadOutcome, MarketRouteDecision
from shunya.data.ohlcv_multindex import flatten_ohlcv_for_symbol
from shunya.data.providers import AlpacaHistoricalMarketDataProvider, YFinanceMarketDataProvider, env_yfinance_repair_default
from shunya.data.timeframes import BarIndexPolicy, BarSpec, default_bar_index_policy
from shunya.data.timescale.intervals import bar_spec_to_interval_key
from shunya.data.timescale.market_cache_lib import fetch_ohlcv_manifest_last_refresh_sync, ohlcv_manifest_is_fresh
from shunya.data.validation import validate_core_ohlcv_coverage
from shunya.data.yfinance_session import build_yfinance_session

_LOG = logging.getLogger(__name__)


def try_timescale_then_live_ohlcv(
    *,
    symbol: str,
    bar_spec: BarSpec,
    start_inclusive: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    decision: MarketRouteDecision,
    dsn: str | None,
    timescale_ok: bool,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    intraday: bool,
    cache_ttl_days: int,
    bar_index_policy: BarIndexPolicy | None = None,
    yfinance_session: Any | None = None,
) -> tuple[pd.DataFrame, BestEffortReadOutcome, str, str]:
    """
    Return ``(flat_df, outcome, read_path, upstream_source_id)``.

    ``read_path`` is :class:`~shunya.data.market_data.types.ReadPath` value string.
    """
    from shunya.data.market_data.types import ReadPath

    policy = bar_index_policy or default_bar_index_policy()
    interval_key = bar_spec_to_interval_key(bar_spec)
    attempted: list[str] = []

    if (
        dsn
        and timescale_ok
        and decision.cache_policy == "prefer_timescale"
        and decision.timescale_upstream_attempts
    ):
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider
        except ImportError:
            TimescaleMarketDataProvider = None  # type: ignore[misc,assignment]
        if TimescaleMarketDataProvider is not None:
            for src in decision.timescale_upstream_attempts:
                attempted.append(f"timescale:{src}")
                try:
                    ts_prov = TimescaleMarketDataProvider(dsn=dsn, source=src)
                    ts_df = ts_prov.download(
                        [symbol],
                        start_inclusive,
                        end_exclusive,
                        bar_spec=bar_spec,
                        bar_index_policy=policy,
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOG.info("timescale read failed for %s source=%s: %s", symbol, src, exc)
                    continue
                if ts_df is None or ts_df.empty:
                    continue
                last_refresh = fetch_ohlcv_manifest_last_refresh_sync(
                    dsn, ticker=symbol, interval=interval_key, source=src
                )
                if not ohlcv_manifest_is_fresh(last_refresh, ttl_days=max(1, int(cache_ttl_days))):
                    _LOG.info(
                        "timescale ohlcv skipped (manifest missing/stale) for %s interval=%s source=%s",
                        symbol,
                        interval_key,
                        src,
                    )
                    continue
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
                except ValueError as exc:
                    _LOG.info("timescale ohlcv incomplete for %s: %s", symbol, exc)
                    continue
                flat = flatten_ohlcv_for_symbol(ts_df, symbol)
                return (
                    flat,
                    BestEffortReadOutcome(
                        satisfied_source=src,
                        attempted=tuple(attempted),
                        partial_coverage=False,
                    ),
                    ReadPath.TIMESCALE.value,
                    src,
                )

    chain = (decision.primary_upstream,) + decision.fallbacks
    for upstream in chain:
        if upstream.startswith("alpaca"):
            attempted.append(upstream)
            try:
                prov = AlpacaHistoricalMarketDataProvider(bar_feed_upstream=upstream)
                raw = prov.download(
                    [symbol],
                    start_inclusive,
                    end_exclusive,
                    bar_spec=bar_spec,
                    bar_index_policy=policy,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.info("alpaca download failed for %s: %s", symbol, exc)
                continue
            flat = flatten_ohlcv_for_symbol(raw, symbol)
            if flat is not None and not flat.empty:
                return (
                    flat,
                    BestEffortReadOutcome(
                        satisfied_source=upstream,
                        attempted=tuple(attempted),
                        partial_coverage=False,
                    ),
                    ReadPath.LIVE_FETCH.value,
                    upstream,
                )
            continue

        if upstream == STORED_OHLCV_DEFAULT_UPSTREAM_ID:
            attempted.append(STORED_OHLCV_DEFAULT_UPSTREAM_ID)
            sess = yfinance_session if yfinance_session is not None else build_yfinance_session()
            prov = YFinanceMarketDataProvider(session=cast(Any, sess), repair=env_yfinance_repair_default())
            try:
                raw = prov.download(
                    [symbol],
                    start_inclusive,
                    end_exclusive,
                    bar_spec=bar_spec,
                    bar_index_policy=policy,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("yfinance download failed for %s: %s", symbol, exc)
                continue
            flat = flatten_ohlcv_for_symbol(raw, symbol)
            if flat is not None and not flat.empty:
                partial = len(chain) > 1 and upstream != chain[-1]
                return (
                    flat,
                    BestEffortReadOutcome(
                        satisfied_source=upstream,
                        attempted=tuple(attempted),
                        partial_coverage=partial,
                    ),
                    ReadPath.LIVE_FETCH.value,
                    upstream,
                )

    return (
        pd.DataFrame(),
        BestEffortReadOutcome(satisfied_source=None, attempted=tuple(attempted), partial_coverage=False),
        ReadPath.LIVE_FETCH.value,
        decision.primary_upstream,
    )
