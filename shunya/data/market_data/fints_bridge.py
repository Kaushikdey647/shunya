"""Map :class:`~shunya.schemas.fints_models.FinTsRequest` to :func:`resolve_market_route` + provider construction."""

from __future__ import annotations

import os
from typing import Optional

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
from shunya.data.market_data.context import MarketDataRouteContext
from shunya.data.market_data.errors import (
    MARKET_ROUTE_NO_CREDENTIALS,
    MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN,
    MarketRouteError,
)
from shunya.data.market_data.resolve import env_alpaca_bar_upstream_id, resolve_market_route
from shunya.data.providers import AlpacaHistoricalMarketDataProvider, MarketDataProvider
from shunya.errors import ErrorCode, FinTsConfigurationError
from shunya.schemas import FinTsRequest, bar_spec_model_to_bar_spec


def _fints_route_context(req: FinTsRequest) -> MarketDataRouteContext:
    syms = tuple(str(t).strip() for t in req.ticker_list if str(t).strip())
    return MarketDataRouteContext(
        symbols=syms,
        bar_spec=bar_spec_model_to_bar_spec(req.bar_spec),
        demo_relaxed=False,
    )


def _route_mode_for_fints(provider: str) -> str:
    p = str(provider).strip().lower()
    if p == "alpaca":
        return env_alpaca_bar_upstream_id()
    return p


def resolve_market_data_provider(req: FinTsRequest) -> Optional[MarketDataProvider]:
    """
    Same eligibility as HTTP OHLCV routing (:func:`resolve_market_route`), then pick the
    primary :class:`~shunya.data.providers.MarketDataProvider` (v1: no manifest TTL / writeback).
    """
    mode_in = str(req.market_data_provider).strip().lower()
    ctx = _fints_route_context(req)
    route_mode = _route_mode_for_fints(mode_in)

    try:
        resolve_market_route(ctx, route_mode)
    except MarketRouteError as exc:
        if exc.code == MARKET_ROUTE_NO_CREDENTIALS:
            raise FinTsConfigurationError(
                "market_data_provider=alpaca requires configured Alpaca API keys.",
                code=ErrorCode.FIN_TS_ALPACA_KEYS_REQUIRED,
                status_code=503,
            ) from exc
        if exc.code == MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN:
            raise FinTsConfigurationError(
                exc.message,
                code=str(exc.code),
                status_code=503,
            ) from exc
        raise FinTsConfigurationError(
            exc.message,
            code=str(exc.code),
            status_code=503,
        ) from exc

    if mode_in == "yfinance":
        return None

    if mode_in == "alpaca":
        return AlpacaHistoricalMarketDataProvider(bar_feed_upstream=route_mode)

    if mode_in == "timescale":
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider
        except ImportError as exc:
            raise FinTsConfigurationError(
                "Timescale provider requires: pip install 'shunya-py[timescale]'",
                code=ErrorCode.FIN_TS_TIMESCALE_DEPENDENCY,
                status_code=503,
            ) from exc
        try:
            return TimescaleMarketDataProvider(source=STORED_OHLCV_DEFAULT_UPSTREAM_ID)
        except ValueError as exc:
            raise FinTsConfigurationError(
                "Timescale provider requires DATABASE_URL or SHUNYA_DATABASE_URL.",
                code=ErrorCode.FIN_TS_TIMESCALE_DSN_REQUIRED,
                status_code=503,
            ) from exc

    if mode_in == "best_effort":
        if os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL"):
            try:
                from shunya.data.timescale.market_provider import TimescaleMarketDataProvider

                return TimescaleMarketDataProvider(source=STORED_OHLCV_DEFAULT_UPSTREAM_ID)
            except (ImportError, ValueError):
                return None
        return None

    if mode_in == "auto":
        if os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL"):
            try:
                from shunya.data.timescale.market_provider import TimescaleMarketDataProvider

                return TimescaleMarketDataProvider(source=STORED_OHLCV_DEFAULT_UPSTREAM_ID)
            except (ImportError, ValueError):
                return None
        return None

    raise FinTsConfigurationError(
        f"Unsupported market_data_provider {mode_in!r}.",
        code=ErrorCode.FIN_TS_TIMESCALE_UNAVAILABLE,
        status_code=503,
    )
