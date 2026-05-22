"""Pure route resolution: eligibility + MarketRouteDecision (no IO)."""

# TODO(market-data-router): Single entry for route policy; extend registry rows until worker,
# bootstrap scripts, and SQL FIXME sites consume MarketRouteDecision instead of hard-coded Yahoo.

from __future__ import annotations

import os

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
from shunya.data.market_data.context import MarketDataRouteContext
from shunya.data.market_data.decision import MarketRouteDecision
from shunya.data.market_data.errors import (
    MARKET_ROUTE_NO_CREDENTIALS,
    MARKET_ROUTE_UNKNOWN_UPSTREAM,
    MARKET_ROUTE_UNSUPPORTED_MODE,
    MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN,
    MarketRouteError,
)
from shunya.data.market_data.registry import capability_for
from shunya.data.market_data.types import RouteMode, is_upstream_source_id
from shunya.data.timeframes import bar_spec_is_intraday
from shunya.integration.alpaca_settings import try_load_alpaca_settings_from_env


def env_demo_relaxed() -> bool:
    raw = os.environ.get("SHUNYA_MARKET_DATA_DEMO_RELAXED", "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def env_alpaca_bar_upstream_id() -> str:
    raw = (os.environ.get("SHUNYA_ALPACA_BAR_FEED") or "sip").strip().lower()
    if raw == "iex":
        return "alpaca_iex"
    if raw in ("delayed_sip", "delayed-sip"):
        return "alpaca_delayed_sip"
    return "alpaca_sip"


def alpaca_market_keys_available() -> bool:
    return try_load_alpaca_settings_from_env() is not None


def _effective_demo_relaxed(ctx: MarketDataRouteContext) -> bool:
    return bool(ctx.demo_relaxed) or env_demo_relaxed()


def _require_alpaca_keys(*, rule_id: str) -> None:
    if not alpaca_market_keys_available():
        raise MarketRouteError(
            MARKET_ROUTE_NO_CREDENTIALS,
            "Market data routing requires credentials that are not configured.",
            rule_id=rule_id,
        )


def _reject_yfinance_intraday(ctx: MarketDataRouteContext, *, rule_id: str) -> None:
    if bar_spec_is_intraday(ctx.bar_spec) and not _effective_demo_relaxed(ctx):
        raise MarketRouteError(
            MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN,
            "This bar resolution is not available from the requested data source.",
            rule_id=rule_id,
        )


def resolve_market_route(ctx: MarketDataRouteContext, mode: RouteMode) -> MarketRouteDecision:
    """
    Decide upstream order and Timescale cache policy.

    ``mode``:
    - ``\"auto\"`` | ``\"best_effort\"`` — registry + env
    - any :func:`is_upstream_source_id` value — deterministic upstream (may still use Timescale when policy allows)
    """
    intraday = bar_spec_is_intraday(ctx.bar_spec)
    keys = alpaca_market_keys_available()
    alpaca_upstream = env_alpaca_bar_upstream_id()

    if mode == "auto":
        if intraday:
            if keys:
                return MarketRouteDecision(
                    primary_upstream=alpaca_upstream,  # type: ignore[arg-type]
                    fallbacks=(),
                    cache_policy="prefer_timescale",
                    timescale_upstream_attempts=(alpaca_upstream,),
                    rule_id="auto_intraday_alpaca",
                    reason="intraday auto routes to Alpaca when keys are present",
                )
            if _effective_demo_relaxed(ctx):
                return MarketRouteDecision(
                    primary_upstream=STORED_OHLCV_DEFAULT_UPSTREAM_ID,
                    fallbacks=(),
                    cache_policy="bypass_cache",
                    timescale_upstream_attempts=(),
                    rule_id="auto_intraday_demo_yfinance",
                    reason="intraday auto allows yfinance only when demo_relaxed",
                )
            _require_alpaca_keys(rule_id="auto_intraday_no_keys")
        return MarketRouteDecision(
            primary_upstream=STORED_OHLCV_DEFAULT_UPSTREAM_ID,
            fallbacks=(),
            cache_policy="prefer_timescale",
            timescale_upstream_attempts=(STORED_OHLCV_DEFAULT_UPSTREAM_ID,),
            rule_id="auto_daily_yfinance",
            reason="daily auto prefers cached yfinance upstream then live yfinance",
        )

    if mode == "best_effort":
        if intraday:
            chain: list[str] = []
            if keys:
                chain.append(alpaca_upstream)
            if _effective_demo_relaxed(ctx):
                chain.append(STORED_OHLCV_DEFAULT_UPSTREAM_ID)
            if not chain:
                _require_alpaca_keys(rule_id="best_effort_intraday_empty")
            primary = chain[0]
            fallbacks = tuple(str(x) for x in chain[1:])
            return MarketRouteDecision(
                primary_upstream=primary,  # type: ignore[arg-type]
                fallbacks=fallbacks,  # type: ignore[arg-type]
                cache_policy="prefer_timescale",
                timescale_upstream_attempts=tuple(chain),
                rule_id="best_effort_intraday_chain",
                reason="best_effort intraday tries Alpaca then optional yfinance (demo_relaxed)",
            )
        # daily best effort: timescale yfinance then live yfinance (alpaca optional second?)
        return MarketRouteDecision(
            primary_upstream=STORED_OHLCV_DEFAULT_UPSTREAM_ID,
            fallbacks=(env_alpaca_bar_upstream_id(),) if keys else (),
            cache_policy="prefer_timescale",
            timescale_upstream_attempts=(STORED_OHLCV_DEFAULT_UPSTREAM_ID,),
            rule_id="best_effort_daily",
            reason="daily best effort prefers stored yfinance bars",
        )

    if mode == "timescale":
        _reject_yfinance_intraday(ctx, rule_id="timescale_intraday_yfinance_forbidden")
        return MarketRouteDecision(
            primary_upstream=STORED_OHLCV_DEFAULT_UPSTREAM_ID,
            fallbacks=(),
            cache_policy="prefer_timescale",
            timescale_upstream_attempts=(STORED_OHLCV_DEFAULT_UPSTREAM_ID,),
            rule_id="explicit_timescale",
            reason="FinTs timescale provider reads stored Yahoo OHLCV",
        )

    if not isinstance(mode, str):
        raise MarketRouteError(
            MARKET_ROUTE_UNSUPPORTED_MODE,
            "Unsupported market route mode.",
            rule_id="unsupported_mode_type",
        )

    if not is_upstream_source_id(mode):
        raise MarketRouteError(
            MARKET_ROUTE_UNKNOWN_UPSTREAM,
            "Unsupported market data source.",
            rule_id="unknown_upstream",
        )

    cap = capability_for(mode)
    if cap is None:
        raise MarketRouteError(
            MARKET_ROUTE_UNKNOWN_UPSTREAM,
            "Unsupported market data source.",
            rule_id="unknown_upstream_capability",
        )

    if mode == STORED_OHLCV_DEFAULT_UPSTREAM_ID:
        _reject_yfinance_intraday(ctx, rule_id="explicit_yfinance_intraday")

    if cap.requires_alpaca_keys:
        _require_alpaca_keys(rule_id="explicit_alpaca_requires_keys")

    if mode == STORED_OHLCV_DEFAULT_UPSTREAM_ID:
        return MarketRouteDecision(
            primary_upstream=STORED_OHLCV_DEFAULT_UPSTREAM_ID,
            fallbacks=(),
            cache_policy="prefer_timescale",
            timescale_upstream_attempts=(STORED_OHLCV_DEFAULT_UPSTREAM_ID,),
            rule_id="explicit_yfinance",
            reason="explicit yfinance with optional Timescale cache",
        )

    return MarketRouteDecision(
        primary_upstream=mode,  # type: ignore[arg-type]
        fallbacks=(),
        cache_policy="prefer_timescale",
        timescale_upstream_attempts=(mode,),
        rule_id="explicit_upstream",
        reason=f"explicit upstream {mode}",
    )
