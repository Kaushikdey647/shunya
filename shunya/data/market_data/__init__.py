"""Market data routing package (registry, pure resolve, types)."""

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
from shunya.data.market_data.context import MarketDataRouteContext
from shunya.data.market_data.decision import BestEffortReadOutcome, MarketRouteDecision
from shunya.data.market_data.errors import (
    MARKET_ROUTE_NO_CREDENTIALS,
    MARKET_ROUTE_UNKNOWN_UPSTREAM,
    MARKET_ROUTE_UNSUPPORTED_MODE,
    MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN,
    MarketRouteError,
)
from shunya.data.market_data.registry import UPSTREAM_CAPABILITIES, UpstreamCapability, capability_for
from shunya.data.market_data.resolve import (
    alpaca_market_keys_available,
    env_alpaca_bar_upstream_id,
    env_demo_relaxed,
    resolve_market_route,
)
from shunya.data.market_data.types import (
    UPSTREAM_IDS,
    CachePolicy,
    Dataset,
    QualityTier,
    ReadPath,
    RouteMode,
    UpstreamSourceId,
    is_upstream_source_id,
)

__all__ = [
    "STORED_OHLCV_DEFAULT_UPSTREAM_ID",
    "BestEffortReadOutcome",
    "CachePolicy",
    "Dataset",
    "MARKET_ROUTE_NO_CREDENTIALS",
    "MARKET_ROUTE_UNKNOWN_UPSTREAM",
    "MARKET_ROUTE_UNSUPPORTED_MODE",
    "MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN",
    "MarketDataRouteContext",
    "MarketRouteDecision",
    "MarketRouteError",
    "QualityTier",
    "ReadPath",
    "RouteMode",
    "UPSTREAM_CAPABILITIES",
    "UPSTREAM_IDS",
    "UpstreamCapability",
    "UpstreamSourceId",
    "alpaca_market_keys_available",
    "capability_for",
    "env_alpaca_bar_upstream_id",
    "env_demo_relaxed",
    "is_upstream_source_id",
    "resolve_market_route",
]
