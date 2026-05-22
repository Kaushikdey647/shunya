"""Pure routing decision (separate from IO)."""

from __future__ import annotations

from dataclasses import dataclass

from shunya.data.market_data.types import CachePolicy


@dataclass(frozen=True)
class MarketRouteDecision:
    """Result of :func:`~shunya.data.market_data.resolve.resolve_market_route`."""

    primary_upstream: str
    fallbacks: tuple[str, ...]
    cache_policy: CachePolicy
    #: Ordered ``ohlcv_bars.source`` values to try when ``cache_policy == prefer_timescale``.
    timescale_upstream_attempts: tuple[str, ...]
    rule_id: str
    reason: str


@dataclass(frozen=True)
class BestEffortReadOutcome:
    """Which upstream produced data (read paths); used by tests and API envelopes."""

    satisfied_source: str | None
    attempted: tuple[str, ...]
    partial_coverage: bool = False
