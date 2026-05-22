"""Immutable routing context (eligibility derives from mode + env, not user flags)."""

from __future__ import annotations

from dataclasses import dataclass

from shunya.data.timeframes import BarSpec


@dataclass(frozen=True)
class MarketDataRouteContext:
    """
    Inputs to :func:`~shunya.data.market_data.resolve.resolve_market_route`.

    ``demo_relaxed`` is **computed** by callers from ``SHUNYA_MARKET_DATA_DEMO_RELAXED``
    (or passed explicitly in tests)—not an end-user HTTP knob in production.
    """

    symbols: tuple[str, ...]
    bar_spec: BarSpec
    dataset: str = "ohlcv"
    demo_relaxed: bool = False
