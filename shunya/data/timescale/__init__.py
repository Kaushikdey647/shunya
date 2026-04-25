"""Local TimescaleDB / Postgres persistence for OHLCV and fundamentals (optional dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .market_provider import TimescaleMarketDataProvider
    from .fundamental_provider import TimescaleFundamentalDataProvider

__all__ = [
    "TimescaleMarketDataProvider",
    "TimescaleFundamentalDataProvider",
    "bar_spec_to_interval_key",
]


def __getattr__(name: str) -> Any:
    if name == "TimescaleMarketDataProvider":
        from .market_provider import TimescaleMarketDataProvider

        return TimescaleMarketDataProvider
    if name == "TimescaleFundamentalDataProvider":
        from .fundamental_provider import TimescaleFundamentalDataProvider

        return TimescaleFundamentalDataProvider
    if name == "bar_spec_to_interval_key":
        from .intervals import bar_spec_to_interval_key

        return bar_spec_to_interval_key
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
