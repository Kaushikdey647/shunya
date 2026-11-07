"""Typing protocols for market dashboard services (stable seams for tests)."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from api.schemas.models import MarketHeadlineItem, MarketMoverRow, MarketSnapshotRow


@runtime_checkable
class MarketSnapshotPort(Protocol):
    """Callable protocol for symbol snapshot rows used by market routes."""
    def __call__(
        self,
        symbols: list[str],
        *,
        session: Any | None = None,
    ) -> list[MarketSnapshotRow]:
        ...


@runtime_checkable
class MarketMoversPort(Protocol):
    """Callable protocol for gainers, losers, or active mover lists."""
    def __call__(
        self,
        kind: Literal["gainers", "losers", "active"],
        limit: int,
        *,
        session: Any | None = None,
    ) -> list[MarketMoverRow]:
        ...


@runtime_checkable
class MarketHeadlinesPort(Protocol):
    """Callable protocol for financial headline items."""
    def __call__(self, limit: int, *, session: Any | None = None) -> list[MarketHeadlineItem]:
        ...
