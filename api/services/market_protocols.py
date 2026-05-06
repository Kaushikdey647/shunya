"""Typing protocols for market dashboard services (stable seams for tests)."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from api.schemas.models import MarketHeadlineItem, MarketMoverRow, MarketSnapshotRow


@runtime_checkable
class MarketSnapshotPort(Protocol):
    def __call__(
        self,
        symbols: list[str],
        *,
        session: Any | None = None,
    ) -> list[MarketSnapshotRow]:
        ...


@runtime_checkable
class MarketMoversPort(Protocol):
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
    def __call__(self, limit: int, *, session: Any | None = None) -> list[MarketHeadlineItem]:
        ...
