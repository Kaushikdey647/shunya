"""Cross-section transforms used by :class:`~shunya.algorithm.finstrat.FinStrat` (swappable backend)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CrossSectionOps(Protocol):
    def winsorize(self, x: Any, tail: float) -> Any: ...
    def neutralize_market(self, x: Any) -> Any: ...
    def neutralize_groups(self, x: Any, group_ids: Any) -> Any: ...
