"""Cross-section transforms used by :class:`~shunya.algorithm.finstrat.FinStrat` (swappable backend)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CrossSectionOps(Protocol):
    """Pluggable cross-section backend for :class:`~shunya.algorithm.finstrat.FinStrat`."""

    def winsorize(self, x: Any, tail: float) -> Any:
        """Winsorize one cross-section."""
        ...

    def neutralize_market(self, x: Any) -> Any:
        """Demean one cross-section."""
        ...

    def neutralize_groups(self, x: Any, group_ids: Any) -> Any:
        """Demean within group labels."""
        ...
