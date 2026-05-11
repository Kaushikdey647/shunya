"""JAX-backed cross-section ops (default; delegates to :mod:`shunya.algorithm.cross_section`)."""

from __future__ import annotations

from typing import Any

from shunya.algorithm import cross_section
from shunya.ports.cross_section_ops import CrossSectionOps


class JaxCrossSectionOps:
    __slots__ = ()

    def winsorize(self, x: Any, tail: float) -> Any:
        return cross_section.winsorize(x, tail)

    def neutralize_market(self, x: Any) -> Any:
        return cross_section.neutralize_market(x)

    def neutralize_groups(self, x: Any, group_ids: Any) -> Any:
        return cross_section.neutralize_groups(x, group_ids)


def default_cross_section_ops() -> CrossSectionOps:
    return JaxCrossSectionOps()
