"""Transformations between Pydantic contract models and runtime Shunya types."""

from __future__ import annotations

from typing import Any, Optional

from shunya.data.timeframes import BarSpec, BarUnit, default_bar_spec

from .fints_models import BarSpecModel, FinStratConfig


def bar_spec_model_to_bar_spec(m: Optional[BarSpecModel]) -> BarSpec:
    if m is None:
        return default_bar_spec()
    unit = BarUnit(m.unit)
    return BarSpec(unit=unit, step=m.step)


def merge_finstrat_runtime_dict(
    stored: dict[str, Any],
    override: Optional[FinStratConfig],
) -> dict[str, Any]:
    """Merge stored alpha FinStrat JSON with an optional backtest override (validated round-trip)."""
    base = FinStratConfig.model_validate(stored).model_dump(mode="json", exclude_none=True)
    if override is not None:
        base.update(override.model_dump(mode="json", exclude_none=True))
    return FinStratConfig.model_validate(base).model_dump(mode="json", exclude_none=True)
