"""FinStratConfig schema normalization (no DB)."""

from __future__ import annotations

from backtest_api.schemas.models import FinStratConfig


def test_finstrat_config_maps_legacy_group_to_sector() -> None:
    c = FinStratConfig.model_validate({"neutralization": "group"})
    assert c.neutralization == "sector"


def test_finstrat_config_accepts_sector_and_industry() -> None:
    assert FinStratConfig.model_validate({"neutralization": "sector"}).neutralization == "sector"
    assert FinStratConfig.model_validate({"neutralization": "industry"}).neutralization == "industry"
