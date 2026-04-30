"""Round-trip for shared schema transforms (no API, no DB)."""

from shunya.data.timeframes import BarUnit
from shunya.schemas import (
    BarSpecModel,
    FinStratConfig,
    FinTsRequest,
    bar_spec_model_to_bar_spec,
    merge_finstrat_runtime_dict,
)


def test_bar_spec_model_to_bar_spec_default_and_custom() -> None:
    assert bar_spec_model_to_bar_spec(None).unit == BarUnit.DAYS
    spec = bar_spec_model_to_bar_spec(BarSpecModel(unit="HOURS", step=2))
    assert spec.unit == BarUnit.HOURS and spec.step == 2


def test_fints_request_to_bar_spec() -> None:
    req = FinTsRequest(
        start_date="2020-01-01",
        end_date="2021-01-01",
        ticker_list=["SPY"],
        bar_spec=BarSpecModel(unit="DAYS", step=1),
    )
    bs = bar_spec_model_to_bar_spec(req.bar_spec)
    assert bs.unit == BarUnit.DAYS and bs.step == 1


def test_merge_finstrat_runtime_dict() -> None:
    stored = {"neutralization": "group", "decay": 0.1}
    out = merge_finstrat_runtime_dict(stored, None)
    assert out["neutralization"] == "sector"
    out2 = merge_finstrat_runtime_dict(stored, FinStratConfig(decay=0.2))
    assert out2["decay"] == 0.2
    assert out2["neutralization"] == "sector"
