"""Fixed backtest calendar windows and tune-only result trimming."""

from __future__ import annotations

import pandas as pd

from backtest_api.backtest_windows import (
    BACKTEST_SIM_END_EXCLUSIVE,
    BACKTEST_SIM_START,
    BACKTEST_TEST_START,
    normalize_backtest_create,
    standard_backtest_bar_spec,
)
from backtest_api.result_tune_filter import apply_tune_only_to_finbt_results
from backtest_api.schemas.models import BacktestCreate, FinTsRequest


def test_normalize_backtest_create_overwrites_window_and_bar_spec() -> None:
    b = BacktestCreate(
        alpha_id="00000000-0000-0000-0000-000000000000",
        index_code="SP500",
        fin_ts=FinTsRequest(
            start_date="2010-01-01",
            end_date="2011-01-01",
            ticker_list=[],
            bar_spec=None,
        ),
    )
    n = normalize_backtest_create(b)
    assert n.fin_ts.start_date == BACKTEST_SIM_START
    assert n.fin_ts.end_date == BACKTEST_SIM_END_EXCLUSIVE
    assert n.fin_ts.bar_spec is not None
    assert n.fin_ts.bar_spec.unit == "DAYS"
    assert n.fin_ts.bar_spec.step == 1


def test_standard_backtest_bar_spec_is_daily() -> None:
    s = standard_backtest_bar_spec()
    assert s.unit == "DAYS" and s.step == 1


def test_apply_tune_only_trims_rows_before_test_start() -> None:
    idx = pd.date_range("2024-06-01", periods=400, freq="D")
    eq = pd.DataFrame({"Equity": range(len(idx))}, index=idx)
    raw = {
        "equity_curve": eq,
        "turnover_history": pd.DataFrame(),
        "target_history": [],
        "group_exposure_history": [],
        "metrics": {"bar_unit": "DAYS", "bar_step": 1, "start_value": 100_000.0},
        "returns_analysis": {},
        "drawdown_analysis": {},
        "sharpe_analysis": {},
    }
    out = apply_tune_only_to_finbt_results(raw, include_test=False)
    assert len(out["equity_curve"]) < len(eq)
    assert out["equity_curve"].index.max() < pd.Timestamp(BACKTEST_TEST_START)
    assert out["returns_analysis"] is None


def test_apply_tune_only_noop_when_include_test() -> None:
    idx = pd.date_range("2024-06-01", periods=10, freq="D")
    eq = pd.DataFrame({"Equity": [1.0] * 10}, index=idx)
    raw = {
        "equity_curve": eq,
        "turnover_history": pd.DataFrame(),
        "target_history": [],
        "group_exposure_history": [],
        "metrics": {"bar_unit": "DAYS", "bar_step": 1},
        "returns_analysis": {"x": 1},
        "drawdown_analysis": {},
        "sharpe_analysis": {},
    }
    out = apply_tune_only_to_finbt_results(raw, include_test=True)
    assert len(out["equity_curve"]) == len(eq)
    assert out["returns_analysis"] == {"x": 1}
