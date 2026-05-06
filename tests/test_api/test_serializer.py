from __future__ import annotations

import pandas as pd

from api.serializer import equity_curve_to_records, result_summary_from_metrics, serialize_backtest_result


def test_equity_curve_to_records() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    df = pd.DataFrame({"Equity": [100.0, 101.0], "Peak": [100.0, 101.0], "DrawdownPct": [0.0, 0.0]}, index=idx)
    df.index.name = "Date"
    recs = equity_curve_to_records(df)
    assert len(recs) == 2
    assert "Date" in recs[0] or "Date" in str(recs[0])


def test_serialize_minimal() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    eq = pd.DataFrame({"Equity": [100_000.0], "Peak": [100_000.0], "DrawdownPct": [0.0]}, index=idx)
    eq.index.name = "Date"
    raw = {
        "metrics": {"total_return_pct": 0.0, "sharpe_ratio": None, "bar_unit": "DAYS", "bar_step": 1},
        "equity_curve": eq,
        "turnover_history": pd.DataFrame(),
        "returns_analysis": {},
        "drawdown_analysis": {},
        "sharpe_analysis": {},
        "target_history": [],
    }
    ser = serialize_backtest_result(raw, max_target_history=10)
    assert "metrics" in ser
    summ = result_summary_from_metrics(ser["metrics"])
    assert "total_return_pct" in summ
    assert "sharpe_ratio" in summ


def test_result_summary_includes_new_headline_keys_when_present() -> None:
    metrics = {
        "total_return_pct": 10.0,
        "cagr_pct": 8.5,
        "sharpe_ratio": 1.2,
        "max_drawdown_pct": 5.0,
        "win_rate_pct": 52.3,
        "end_value": 110_000.0,
        "bar_unit": "DAYS",
        "bar_step": 1,
    }
    summ = result_summary_from_metrics(metrics)
    assert summ["cagr_pct"] == 8.5
    assert summ["win_rate_pct"] == 52.3
