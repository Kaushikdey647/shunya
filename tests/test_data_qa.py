"""Tests for finTs panel QA diagnostics."""

from __future__ import annotations

import pandas as pd

from tests.conftest import make_stub_fints


def test_data_qa_detects_staleness_and_missing_grid():
    fts = make_stub_fints(["AAA", "BBB"], ["2024-01-02", "2024-01-03"])
    # Remove one row to create missing ticker/date coverage.
    fts.df = fts.df.drop(index=("BBB", pd.Timestamp("2024-01-03")))
    qa = fts.qa_diagnostics(as_of=pd.Timestamp("2024-01-10"), max_stale_days=2)
    assert qa.missing_ticker_dates == 1
    assert qa.stale_days_from_last_date >= 7
    assert any("stale_days_exceeds_limit" in w for w in qa.warnings)


def test_data_qa_detects_invalid_ohlcv_rows():
    fts = make_stub_fints(["AAA"], ["2024-01-02", "2024-01-03"])
    fts.df.loc[("AAA", "2024-01-03"), "High"] = 90.0
    fts.df.loc[("AAA", "2024-01-03"), "Low"] = 95.0
    qa = fts.qa_diagnostics(as_of=pd.Timestamp("2024-01-03"))
    assert qa.invalid_ohlcv_rows >= 1
    assert any("invalid_ohlcv_rows" in w for w in qa.warnings)
