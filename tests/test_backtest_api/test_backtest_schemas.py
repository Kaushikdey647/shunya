"""Pydantic rules for backtest and data-summary requests."""

from __future__ import annotations

import pytest

from backtest_api.index_catalog import benchmark_for_index
from backtest_api.schemas.models import BacktestCreate, DataSummaryRequest, FinTsRequest


def test_benchmark_for_index_raw_sp500() -> None:
    assert benchmark_for_index("SP500") == "^GSPC"
    assert benchmark_for_index("sp500") == "^GSPC"


def test_benchmark_for_index_bel20_raw() -> None:
    assert benchmark_for_index("BEL20") == "^BFX"


def test_benchmark_unknown_raises() -> None:
    with pytest.raises(KeyError):
        benchmark_for_index("NOT_AN_INDEX")


def test_backtest_create_requires_tickers_without_index() -> None:
    with pytest.raises(ValueError, match="ticker_list"):
        BacktestCreate(
            alpha_id="00000000-0000-0000-0000-000000000000",
            fin_ts=FinTsRequest(
                start_date="2024-01-01",
                end_date="2024-02-01",
                ticker_list=[],
            ),
        )


def test_backtest_create_allows_empty_tickers_with_index_code() -> None:
    b = BacktestCreate(
        alpha_id="00000000-0000-0000-0000-000000000000",
        index_code="SP500",
        fin_ts=FinTsRequest(
            start_date="2024-01-01",
            end_date="2024-02-01",
            ticker_list=[],
        ),
    )
    assert b.index_code == "SP500"
    assert b.omit_index_members_missing_ohlcv is False


def test_backtest_create_partial_index_universe_flag() -> None:
    b = BacktestCreate(
        alpha_id="00000000-0000-0000-0000-000000000000",
        index_code="NASDAQ100",
        omit_index_members_missing_ohlcv=True,
        fin_ts=FinTsRequest(
            start_date="2020-01-01",
            end_date="2026-01-01",
            ticker_list=[],
        ),
    )
    assert b.omit_index_members_missing_ohlcv is True


def test_data_summary_rejects_empty_tickers() -> None:
    with pytest.raises(ValueError, match="ticker_list"):
        DataSummaryRequest(
            start_date="2024-01-01",
            end_date="2024-02-01",
            ticker_list=[],
        )
