"""Unit tests for index OHLCV backfill helpers (no DB, no yfinance)."""

from __future__ import annotations

from api.services.ohlcv_yfinance_backfill import (
    payload_has_index_code,
    tickers_for_ohlcv_backfill,
)
from api import worker_job as worker_job_mod


def test_tickers_for_ohlcv_backfill_includes_benchmark_once() -> None:
    p = {
        "fin_ts": {"ticker_list": ["SPY", "QQQ", "SPY"]},
        "benchmark_ticker": "^GSPC",
    }
    assert tickers_for_ohlcv_backfill(p) == ["SPY", "QQQ", "^GSPC"]


def test_payload_has_index_code() -> None:
    assert payload_has_index_code({"index_code": "SP500"})
    assert not payload_has_index_code({"index_code": ""})
    assert not payload_has_index_code({})


def test_recoverable_fin_ts_data_error() -> None:
    assert worker_job_mod._recoverable_fin_ts_data_error(ValueError("strict_ohlcv: bad"))
    assert worker_job_mod._recoverable_fin_ts_data_error(ValueError("strict_empty: x"))
    assert not worker_job_mod._recoverable_fin_ts_data_error(ValueError("other: x"))
    assert not worker_job_mod._recoverable_fin_ts_data_error(RuntimeError("strict_ohlcv"))
