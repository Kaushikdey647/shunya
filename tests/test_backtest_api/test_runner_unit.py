"""Unit tests for backtest runner wiring (no live FinTs / backtrader)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest_api.errors import FinTsConfigurationError
from backtest_api.fin_ts_factory import build_fin_ts
from backtest_api.runner import run_backtest_from_payload
from backtest_api.schemas.models import FinTsRequest


def test_build_fin_ts_raises_fin_ts_configuration_error_when_timescale_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backtest_api.fin_ts_factory.resolve_market_data_provider",
        lambda _req: None,
    )
    req = FinTsRequest(
        start_date="2020-01-01",
        end_date="2021-01-01",
        ticker_list=["AAPL"],
        market_data_provider="timescale",
    )
    with pytest.raises(FinTsConfigurationError) as ei:
        build_fin_ts(req)
    assert ei.value.status_code == 503
    assert "timescale" in ei.value.message.lower()


def test_run_backtest_from_payload_success(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    fake_out = {
        "metrics": {
            "total_return_pct": 1.0,
            "sharpe_ratio": 0.1,
            "max_drawdown_pct": 0.0,
            "end_value": 101000.0,
            "bar_unit": "DAYS",
            "bar_step": 1,
            "start_value": 100000.0,
        },
        "equity_curve": pd.DataFrame({"Equity": [100000.0, 100250.0, 100500.0, 100750.0, 101000.0]}, index=idx),
        "turnover_history": pd.DataFrame(),
        "returns_analysis": None,
        "drawdown_analysis": None,
        "sharpe_analysis": None,
        "target_history": [],
        "group_exposure_history": [],
    }
    mock_bt = MagicMock()
    mock_bt.run.return_value = mock_bt
    mock_bt.results.return_value = fake_out

    monkeypatch.setattr("backtest_api.runner.build_fin_ts", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("backtest_api.runner.resolve_alpha_for_backtest", lambda *_a, **_k: (lambda ctx: 0.0))
    monkeypatch.setattr("backtest_api.runner.FinStrat", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("backtest_api.runner.FinBT", lambda *_a, **_k: mock_bt)

    payload = {
        "alpha_id": "00000000-0000-0000-0000-000000000001",
        "fin_ts": {
            "start_date": "2020-01-01",
            "end_date": "2026-01-01",
            "ticker_list": ["AAPL"],
            "market_data_provider": "yfinance",
        },
        "finbt": {"cash": 100000.0},
        "include_test_period_in_results": True,
    }
    finstrat_stored: dict = {}
    serialized, summary = run_backtest_from_payload(
        payload,
        "examples.alphas.sma_ratio_20:alpha",
        None,
        finstrat_stored,
    )
    assert summary["total_return_pct"] == serialized["metrics"]["total_return_pct"]
    assert "equity_curve" in serialized
    assert isinstance(serialized["equity_curve"], list)


def test_run_backtest_from_payload_propagates_fin_ts_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_req):
        raise FinTsConfigurationError("no timescale", status_code=503)

    monkeypatch.setattr("backtest_api.runner.build_fin_ts", _boom)
    payload = {
        "alpha_id": "00000000-0000-0000-0000-000000000001",
        "fin_ts": {
            "start_date": "2020-01-01",
            "end_date": "2026-01-01",
            "ticker_list": ["AAPL"],
            "market_data_provider": "timescale",
        },
        "finbt": {"cash": 100000.0},
    }
    with pytest.raises(FinTsConfigurationError, match="no timescale"):
        run_backtest_from_payload(payload, "examples.alphas.sma_ratio_20:alpha", None, {})
