"""Tests for :class:`FinTrade` with a mocked Alpaca client."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pandas as pd
import pytest

from src.algorithm.decision import DecisionContext
from src.algorithm.execution import ExecutionReport
from src.algorithm.finstrat import FinStrat
from src.algorithm.fintrade import FinTrade
from src.utils import indicators

from tests.conftest import make_stub_fints


def _mock_trading_client():
    clock = MagicMock()
    clock.is_open = True
    client = MagicMock()
    client.get_clock.return_value = clock
    acct = MagicMock()
    acct.buying_power = "100000"
    client.get_account.return_value = acct
    client.get_all_positions.return_value = []
    asset = MagicMock()
    asset.tradable = True
    asset.fractionable = True
    asset.shortable = True
    client.get_asset.return_value = asset
    ord_mock = MagicMock()
    ord_mock.id = "ord-1"
    ord_mock.status = "new"
    ord_mock.filled_qty = "0"
    ord_mock.filled_avg_price = None
    client.submit_order.return_value = ord_mock
    obs = MagicMock()
    obs.status = "filled"
    obs.filled_qty = "10"
    obs.filled_avg_price = "101.25"
    client.get_order_by_id.return_value = obs
    return client


def test_fintrade_dry_run_returns_execution_report():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d3 = pd.Timestamp("2020-01-03").normalize()
    fts.df.loc[("AAA", d3), "Close"] = 110.0
    fts.df.loc[("BBB", d3), "Close"] = 90.0

    def algo(panel: jnp.ndarray) -> jnp.ndarray:
        return panel[:, 3].astype(jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        neutralization="market",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    client = _mock_trading_client()
    ft = FinTrade(fs, trading_client=client, paper=True)

    decision = DecisionContext(as_of=d3, data_source="yfinance_research")
    rep = ft.run(20_000.0, fts, decision=decision, dry_run=True, cap_to_buying_power=False)

    assert isinstance(rep, ExecutionReport)
    assert rep.dry_run is True
    assert rep.as_of == decision.as_of_date()
    assert any("yfinance" in w for w in rep.warnings)
    client.submit_order.assert_not_called()


def test_fintrade_submits_when_not_dry_run():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d3 = pd.Timestamp("2020-01-03").normalize()
    fts.df.loc[("AAA", d3), "Close"] = 200.0
    fts.df.loc[("BBB", d3), "Close"] = 50.0

    def algo(panel: jnp.ndarray) -> jnp.ndarray:
        return panel[:, 3].astype(jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        neutralization="market",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    client = _mock_trading_client()
    ft = FinTrade(fs, trading_client=client, paper=True)
    rep = ft.run(
        20_000.0,
        fts,
        as_of=d3,
        dry_run=False,
        min_order_notional=1.0,
        cap_to_buying_power=False,
    )
    assert rep.dry_run is False
    assert client.submit_order.called
    assert all(a.success for a in rep.order_attempts if a.notional >= 1.0)
    assert rep.status_observation_enabled is True
    assert all(a.final_status in ("filled", "dry_run") for a in rep.order_attempts)


def test_require_market_open_raises_when_closed():
    tickers = ["AAA"]
    dates = ["2020-01-02"]
    fts = make_stub_fints(tickers, dates)
    fs = FinStrat(
        fts,
        lambda p: p[:, 3].astype(jnp.float32),
        neutralization="market",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    client = _mock_trading_client()
    client.get_clock.return_value.is_open = False
    ft = FinTrade(fs, trading_client=client, paper=True)
    with pytest.raises(RuntimeError, match="closed"):
        ft.run(
            10_000.0,
            fts,
            as_of=pd.Timestamp("2020-01-02").normalize(),
            dry_run=False,
            require_market_open=True,
            cap_to_buying_power=False,
        )


def test_fintrade_sector_gross_cap_adds_warning_and_rescales():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d3 = pd.Timestamp("2020-01-03").normalize()
    fts.df.loc[("AAA", d3), "Close"] = 200.0
    fts.df.loc[("BBB", d3), "Close"] = 50.0
    fts.df.loc[("AAA", d3), "Sector"] = "Tech"
    fts.df.loc[("BBB", d3), "Sector"] = "Tech"

    def algo(panel: jnp.ndarray) -> jnp.ndarray:
        return panel[:, 3].astype(jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        neutralization="market",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    client = _mock_trading_client()
    ft = FinTrade(fs, trading_client=client, paper=True)
    rep = ft.run(
        20_000.0,
        fts,
        as_of=d3,
        dry_run=True,
        cap_to_buying_power=False,
        sector_gross_cap_fraction=0.5,
    )
    assert any("sector_gross_cap_applied" in w for w in rep.warnings)
