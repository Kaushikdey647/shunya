"""End-to-end mocked rebalance integration tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pandas as pd

from src.algorithm.finstrat import FinStrat
from src.algorithm.fintrade import FinTrade
from src.utils import indicators
from tests.conftest import make_stub_fints


def _client_with_status(status: str):
    clock = MagicMock()
    clock.is_open = True
    client = MagicMock()
    client.get_clock.return_value = clock
    acct = MagicMock()
    acct.buying_power = "100000"
    client.get_account.return_value = acct
    client.get_all_positions.side_effect = [[], []]
    asset = MagicMock()
    asset.tradable = True
    asset.fractionable = True
    asset.shortable = True
    client.get_asset.return_value = asset
    order = MagicMock()
    order.id = "oid-x"
    order.status = "new"
    client.submit_order.return_value = order
    obs = MagicMock()
    obs.status = status
    obs.filled_qty = "0"
    obs.filled_avg_price = None
    client.get_order_by_id.return_value = obs
    return client


def _build_fintrade(client):
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d3 = pd.Timestamp("2020-01-03").normalize()
    fts.df.loc[("AAA", d3), "Close"] = 200.0
    fts.df.loc[("BBB", d3), "Close"] = 50.0
    fs = FinStrat(
        fts,
        lambda p: p[:, 3].astype(jnp.float32),
        neutralization="market",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    return FinTrade(fs, trading_client=client, paper=True), fts, d3


def test_integration_partial_fill_flow_emits_warning():
    client = _client_with_status("partially_filled")
    ft, fts, d3 = _build_fintrade(client)
    rep = ft.run(
        20_000.0,
        fts,
        as_of=d3,
        dry_run=False,
        min_order_notional=1.0,
        cap_to_buying_power=False,
    )
    assert any("partially_filled_orders" in w for w in rep.warnings)


def test_integration_rejected_flow_keeps_attempt_error_surface():
    client = _client_with_status("rejected")
    ft, fts, d3 = _build_fintrade(client)
    rep = ft.run(
        20_000.0,
        fts,
        as_of=d3,
        dry_run=False,
        min_order_notional=1.0,
        cap_to_buying_power=False,
        reconciliation_policy="warn_only",
    )
    assert all(a.final_status == "rejected" for a in rep.order_attempts)
