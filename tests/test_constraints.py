"""Constraint integration tests on FinTrade/FinBT paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pandas as pd

from src.algorithm.finbt import FinBT
from src.algorithm.finstrat import FinStrat
from src.algorithm.fintrade import FinTrade
from src.utils import indicators
from tests.conftest import make_stub_fints


def _mock_client():
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
    order = MagicMock()
    order.id = "oid1"
    order.status = "new"
    client.submit_order.return_value = order
    obs = MagicMock()
    obs.status = "filled"
    obs.filled_qty = "1"
    obs.filled_avg_price = "100"
    client.get_order_by_id.return_value = obs
    return client


def test_fintrade_constraints_emit_warnings():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d = pd.Timestamp("2020-01-03")
    for t in tickers:
        fts.df.loc[(t, d), "Sector"] = "Tech"
    fts.df.loc[("AAA", d), "Close"] = 200.0
    fts.df.loc[("BBB", d), "Close"] = 50.0

    fs = FinStrat(
        fts,
        lambda p: p[:, 3].astype(jnp.float32),
        neutralization="none",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    ft = FinTrade(fs, trading_client=_mock_client(), paper=True)
    rep = ft.run(
        20_000.0,
        fts,
        as_of=d,
        dry_run=True,
        cap_to_buying_power=False,
        group_net_cap_fraction=0.1,
        turnover_budget_fraction=0.1,
        adv_participation_fraction=0.01,
        constraints_mode="rescale",
    )
    assert any(
        ("group_net_cap_applied" in w) or ("adv_cap_applied" in w) or ("turnover_budget_applied" in w)
        for w in rep.warnings
    )


def test_finbt_accepts_full_constraint_parameters():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    for t in tickers:
        for d in dates:
            fts.df.loc[(t, d), "Sector"] = "Tech" if t == "AAA" else "Energy"
    fs = FinStrat(
        fts,
        lambda p: p[:, 3].astype(jnp.float32),
        neutralization="group",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    bt = FinBT(
        fs,
        fts,
        sector_gross_cap_fraction=0.5,
        group_net_cap_fraction=0.3,
        turnover_budget_fraction=0.8,
        adv_participation_fraction=0.5,
        constraints_mode="rescale",
    ).run()
    out = bt.results(show=False)
    assert out["metrics"]["end_value"] > 0.0
