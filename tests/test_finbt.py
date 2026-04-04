"""Tests for FinBT backtesting wrapper."""

from __future__ import annotations

import jax.numpy as jnp

from shunya.algorithm.finbt import FinBT
from shunya.algorithm.finstrat import FinStrat
from shunya.utils import indicators

from tests.conftest import make_stub_fints


def _close_algo(panel: jnp.ndarray) -> jnp.ndarray:
    return panel[:, indicators.IX_LIVE.CLOSE].astype(jnp.float32)


def test_finbt_runs_and_returns_metrics():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    fs = FinStrat(
        fts,
        _close_algo,
        neutralization="market",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    bt = FinBT(fs, fts, cash=50_000.0).run()
    out = bt.results(show=False)
    assert "metrics" in out
    assert "equity_curve" in out
    assert "turnover_history" in out
    assert "avg_turnover_pct" in out["metrics"]


def test_finbt_group_neutralization_defaults_to_sector_column():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    for t in tickers:
        for d in dates:
            fts.df.loc[(t, d), "Sector"] = "Tech" if t == "AAA" else "Energy"

    fs = FinStrat(
        fts,
        _close_algo,
        neutralization="group",
        panel_columns=indicators.STRATEGY_PANEL_OHLCV_ONLY,
    )
    bt = FinBT(fs, fts, cash=50_000.0).run()
    out = bt.results(show=False)
    assert out["metrics"]["end_value"] > 0.0
    assert "max_group_gross_share_pct" in out["metrics"]
