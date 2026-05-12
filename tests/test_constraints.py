"""Constraint integration tests on FinBT and shared target helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from shunya.algorithm.finbt import FinBT
from shunya.algorithm.finstrat import FinStrat
from shunya.algorithm.targets import (
    apply_group_net_cap,
    cap_deltas_by_adv,
    enforce_turnover_budget,
    target_usd_universe,
)
from tests.conftest import make_stub_fints


def test_constraint_helpers_emit_warnings_style_flags():
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
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="none",
    )
    names = fs.tickers_at(d)
    vec = np.asarray(fs.pass_(None, 20_000.0, execution_date=d, tickers=names), dtype=float)
    targets = target_usd_universe(names, vec, fts.ticker_list)
    group_map = {t: "Tech" for t in fts.ticker_list}
    targets, breached_net = apply_group_net_cap(
        targets,
        group_map,
        max_group_net_fraction=0.1,
        on_breach="rescale",
    )
    current = {t: 0.0 for t in fts.ticker_list}
    targets, obs_turnover, turn_limit = enforce_turnover_budget(
        targets,
        current,
        max_turnover_fraction=0.1,
        on_breach="rescale",
    )
    deltas = {t: float(targets.get(t, 0.0)) - float(current.get(t, 0.0)) for t in fts.ticker_list}
    adv_usd = {"AAA": 200.0 * 1e6, "BBB": 50.0 * 1e6}
    deltas, breached_adv = cap_deltas_by_adv(
        deltas,
        adv_usd,
        max_adv_fraction=0.01,
        on_breach="rescale",
    )
    assert breached_net or obs_turnover > turn_limit + 1e-9 or breached_adv


def test_finbt_accepts_full_constraint_parameters():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    for t in tickers:
        for d in dates:
            fts.df.loc[(t, d), "Sector"] = "Tech" if t == "AAA" else "Energy"
    fs = FinStrat(
        fts,
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="sector",
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
