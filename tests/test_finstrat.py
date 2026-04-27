"""Tests for :class:`FinStrat`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shunya.algorithm.finstrat import FinStrat

from tests.conftest import make_stub_fints


def test_pass_market_neutral_sums_to_zero_and_hits_gross():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    for i, (t, d) in enumerate(fts.df.index):
        fts.df.loc[(t, d), "Close"] = 50.0 + float(i)

    def algo(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        neutralization="market",
        decay=0.0,
        truncation=0.0,
    )
    names = fs.tickers_at("2020-01-03")
    notionals = fs.pass_(None, 30_000.0, tickers=names, execution_date="2020-01-03")
    assert float(np.sum(np.asarray(notionals))) == pytest.approx(0.0, abs=5e-3)
    assert float(np.sum(np.abs(np.asarray(notionals)))) == pytest.approx(30_000.0, rel=1e-4)


def test_group_labels_at_reads_sector_and_industry_columns():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    d = "2020-01-03"
    mapping = {
        "AAA": ("Tech", "Semis"),
        "BBB": ("Tech", "Software"),
        "CCC": ("Energy", "OilGas"),
    }
    for t in tickers:
        fts.df.loc[(t, d), "Sector"] = mapping[t][0]
        fts.df.loc[(t, d), "Industry"] = mapping[t][1]

    fs = FinStrat(
        fts,
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="market",
    )
    names = fs.tickers_at("2020-01-03")
    sec = fs.group_labels_at("2020-01-03", names, "Sector")
    ind = fs.group_labels_at("2020-01-03", names, "Industry")
    assert sec.shape[0] == len(names)
    assert ind.shape[0] == len(names)


def test_pass_sector_resolves_labels_without_group_ids():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    d = "2020-01-03"
    for i, t in enumerate(tickers):
        fts.df.loc[(t, d), "Close"] = 50.0 + float(i)
        fts.df.loc[(t, d), "Sector"] = "Tech" if t != "CCC" else "Energy"

    def algo(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(fts, algo, neutralization="sector", decay=0.0, truncation=0.0)
    names = fs.tickers_at(d)
    notionals = fs.pass_(None, 30_000.0, tickers=names, execution_date=d)
    assert float(np.sum(np.abs(np.asarray(notionals)))) == pytest.approx(30_000.0, rel=1e-4)


def test_pass_industry_demean_within_industry():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    d = "2020-01-03"
    fts.df.loc[("AAA", d), "Industry"] = "Semis"
    fts.df.loc[("BBB", d), "Industry"] = "Semis"
    fts.df.loc[("CCC", d), "Industry"] = "OilGas"

    def algo(ctx) -> jnp.ndarray:
        return jnp.array([1.0, 3.0, 10.0], dtype=jnp.float32)

    fs = FinStrat(fts, algo, neutralization="industry", decay=0.0, truncation=0.0)
    names = ["AAA", "BBB", "CCC"]
    raw = fs.scores_at(d, tickers=names)
    s = fs.process_raw_scores(raw, 1.0, tickers=names, execution_date=d)
    arr = np.asarray(s)
    assert arr.shape == (3,)
    assert float(arr[0] + arr[1]) == pytest.approx(0.0, abs=1e-4)


def test_process_raw_scores_group_requires_group_ids():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    fs = FinStrat(
        fts,
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="group",
    )
    names = fs.tickers_at("2020-01-03")
    raw = fs.scores_at("2020-01-03", tickers=names)
    with pytest.raises(ValueError, match="group_ids"):
        fs.process_raw_scores(raw, 1.0, tickers=names, execution_date="2020-01-03")
