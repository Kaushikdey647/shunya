"""Tests for :mod:`shunya.algorithm.portfolio_manager`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from shunya.algorithm.finstrat import FinStrat
from shunya.algorithm.portfolio_manager import (
    PORTFOLIO_PERF_KEY,
    AlphaBlendConfig,
    AlphaBlendPortfolioManager,
    PortfolioConstructionService,
    PortfolioManager,
    RollingSharpeTracker,
    StrategyRegistry,
    StrategySpec,
    TargetBlendConfig,
    VirtualLedger,
    allocate_proportional_by_request,
    combine_weighted_targets,
    inverse_vol_weights,
    mark_to_market_strategy_pnl_usd,
    sum_target_maps,
)
from shunya.algorithm.targets import target_usd_universe
from tests.conftest import make_stub_fints


def test_combine_weighted_targets() -> None:
    net = combine_weighted_targets([({"A": 100.0, "B": -50.0}, 0.5), ({"A": 20.0, "C": 10.0}, 0.5)])
    assert net["A"] == pytest.approx(60.0)
    assert net["B"] == pytest.approx(-25.0)
    assert net["C"] == pytest.approx(5.0)


def test_sum_target_maps() -> None:
    s = sum_target_maps([{"A": 1.0}, {"A": 2.0, "B": 3.0}])
    assert s == {"A": 3.0, "B": 3.0}


def test_rolling_sharpe_tracker() -> None:
    tr = RollingSharpeTracker(window=5)
    tr.record_return("s1", 0.01)
    assert tr.rolling_sharpe("s1") is None
    tr.record_return("s1", 0.02)
    sh = tr.rolling_sharpe("s1")
    assert sh is not None and np.isfinite(sh)


def test_portfolio_manager_weights_must_sum_to_one() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    fs = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none")
    with pytest.raises(ValueError, match="sum to 1"):
        PortfolioManager([("a", fs, 0.5), ("b", fs, 0.4)])


def test_portfolio_manager_two_identical_halves_matches_full_book() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d = pd.Timestamp("2020-01-03")
    fs = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none")
    names = fs.tickers_at(d)
    full = target_usd_universe(
        names,
        np.asarray(fs.pass_(None, 10_000.0, execution_date=d, tickers=names), dtype=float),
        fts.ticker_list,
    )
    pm = PortfolioManager([("a", fs, 0.5), ("b", fs, 0.5)], sharpe_window=10)
    net = pm.net_targets(capital=10_000.0, execution_date=d)
    for k in full:
        assert net[k] == pytest.approx(float(full[k]))


def test_rolling_sharpe_portfolio_key() -> None:
    trk = RollingSharpeTracker(window=3)
    trk.record_return("s1", 0.01)
    assert trk.rolling_sharpe("s1") is None
    trk.record_return("s1", 0.02)
    assert trk.rolling_sharpe("s1") is not None
    trk.record_return(PORTFOLIO_PERF_KEY, 0.01)
    assert trk.rolling_sharpe(PORTFOLIO_PERF_KEY) is None
    trk.record_return(PORTFOLIO_PERF_KEY, 0.02)
    assert trk.rolling_sharpe(PORTFOLIO_PERF_KEY) is not None


def test_allocate_proportional_by_request_splits_signed_fill() -> None:
    out = allocate_proportional_by_request(1000.0, {"a": 600.0, "b": 400.0})
    assert out["a"] == pytest.approx(600.0)
    assert out["b"] == pytest.approx(400.0)
    assert sum(out.values()) == pytest.approx(1000.0)
    out2 = allocate_proportional_by_request(2000.0, {"a": 6000.0, "b": -4000.0})
    assert sum(out2.values()) == pytest.approx(2000.0)
    assert out2["a"] + out2["b"] == pytest.approx(2000.0)


def test_inverse_vol_weights() -> None:
    w = inverse_vol_weights({"x": 0.04, "y": 0.01})
    assert w["y"] > w["x"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_strategy_registry_rejects_duplicate_ids() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    fs = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none")
    with pytest.raises(ValueError, match="unique"):
        StrategyRegistry.from_specs(
            [
                StrategySpec("dup", fs, conviction_z=1.0),
                StrategySpec("dup", fs, conviction_z=1.0),
            ]
        )


def test_alpha_blend_matches_manual_blended_raw() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d = pd.Timestamp("2020-01-03")
    fs_a = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none", decay=0.0)
    fs_b = FinStrat(fts, lambda ctx: (ctx.close.latest * 2.0).astype(jnp.float32), neutralization="none", decay=0.0)
    master = FinStrat(fts, lambda ctx: ctx.close.latest, neutralization="none", decay=0.0)
    reg = StrategyRegistry.from_specs(
        [
            StrategySpec("a", fs_a, conviction_z=1.0),
            StrategySpec("b", fs_b, conviction_z=1.0),
        ]
    )
    ab = AlphaBlendPortfolioManager(registry=reg, master=master, sharpe_window=10)
    names = fs_a.tickers_at(d)
    ctx = fs_a.context_at(d, tickers=names)
    r1 = fs_a.scores_from_context(ctx)
    r2 = fs_b.scores_from_context(ctx)
    blended = 0.5 * np.asarray(r1, dtype=float) + 0.5 * np.asarray(r2, dtype=float)
    expected = target_usd_universe(
        names,
        np.asarray(master.process_raw_scores(blended, 10_000.0, tickers=names, execution_date=d), dtype=float),
        fts.ticker_list,
    )
    out = ab.net_targets(capital=10_000.0, execution_date=d)
    for k in expected:
        assert out[k] == pytest.approx(float(expected[k]))


def test_alpha_blend_sub_sector_neutralization_not_used_on_raw_path() -> None:
    """Sector neutralization on a sub-strat does not affect scores_from_context (only pass_)."""
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d = pd.Timestamp("2020-01-03")
    for t in tickers:
        fts.df.loc[(t, d), "Sector"] = "Tech" if t == "AAA" else "Energy"
    fs_sector = FinStrat(
        fts,
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="sector",
        decay=0.0,
    )
    fs_plain = FinStrat(
        fts,
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="none",
        decay=0.0,
    )
    master = FinStrat(fts, lambda ctx: ctx.close.latest, neutralization="none", decay=0.0)
    reg = StrategyRegistry.from_specs(
        [
            StrategySpec("sec", fs_sector, conviction_z=1.0),
            StrategySpec("plain", fs_plain, conviction_z=1.0),
        ]
    )
    ab = AlphaBlendPortfolioManager(registry=reg, master=master, sharpe_window=10)
    out_sector = ab.net_targets(capital=10_000.0, execution_date=d)
    reg2 = StrategyRegistry.from_specs(
        [
            StrategySpec("p1", fs_plain, conviction_z=1.0),
            StrategySpec("p2", fs_plain, conviction_z=1.0),
        ]
    )
    ab2 = AlphaBlendPortfolioManager(registry=reg2, master=master, sharpe_window=10)
    out_plain = ab2.net_targets(capital=10_000.0, execution_date=d)
    for k in out_plain:
        assert out_sector[k] == pytest.approx(float(out_plain[k]))


def test_virtual_ledger_and_allocate() -> None:
    leg = VirtualLedger()
    alloc = allocate_proportional_by_request(100.0, {"s1": 30.0, "s2": 70.0})
    for sid, dv in alloc.items():
        leg.apply_delta(sid, "SPY", dv)
    assert leg.position_usd("s1", "SPY") == pytest.approx(30.0)
    assert leg.position_usd("s2", "SPY") == pytest.approx(70.0)


def test_rolling_path_correlation_after_two_evals() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d2 = pd.Timestamp("2020-01-03")
    d3 = pd.Timestamp("2020-01-06")
    fts.df.loc[("AAA", d3), "Close"] = 105.0
    fs_a = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none", decay=0.0)
    fs_b = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none", decay=0.0)
    master = FinStrat(fts, lambda ctx: ctx.close.latest, neutralization="none", decay=0.0)
    reg = StrategyRegistry.from_specs(
        [
            StrategySpec("a", fs_a, conviction_z=1.0),
            StrategySpec("b", fs_b, conviction_z=1.0),
        ]
    )
    ab = AlphaBlendPortfolioManager(registry=reg, master=master, sharpe_window=10, raw_history_maxlen=10)
    ab.net_targets(capital=10_000.0, execution_date=d2)
    assert ab.rolling_cross_section_path_correlation("a", "b") is None
    ab.net_targets(capital=10_000.0, execution_date=d3)
    c = ab.rolling_cross_section_path_correlation("a", "b")
    assert c is not None and abs(c - 1.0) < 1e-5


def test_correlation_penalty_scales_master_capital_not_convictions() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d2 = pd.Timestamp("2020-01-03")
    d3 = pd.Timestamp("2020-01-06")
    fts.df.loc[("AAA", d3), "Close"] = 105.0
    fs_a = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none", decay=0.0)
    fs_b = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none", decay=0.0)
    master = FinStrat(fts, lambda ctx: ctx.close.latest, neutralization="none", decay=0.0)
    reg = StrategyRegistry.from_specs(
        [
            StrategySpec("a", fs_a, conviction_z=1.0),
            StrategySpec("b", fs_b, conviction_z=1.0),
        ]
    )
    capital = 10_000.0
    ab_baseline = AlphaBlendPortfolioManager(
        registry=reg, master=master, sharpe_window=10, raw_history_maxlen=10
    )
    ab_baseline.net_targets(capital=capital, execution_date=d2)
    base = ab_baseline.net_targets(capital=capital, execution_date=d3)
    g_base = sum(abs(v) for v in base.values())

    reg2 = StrategyRegistry.from_specs(
        [
            StrategySpec("a", fs_a, conviction_z=1.0),
            StrategySpec("b", fs_b, conviction_z=1.0),
        ]
    )
    ab_pen = AlphaBlendPortfolioManager(
        registry=reg2,
        master=master,
        sharpe_window=10,
        raw_history_maxlen=10,
        correlation_max_pairwise_threshold=0.01,
        correlation_penalty=0.5,
    )
    ab_pen.net_targets(capital=capital, execution_date=d2)
    out = ab_pen.construct(capital=capital, execution_date=d3)
    assert out.correlation_penalty_applied is True
    assert out.active_capital == pytest.approx(capital * 0.5)
    assert out.requested_capital == pytest.approx(capital)
    g_pen = sum(abs(v) for v in out.targets.values())
    assert g_pen < g_base - 1e-6


def test_mark_to_market_strategy_pnl_usd() -> None:
    leg = VirtualLedger()
    leg.apply_delta("s1", "SPY", 1000.0)
    pnl = mark_to_market_strategy_pnl_usd(leg, {"SPY": 110.0}, {"SPY": 100.0})
    assert pnl["s1"] == pytest.approx(100.0)


def test_portfolio_construction_service_return_feed_records_returns() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    d = pd.Timestamp("2020-01-03")
    fs = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none")

    class _Feed:
        def __init__(self) -> None:
            self._n = 0

        def daily_returns_asof(self, execution_date: pd.Timestamp):
            if execution_date == d:
                self._n += 1
                return {"m": 0.01 * self._n}
            return None

    svc = PortfolioConstructionService(
        TargetBlendConfig((("m", fs, 1.0),)),
        sharpe_window=5,
        return_feed=_Feed(),
        record_returns_from_feed_on_construct=True,
    )
    assert svc.rolling_sharpe("m") is None
    svc.construct(capital=1.0, execution_date=d)
    assert svc.rolling_sharpe("m") is None
    svc.construct(capital=1.0, execution_date=d)
    assert svc.rolling_sharpe("m") is not None
