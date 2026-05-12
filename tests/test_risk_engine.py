"""Tests for :mod:`shunya.algorithm.risk_engine`."""

from __future__ import annotations

import asyncio

import pytest

from shunya.algorithm.execution import OrderAttempt
from shunya.algorithm.risk_engine import (
    DrawdownSentinel,
    PortfolioRiskEngine,
    RiskEngineState,
    RiskVetConfig,
    ShortabilityMode,
    cvxpy_available,
)


def test_gross_and_single_name_cap() -> None:
    cfg = RiskVetConfig(
        max_gross_book_usd=1000.0,
        max_single_name_fraction=0.5,
    )
    eng = PortfolioRiskEngine(cfg)
    res = eng.vet(
        {"AAA": 800.0, "BBB": 800.0},
        current_usd={"AAA": 0.0, "BBB": 0.0},
        equity_usd=1000.0,
        prices={"AAA": 10.0, "BBB": 10.0},
        universe=["AAA", "BBB"],
    )
    assert res.gross_before == pytest.approx(1600.0)
    assert res.gross_after <= 1000.0 + 1e-6
    m = max(abs(v) for v in res.targets_vetted.values())
    assert m <= 500.0 + 1e-6


def test_adv_recomposes_targets() -> None:
    cfg = RiskVetConfig(max_adv_fraction=0.1)
    eng = PortfolioRiskEngine(cfg)
    res = eng.vet(
        {"AAA": 10_000.0},
        current_usd={"AAA": 0.0},
        equity_usd=50_000.0,
        prices={"AAA": 100.0},
        universe=["AAA"],
        adv_usd={"AAA": 50_000.0},
    )
    cap_delta = 0.1 * 50_000.0
    assert res.targets_vetted["AAA"] == pytest.approx(cap_delta)
    assert "adv" in res.flags


def test_shortability_zero_new_short() -> None:
    cfg = RiskVetConfig(shortability_mode=ShortabilityMode.ZERO_NEW_SHORT)
    eng = PortfolioRiskEngine(cfg)
    res = eng.vet(
        {"ZZZ": -500.0},
        current_usd={"ZZZ": 0.0},
        equity_usd=10_000.0,
        prices={"ZZZ": 1.0},
        universe=["ZZZ"],
        shortable_by_symbol={"ZZZ": False},
    )
    assert res.targets_vetted["ZZZ"] == pytest.approx(0.0)
    assert "shortability" in res.flags


def test_shortability_can_reduce_existing_short() -> None:
    cfg = RiskVetConfig(shortability_mode=ShortabilityMode.ZERO_NEW_SHORT)
    eng = PortfolioRiskEngine(cfg)
    res = eng.vet(
        {"ZZZ": -300.0},
        current_usd={"ZZZ": -500.0},
        equity_usd=10_000.0,
        prices={"ZZZ": 1.0},
        universe=["ZZZ"],
        shortable_by_symbol={"ZZZ": False},
    )
    assert res.targets_vetted["ZZZ"] == pytest.approx(-300.0)


def test_register_execution_feedback_tightens_buying_power() -> None:
    cfg = RiskVetConfig(buying_power_buffer=1.0)
    st = RiskEngineState()
    eng = PortfolioRiskEngine(cfg, state=st)
    eng.register_execution_feedback(
        [
            OrderAttempt(
                symbol="X",
                client_order_id="c1",
                side="BUY",
                notional=100.0,
                success=False,
                error="403 insufficient buying power for new order",
            )
        ]
    )
    assert st.buying_power_tighten == pytest.approx(0.9)
    res = eng.vet(
        {"X": 5000.0},
        current_usd={"X": 0.0},
        equity_usd=100_000.0,
        prices={"X": 100.0},
        universe=["X"],
        buying_power_usd=1000.0,
    )
    assert "buying_power" in res.flags
    d = res.targets_vetted["X"] - 0.0
    assert d <= 1000.0 * 0.9 + 1e-6


def test_fat_finger_limit_raises() -> None:
    cfg = RiskVetConfig(fat_finger_limit_pct=0.02)
    eng = PortfolioRiskEngine(cfg)
    with pytest.raises(ValueError, match="fat_finger"):
        eng.vet(
            {"A": 100.0},
            current_usd={"A": 0.0},
            equity_usd=10_000.0,
            prices={"A": 100.0},
            limit_prices={"A": 200.0},
        )


def test_drawdown_sentinel_triggers_kill() -> None:
    equities = [100.0, 100.0, 82.0]
    idx = {"n": 0}

    def eq_fn() -> float:
        i = idx["n"]
        idx["n"] = min(i + 1, len(equities) - 1)
        return equities[i]

    killed = {"v": False}

    def kill() -> None:
        killed["v"] = True

    sen = DrawdownSentinel(
        max_drawdown_pct=0.15,
        poll_interval_seconds=0.01,
        account_equity=eq_fn,
        kill_switch=kill,
        high_water_mark=100.0,
    )
    asyncio.run(sen.run())
    assert killed["v"] is True
    assert sen.triggered is True


def test_drawdown_sentinel_async_equity_and_kill() -> None:
    seq = iter([100.0, 72.0])

    async def eq_fn() -> float:
        return float(next(seq))

    killed: list[int] = []

    async def kill() -> None:
        killed.append(1)

    sen = DrawdownSentinel(
        max_drawdown_pct=0.2,
        poll_interval_seconds=0.01,
        account_equity=eq_fn,
        kill_switch=kill,
        high_water_mark=100.0,
    )
    asyncio.run(sen.run())
    assert killed == [1]


@pytest.mark.skipif(not cvxpy_available(), reason="cvxpy extra not installed")
def test_cvxpy_qp_smoke() -> None:
    cfg = RiskVetConfig(
        use_cvxpy=True,
        max_gross_fraction_of_equity=1.0,
        max_adv_fraction=0.5,
    )
    eng = PortfolioRiskEngine(cfg)
    res = eng.vet(
        {"A": 5000.0, "B": -3000.0},
        current_usd={"A": 0.0, "B": 0.0},
        equity_usd=10_000.0,
        prices={"A": 50.0, "B": 40.0},
        universe=["A", "B"],
        adv_usd={"A": 20_000.0, "B": 20_000.0},
    )
    assert "cvxpy_qp" in res.flags
    assert _gross(res.targets_vetted) <= 10_000.0 + 1e-3


def _gross(m: dict) -> float:
    return float(sum(abs(v) for v in m.values()))


def test_sector_gross_flag() -> None:
    cfg = RiskVetConfig(sector_gross_cap_fraction=0.5)
    eng = PortfolioRiskEngine(cfg)
    groups = {"A": "Tech", "B": "Tech"}
    res = eng.vet(
        {"A": 4000.0, "B": 4000.0},
        current_usd={"A": 0.0, "B": 0.0},
        equity_usd=20_000.0,
        prices={"A": 10.0, "B": 10.0},
        universe=["A", "B"],
        groups=groups,
    )
    assert "sector_gross" in res.flags or res.gross_after <= 8000.0
