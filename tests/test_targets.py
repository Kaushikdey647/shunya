"""Tests for shared target / delta helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithm.targets import (
    apply_group_gross_cap,
    apply_slippage_to_fill_price,
    broker_deltas,
    target_usd_universe,
)


def test_broker_deltas_matches_targets_minus_current():
    universe = ["A", "B", "C"]
    targets = {"A": 100.0, "B": -50.0, "C": 0.0}
    current = {"A": 80.0, "B": -30.0, "C": 10.0}
    d = broker_deltas(targets, current, universe)
    assert d["A"] == pytest.approx(20.0)
    assert d["B"] == pytest.approx(-20.0)
    assert d["C"] == pytest.approx(-10.0)


def test_target_usd_universe_maps_panel_order():
    vec = np.array([1.0, 2.0])
    out = target_usd_universe(["B", "A"], vec, ["A", "B", "C"])
    assert out["B"] == pytest.approx(1.0)
    assert out["A"] == pytest.approx(2.0)
    assert out["C"] == pytest.approx(0.0)


def test_apply_slippage_buy_sell_symmetric():
    assert apply_slippage_to_fill_price(100.0, side_is_buy=True, slippage_pct=0.01) == pytest.approx(
        101.0
    )
    assert apply_slippage_to_fill_price(100.0, side_is_buy=False, slippage_pct=0.01) == pytest.approx(
        99.0
    )


def test_apply_group_gross_cap_rescales_breached_group():
    targets = {"A": 100.0, "B": 50.0, "C": -50.0}
    groups = {"A": "Tech", "B": "Tech", "C": "Energy"}
    out, breached = apply_group_gross_cap(
        targets,
        groups,
        max_group_gross_fraction=0.5,
        on_breach="rescale",
    )
    # Initial gross 200; per-group cap 100. Tech gross (150) should be scaled down.
    assert "Tech" in breached
    assert abs(out["A"]) + abs(out["B"]) == pytest.approx(100.0)


def test_apply_group_gross_cap_raise_mode():
    targets = {"A": 100.0, "B": 50.0}
    groups = {"A": "Tech", "B": "Tech"}
    with pytest.raises(ValueError, match="breached"):
        apply_group_gross_cap(
            targets,
            groups,
            max_group_gross_fraction=0.4,
            on_breach="raise",
        )
