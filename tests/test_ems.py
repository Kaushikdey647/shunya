"""EMS schedule and id unit tests."""

from __future__ import annotations

import numpy as np

from shunya.ems.ids import child_client_order_id, parent_root_from_client_order_id, parse_child_client_order_id
from shunya.ems.micro_price import MicroPriceUrgency, QuoteL1, limit_price_for_child
from shunya.ems.schedules import smooth_volume_profile_jax, twap_slice_quantities, vwap_slice_quantities


def test_twap_slice_quantities_sums() -> None:
    q = twap_slice_quantities(100, 7, lot_size=1)
    assert sum(q) == 100
    assert len(q) == 7


def test_vwap_slice_quantities_sums() -> None:
    prof = np.array([1.0, 3.0, 2.0, 0.0])
    q = vwap_slice_quantities(50, prof, lot_size=1)
    assert sum(q) == 50


def test_child_client_order_id_escalation() -> None:
    a = child_client_order_id("pid", 3, 0)
    b = child_client_order_id("pid", 3, 1)
    assert a == "pid:3"
    assert b == "pid:3u1"
    assert parent_root_from_client_order_id(b) == "pid"


def test_parse_child_client_order_id() -> None:
    assert parse_child_client_order_id("abc:2u1") == ("abc", 2)


def test_limit_price_for_child() -> None:
    q = QuoteL1(100.0, 102.0)
    assert limit_price_for_child(q, side="BUY", urgency=MicroPriceUrgency.MID) == 101.0
    assert limit_price_for_child(q, side="BUY", urgency=MicroPriceUrgency.CROSS) == 102.0


def test_smooth_volume_profile_jax_shape() -> None:
    p = np.array([1.0, 4.0, 1.0], dtype=float)
    s = smooth_volume_profile_jax(p, sigma_bins=1.0)
    assert s.shape == p.shape
    assert np.sum(s) > 0
