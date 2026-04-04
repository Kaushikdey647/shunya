"""Tests for JAX cross-section helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shunya.algorithm import cross_section


def test_zscore_finite_vector():
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    z = cross_section.zscore(x)
    assert z.shape == x.shape
    assert float(jnp.nanmean(z)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.nanstd(z)) == pytest.approx(1.0, abs=1e-4)


def test_rank_monotone_and_endpoints():
    x = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
    r = cross_section.rank(x)
    assert float(r[0]) == pytest.approx(0.0)
    assert float(r[2]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(0.5)


def test_neutralize_market_sums_to_zero():
    s = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    n = cross_section.neutralize_market(s)
    assert float(jnp.sum(n)) == pytest.approx(0.0, abs=1e-5)


def test_neutralize_groups_demean_within_group():
    s = jnp.array([1.0, 2.0, 10.0, 20.0], dtype=jnp.float32)
    g = np.array(["a", "a", "b", "b"])
    out = cross_section.neutralize_groups(s, g)
    assert float(out[0] + out[1]) == pytest.approx(0.0, abs=1e-5)
    assert float(out[2] + out[3]) == pytest.approx(0.0, abs=1e-5)


def test_winsorize_tail_zero_identity():
    x = jnp.array([1.0, 2.0, 100.0], dtype=jnp.float32)
    y = cross_section.winsorize(x, 0.0)
    assert jnp.array_equal(x, y)


def test_winsorize_invalid_tail_raises():
    with pytest.raises(ValueError):
        cross_section.winsorize(jnp.array([1.0, 2.0]), 0.6)


def test_scale_targets_requested_gross():
    x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    y = cross_section.scale(x, target=2.0)
    assert float(jnp.sum(jnp.abs(y))) == pytest.approx(2.0, abs=1e-6)


def test_scale_non_finite_inputs_zeroed():
    x = jnp.array([jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float32)
    y = cross_section.scale(x)
    assert jnp.array_equal(y, jnp.zeros_like(x))


def test_sign_maps_to_minus_one_zero_plus_one():
    x = jnp.array([-2.0, 0.0, 3.0, jnp.nan], dtype=jnp.float32)
    y = cross_section.sign(x)
    assert np.allclose(np.asarray(y), np.array([-1.0, 0.0, 1.0, 0.0], dtype=float))
