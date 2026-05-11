"""JAX vs NumPy cross-section backends stay aligned on toy panels."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shunya.adapters.jax_cross_section_ops import JaxCrossSectionOps
from shunya.adapters.numpy_cross_section import NumpyCrossSectionOps


@pytest.mark.parametrize("tail", [0.0, 0.1])
def test_winsorize_jax_numpy_close(tail: float) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20).astype(np.float32)
    x[3] = np.nan
    jax_o = JaxCrossSectionOps()
    np_o = NumpyCrossSectionOps()
    j = np.asarray(jax_o.winsorize(jnp.asarray(x), tail), dtype=np.float32)
    n = np.asarray(np_o.winsorize(x, tail), dtype=np.float32)
    np.testing.assert_allclose(j, n, rtol=1e-5, atol=1e-5)


def test_neutralize_market_jax_numpy_close() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    jax_o = JaxCrossSectionOps()
    np_o = NumpyCrossSectionOps()
    j = np.asarray(jax_o.neutralize_market(jnp.asarray(x)), dtype=np.float32)
    n = np.asarray(np_o.neutralize_market(x), dtype=np.float32)
    np.testing.assert_allclose(j, n, rtol=1e-5, atol=1e-5)


def test_neutralize_groups_jax_numpy_close() -> None:
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    g = np.array(["a", "a", "b", "b", "b"], dtype=object)
    jax_o = JaxCrossSectionOps()
    np_o = NumpyCrossSectionOps()
    j = np.asarray(jax_o.neutralize_groups(jnp.asarray(s), g), dtype=np.float32)
    n = np.asarray(np_o.neutralize_groups(s, g), dtype=np.float32)
    np.testing.assert_allclose(j, n, rtol=1e-4, atol=1e-4)
