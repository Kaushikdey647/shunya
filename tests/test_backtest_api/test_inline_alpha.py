from __future__ import annotations

import pytest

from backtest_api.inline_alpha import resolve_alpha_from_source
from backtest_api.resolver import resolve_alpha_for_backtest
from shunya.algorithm.alpha_context import AlphaContext
import jax.numpy as jnp


def test_resolve_from_source_valid() -> None:
    src = """
import jax.numpy as jnp

def alpha(ctx):
    return ctx.cs.rank(ctx.close)
"""
    fn = resolve_alpha_from_source(src)
    n = 3
    t = 5
    zero = jnp.zeros((t, n), dtype=jnp.float32)
    ctx = AlphaContext(
        open=zero, high=zero, low=zero, close=zero, adj_volume=zero, features=None
    )
    out = fn(ctx)
    assert out.shape == (n,)


def test_resolve_missing_alpha() -> None:
    with pytest.raises(ValueError, match="must define.*alpha"):
        resolve_alpha_from_source("x = 1\n")


def test_resolve_syntax_error() -> None:
    with pytest.raises(ValueError, match="syntax"):
        resolve_alpha_from_source("def oops(")


def test_resolve_alpha_not_callable() -> None:
    with pytest.raises(ValueError, match="callable"):
        resolve_alpha_from_source("alpha = 3\n")


def test_resolve_for_backtest_prefers_source() -> None:
    src = """
import jax.numpy as jnp
def alpha(ctx):
    return ctx.cs.rank(ctx.close)
"""
    fn = resolve_alpha_for_backtest("examples.alphas.sma_ratio_20:alpha", src)
    z = jnp.zeros((3, 2), dtype=jnp.float32)
    ctx = AlphaContext(
        open=z, high=z, low=z, close=z, adj_volume=z, features=None
    )
    out = fn(ctx)
    assert out.shape == (2,)


def test_resolve_for_backtest_uses_import_when_no_source() -> None:
    fn = resolve_alpha_for_backtest("examples.alphas.sma_ratio_20:alpha", None)
    assert callable(fn)
