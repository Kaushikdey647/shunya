from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Trend alpha: rank(close / SMA50).
    """
    sma_20 = ctx.ts.mean(ctx.close, 20)
    signal = ctx.close / sma_20
    return ctx.cs.rank(signal)

