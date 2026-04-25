from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Short-horizon mean reversion: rank(-(close / SMA5 - 1)).
    """
    sma_20 = ctx.ts.mean(ctx.close, 20)
    deviation = (ctx.close / sma_20) - 1.0
    return ctx.cs.rank(-deviation)

