from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Breakout-style momentum: rank(close / delay(close, 20)).
    """
    signal = ctx.close / ctx.ts.delay(ctx.close, 20)
    return ctx.cs.rank(signal)
