from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Cross-sectional rank of -(close - SMA20), mapped to ~[-1, 1].

    Long names most below the SMA; short names most above.
    """
    sma_20 = ctx.ts.mean(ctx.close, 20)
    spread = ctx.close - sma_20
    return -ctx.cs.zscore(spread)
