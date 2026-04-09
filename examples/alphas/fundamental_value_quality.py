from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Blend price trend with fundamental quality and valuation.
    """
    trend = ctx.close / ctx.ts.mean(ctx.close, 20)
    roe = ctx.feature("Return_On_Equity")
    pe = ctx.feature("Price_To_Earnings")
    signal = ctx.ts.zscore(trend, 20) + ctx.ts.zscore(roe, 4) - ctx.ts.zscore(pe, 4)
    return ctx.cs.rank(signal)
