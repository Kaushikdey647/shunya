from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Volume scaled by cross-sectional rank of the bar's (close - open).

    High rank means the stock moved up more vs peers on that bar; multiplying by
    ``adj_volume`` emphasizes names with both strong intraday direction and liquidity.
    """
    intraday = ctx.close - ctx.open
    return ctx.adj_volume * ctx.cs.rank(intraday)
