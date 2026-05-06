from __future__ import annotations

"""
Zaremba, Umutlu & Karathanasopoulos (2019), *Alpha Momentum and Alpha Reversal in
Country and Industry Equity Indexes* (SSRN 3235350, J. Empirical Finance).

The paper shows that past short-term alphas positively predict future returns while
past long-term alphas negatively predict them. This example adapts that idea to a
stock cross-section: rolling CAPM intercepts vs an equal-weight market return proxy
computed each day across names.

WorldQuant BRAIN (FASTEXPR) notes
---------------------------------
``group_mean(x, weight, group)`` is documented as a **harmonic** (optionally
weighted) group aggregate—well suited to **ratios** (e.g.
``group_mean(close / eps, 1, densify(industry))`` for industry P/E). It is **not**
the same object as the **arithmetic** equal-weight mean of returns used below.

For a BRAIN-style CAPM line close to common templates, prefer a **defensible**
market proxy such as a cap-smoothed value weight, e.g.
``mkt = group_mean(returns, ts_mean(cap, 21), densify(market))``, then
``ts_regression(returns, mkt, 21, lag=0, rettype=1)`` (α) and the long window
analog with ``rettype=1``. Wrap ``group`` with ``densify(...)`` when simulation
is slow or groups are sparse. Adjust windows to match your universe and delay.
"""

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Combine short-horizon CAPM alpha (momentum) minus long-horizon CAPM alpha
    (reversal), then cross-sectionally rank.
    """
    ret = ctx.close / ctx.ts.delay(ctx.close, 1) - 1.0
    # Arithmetic equal-weight market return each day (not BRAIN's harmonic group_mean).
    mkt = jnp.mean(ret.data, axis=1, keepdims=True)
    mkt_xs = jnp.broadcast_to(mkt, ret.data.shape)
    short_w, long_w = 21, 252
    a_short = ctx.ts.regression(ret, mkt_xs, short_w, retval="a")
    a_long = ctx.ts.regression(ret, mkt_xs, long_w, retval="a")
    combo = ctx.cs.rank(a_short) - ctx.cs.rank(a_long)
    return ctx.cs.rank(combo)
