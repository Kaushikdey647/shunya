"""
Cross-sectional helpers (WorldQuant Brain–style) for 1D slices across equities.

Each function expects ``x`` shaped ``(n_stocks,)``. Compose with panel columns, e.g.
``rank(panel[:, IX.LOG_RET])``.
"""

from __future__ import annotations

import jax.numpy as jnp

_EPS = 1e-12


def zscore(x: jnp.ndarray, *, eps: float = _EPS) -> jnp.ndarray:
    """
    Cross-sectional z-score: ``(x - mean) / (std + eps)``.
    Uses ``nanmean`` / ``nanstd``; non-finite inputs or zero std yield 0 on that coordinate.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    mu = jnp.nanmean(x)
    std = jnp.nanstd(x)
    finite = jnp.isfinite(x)
    use = finite & jnp.isfinite(mu) & jnp.isfinite(std) & (std > eps)
    out = jnp.where(use, (x - mu) / (std + eps), 0.0)
    return out.astype(jnp.float32)


def rank(x: jnp.ndarray) -> jnp.ndarray:
    """
    Cross-sectional rank normalized to approximately ``[0, 1]`` (smaller ``x`` → smaller rank).

    Ties follow ``argsort`` ordering (not average rank). Prefer NaN-free panels from
    :meth:`FinStrat.panel_at`; otherwise NaNs sort last.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    n = x.shape[0]
    order = jnp.argsort(x)
    ranks = jnp.empty_like(order, dtype=jnp.float32)
    ranks = ranks.at[order].set(jnp.arange(n, dtype=jnp.float32))
    denom = jnp.maximum(n - 1, 1)
    return (ranks / denom).astype(jnp.float32)
