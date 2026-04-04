"""
Cross-sectional helpers (WorldQuant Brain–style) for 1D slices across equities.

Each function expects ``x`` shaped ``(n_stocks,)``. Compose with panel columns, e.g.
``rank(panel[:, IX.LOG_RET])``.

Pure transforms use :func:`jax.jit` (with static hyperparameters where useful) so
repeated calls amortize XLA compilation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum

_EPS = 1e-12


@partial(jax.jit, static_argnames=("eps",))
def zscore(x: jnp.ndarray, *, eps: float = _EPS) -> jnp.ndarray:
    """
    Cross-sectional z-score: ``(x - mean) / (std + eps)``.
    Uses ``nanmean`` / ``nanstd``; non-finite inputs or zero std yield 0 on that coordinate.

    JIT-compiled; ``eps`` is a static compile-time axis.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    mu = jnp.nanmean(x)
    std = jnp.nanstd(x)
    finite = jnp.isfinite(x)
    use = finite & jnp.isfinite(mu) & jnp.isfinite(std) & (std > eps)
    out = jnp.where(use, (x - mu) / (std + eps), 0.0)
    return out.astype(jnp.float32)


@partial(jax.jit, static_argnames=("target", "eps"))
def scale(x: jnp.ndarray, *, target: float = 1.0, eps: float = _EPS) -> jnp.ndarray:
    """
    Cross-sectional gross scaling to ``sum(abs(out)) == target`` when possible.

    Non-finite inputs are treated as zero contribution. If gross exposure is below
    ``eps``, returns an all-zero vector.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    finite = jnp.isfinite(x)
    clean = jnp.where(finite, x, 0.0)
    gross = jnp.sum(jnp.abs(clean))
    t = jnp.asarray(target, dtype=jnp.float32)
    return jnp.where(gross > eps, clean / gross * t, jnp.zeros_like(clean)).astype(
        jnp.float32
    )


@jax.jit
def sign(x: jnp.ndarray) -> jnp.ndarray:
    """
    Elementwise sign in ``{-1, 0, 1}`` (non-finite inputs map to ``0``).
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    finite = jnp.isfinite(x)
    return jnp.where(finite, jnp.sign(x), 0.0).astype(jnp.float32)


@partial(jax.jit, static_argnames=("tail",))
def _winsorize_jit(x: jnp.ndarray, tail: float) -> jnp.ndarray:
    lo = jnp.quantile(x, tail, method="linear")
    hi = jnp.quantile(x, 1.0 - tail, method="linear")
    return jnp.clip(x, lo, hi).astype(jnp.float32)


def winsorize(x: jnp.ndarray, tail: float) -> jnp.ndarray:
    """
    Symmetric winsorization (BRAIN-style **truncation** on the cross-section).

    Clips ``x`` to ``[quantile(tail), quantile(1 - tail)]``. ``tail`` in ``[0, 0.5)``;
    ``tail == 0`` returns ``x`` unchanged. Empty slices and ``tail >= 0.5`` raise.

    The numerical core is JIT-compiled; ``tail`` is a static compile-time axis.
    """
    if tail < 0 or tail >= 0.5:
        raise ValueError(f"tail must be in [0, 0.5), got {tail}")
    x = jnp.asarray(x, dtype=jnp.float32)
    if tail == 0:
        return x
    if x.size == 0:
        raise ValueError("winsorize: empty array")
    return _winsorize_jit(x, tail)


@jax.jit
def neutralize_market(s: jnp.ndarray) -> jnp.ndarray:
    """Subtract cross-sectional mean (market-neutralize scores). JIT-compiled."""
    s = jnp.asarray(s, dtype=jnp.float32)
    return (s - jnp.mean(s)).astype(jnp.float32)


@partial(jax.jit, static_argnames=("num_segments",))
def _neutralize_groups_jit(
    s: jnp.ndarray, inv: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    inv = jnp.asarray(inv, dtype=jnp.int32)
    ones = jnp.ones(s.shape[0], dtype=jnp.float32)
    counts = segment_sum(ones, inv, num_segments=num_segments)
    gsum = segment_sum(s, inv, num_segments=num_segments)
    means = gsum / jnp.maximum(counts, 1.0e-12)
    return (s - means[inv]).astype(jnp.float32)


def neutralize_groups(s: jnp.ndarray, group_ids: jnp.ndarray) -> jnp.ndarray:
    """
    Demean scores within each group. ``group_ids`` same shape as ``s``.

    Labels may be any hashable type; they are mapped to dense indices with
    :func:`numpy.unique`, then the demeaning step is JIT-compiled via ``segment_sum``.
    """
    s_j = jnp.asarray(s, dtype=jnp.float32)
    g = np.asarray(group_ids)
    if g.shape != s_j.shape:
        raise ValueError(f"group_ids shape {g.shape} != scores shape {s_j.shape}")
    if g.size == 0:
        return s_j
    _u, inv = np.unique(g, return_inverse=True)
    del _u
    num_segments = int(inv.max()) + 1
    return _neutralize_groups_jit(s_j, jnp.asarray(inv, jnp.int32), num_segments)


@jax.jit
def rank(x: jnp.ndarray) -> jnp.ndarray:
    """
    Cross-sectional rank normalized to approximately ``[0, 1]``, **monotone in** ``x``:
    **smallest** raw value → **0**, **largest** → **1** (ties: ``argsort`` order).

    Therefore ``rank(-x)`` **reverses** who gets high vs low scores. With **distinct**
    values and ``n \\ge 2``, ``rank(-x) = 1 - rank(x)`` elementwise. Dollar weights after
    market-demeaning in :meth:`~src.algorithm.finstrat.FinStrat.pass_` flip sign when
    you swap ``rank(raw)`` for ``rank(-raw)``; if results look identical, re-run the
    cell or confirm you are comparing ``scores`` / ``pass_`` after the change.

    Prefer NaN-free inputs; otherwise NaNs sort last. JIT-compiled.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    n = x.shape[0]
    order = jnp.argsort(x)
    ranks = jnp.empty_like(order, dtype=jnp.float32)
    ranks = ranks.at[order].set(jnp.arange(n, dtype=jnp.float32))
    denom = jnp.maximum(n - 1, 1)
    return (ranks / denom).astype(jnp.float32)
