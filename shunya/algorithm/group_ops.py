"""
Group-wise cross-sectional operators.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import cross_section

_EPS = 1e-12


def _validate_shapes(x: np.ndarray, g: np.ndarray) -> None:
    if x.shape != g.shape:
        raise ValueError(f"group_ids shape {g.shape} must match x shape {x.shape}")
    if x.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got shape {x.shape}")


def group_mean(x: jnp.ndarray, group_ids: jnp.ndarray) -> jnp.ndarray:
    x_np = np.asarray(x, dtype=float)
    g_np = np.asarray(group_ids)
    _validate_shapes(x_np, g_np)
    out = np.zeros_like(x_np, dtype=float)
    for gid in np.unique(g_np):
        m = g_np == gid
        vals = x_np[m]
        finite = np.isfinite(vals)
        if not finite.any():
            out[m] = 0.0
        else:
            mu = float(np.mean(vals[finite]))
            out[m] = mu
    return jnp.asarray(out, dtype=jnp.float32)


def group_neutralize(x: jnp.ndarray, group_ids: jnp.ndarray) -> jnp.ndarray:
    return cross_section.neutralize_groups(x, group_ids)


def group_zscore(x: jnp.ndarray, group_ids: jnp.ndarray, *, eps: float = _EPS) -> jnp.ndarray:
    x_np = np.asarray(x, dtype=float)
    g_np = np.asarray(group_ids)
    _validate_shapes(x_np, g_np)
    out = np.zeros_like(x_np, dtype=float)
    for gid in np.unique(g_np):
        m = g_np == gid
        vals = x_np[m]
        finite = np.isfinite(vals)
        if not finite.any():
            out[m] = 0.0
            continue
        mu = float(np.mean(vals[finite]))
        std = float(np.std(vals[finite]))
        if std <= eps:
            out[m] = 0.0
            continue
        out[m] = np.where(finite, (vals - mu) / std, 0.0)
    return jnp.asarray(out, dtype=jnp.float32)


def group_rank(x: jnp.ndarray, group_ids: jnp.ndarray) -> jnp.ndarray:
    x_np = np.asarray(x, dtype=float)
    g_np = np.asarray(group_ids)
    _validate_shapes(x_np, g_np)
    out = np.zeros_like(x_np, dtype=float)
    for gid in np.unique(g_np):
        m = g_np == gid
        vals = x_np[m]
        n = vals.size
        if n <= 1:
            out[m] = 0.0
            continue
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        out[m] = ranks / float(n - 1)
    return jnp.asarray(out, dtype=jnp.float32)

