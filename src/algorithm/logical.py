"""
Logical operators for alpha construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def if_else(
    condition: jnp.ndarray | np.ndarray | bool,
    x: jnp.ndarray | np.ndarray | float,
    y: jnp.ndarray | np.ndarray | float,
) -> jnp.ndarray:
    """Elementwise conditional: ``x`` where ``condition`` else ``y``."""
    return jnp.where(jnp.asarray(condition, dtype=bool), jnp.asarray(x), jnp.asarray(y))


def logical_and(a: jnp.ndarray | np.ndarray, b: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    return jnp.logical_and(jnp.asarray(a, dtype=bool), jnp.asarray(b, dtype=bool))


def logical_or(a: jnp.ndarray | np.ndarray, b: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    return jnp.logical_or(jnp.asarray(a, dtype=bool), jnp.asarray(b, dtype=bool))


def logical_not(a: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    return jnp.logical_not(jnp.asarray(a, dtype=bool))


def trade_when(
    condition: jnp.ndarray | np.ndarray,
    alpha: jnp.ndarray | np.ndarray,
    otherwise: jnp.ndarray | np.ndarray | float = 0.0,
    *,
    exit_condition: jnp.ndarray | np.ndarray | None = None,
) -> jnp.ndarray:
    """
    Trade gating helper.

    - Without ``exit_condition``: returns ``where(condition, alpha, otherwise)``.
    - With ``exit_condition``: maintains an active state per series over time
      (axis 0). Entry activates, exit deactivates, and output uses ``alpha`` while
      active; otherwise ``otherwise``.
    """
    cond = np.asarray(condition, dtype=bool)
    a = np.asarray(alpha, dtype=float)
    oth = np.asarray(otherwise, dtype=float)
    if exit_condition is None:
        return jnp.where(jnp.asarray(cond), jnp.asarray(a), jnp.asarray(oth)).astype(
            jnp.float32
        )

    ex = np.asarray(exit_condition, dtype=bool)
    if cond.shape != ex.shape:
        raise ValueError(
            f"condition and exit_condition must have same shape, got {cond.shape} and {ex.shape}"
        )
    if a.shape != cond.shape:
        raise ValueError(f"alpha shape {a.shape} must match condition shape {cond.shape}")
    if oth.shape not in {(), cond.shape}:
        raise ValueError(
            f"otherwise must be scalar or same shape as condition, got {oth.shape}"
        )
    oth_full = np.full(cond.shape, float(oth), dtype=float) if oth.shape == () else oth
    if cond.ndim == 1:
        cond = cond[:, None]
        ex = ex[:, None]
        a = a[:, None]
        oth_full = oth_full[:, None]
        squeeze = True
    elif cond.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected 1D or 2D inputs for stateful trade_when, got {cond.shape}")

    n, m = cond.shape
    out = np.empty((n, m), dtype=float)
    active = np.zeros(m, dtype=bool)
    for t in range(n):
        active = np.where(ex[t], False, active)
        active = np.where(cond[t], True, active)
        out[t] = np.where(active, a[t], oth_full[t])
    if squeeze:
        out = out[:, 0]
    return jnp.asarray(out, dtype=jnp.float32)

