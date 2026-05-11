"""NumPy-only cross-section ops for CI / debugging (no JAX JIT)."""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-12


def _as_f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def winsorize(x: Any, tail: float) -> np.ndarray:
    if tail < 0 or tail >= 0.5:
        raise ValueError(f"tail must be in [0, 0.5), got {tail}")
    arr = _as_f32(x)
    if arr.size == 0:
        raise ValueError("winsorize: empty array")
    finite = np.isfinite(arr)
    if tail == 0:
        return np.where(finite, arr, 0.0).astype(np.float32)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    sub = arr[finite]
    lo = float(np.quantile(sub, tail))
    hi = float(np.quantile(sub, 1.0 - tail))
    clipped = np.clip(arr, lo, hi)
    return np.where(finite, clipped, 0.0).astype(np.float32)


def neutralize_market(s: Any) -> np.ndarray:
    s = _as_f32(s)
    finite = np.isfinite(s)
    s_f = np.where(finite, s, 0.0)
    cnt = float(np.sum(finite.astype(np.float32)))
    mu = np.sum(s_f) / max(cnt, 1.0) if cnt > 0 else 0.0
    return np.where(finite, s - mu, 0.0).astype(np.float32)


def neutralize_groups(s: Any, group_ids: Any) -> np.ndarray:
    s_j = _as_f32(s)
    g = np.asarray(group_ids)
    if g.shape != s_j.shape:
        raise ValueError(f"group_ids shape {g.shape} != scores shape {s_j.shape}")
    if g.size == 0:
        return s_j
    out = np.zeros_like(s_j, dtype=np.float32)
    for u in np.unique(g):
        m = g == u
        block = s_j[m]
        fin = np.isfinite(block)
        if not fin.any():
            continue
        mu = float(np.mean(block[fin]))
        out[m] = np.where(np.isfinite(s_j[m]), s_j[m] - mu, 0.0).astype(np.float32)
    return out


class NumpyCrossSectionOps:
    __slots__ = ()

    def winsorize(self, x: Any, tail: float) -> Any:
        return winsorize(x, tail)

    def neutralize_market(self, x: Any) -> Any:
        return neutralize_market(x)

    def neutralize_groups(self, x: Any, group_ids: Any) -> Any:
        return neutralize_groups(x, group_ids)
