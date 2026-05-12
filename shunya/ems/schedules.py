"""TWAP and VWAP slice schedules (share quantities per time bin)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def twap_slice_quantities(total_qty: int, n_bins: int, *, lot_size: int = 1) -> List[int]:
    """
    Split ``total_qty`` across ``n_bins`` bins as evenly as possible (integer shares).

    Residual from floor division is spread across the first bins. Each bin is
    then floored to ``lot_size``; remainder is folded into the last bin so the
    sum matches ``total_qty`` when possible.
    """
    if total_qty < 0:
        raise ValueError("total_qty must be non-negative")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    lot = max(1, int(lot_size))
    if total_qty == 0:
        return [0] * n_bins
    base = total_qty // n_bins
    rem = total_qty % n_bins
    raw = [base + (1 if i < rem else 0) for i in range(n_bins)]
    adjusted = [(q // lot) * lot for q in raw]
    diff = total_qty - sum(adjusted)
    if diff != 0 and adjusted:
        adjusted[-1] = adjusted[-1] + diff
    return adjusted


def vwap_slice_quantities(
    total_qty: int,
    volume_profile: Sequence[float],
    *,
    lot_size: int = 1,
) -> List[int]:
    """
    Allocate ``total_qty`` across bins proportionally to ``volume_profile``.

    The profile is normalized to sum to 1; per-bin float weights are rounded with
    largest-remainder so integers sum to ``total_qty``.
    """
    if total_qty < 0:
        raise ValueError("total_qty must be non-negative")
    prof = np.asarray([max(0.0, float(x)) for x in volume_profile], dtype=float)
    n = int(prof.shape[0])
    if n == 0:
        raise ValueError("volume_profile must be non-empty")
    s = float(np.sum(prof))
    if s <= 0.0:
        return twap_slice_quantities(total_qty, n, lot_size=lot_size)
    w = prof / s
    float_shares = w * float(total_qty)
    floors = np.floor(float_shares).astype(int)
    rem = total_qty - int(np.sum(floors))
    frac = float_shares - floors
    order = np.argsort(-frac)
    out = floors.copy()
    for i in range(rem):
        out[order[i % n]] += 1
    lot = max(1, int(lot_size))
    adjusted = [(int(x) // lot) * lot for x in out.tolist()]
    diff = total_qty - sum(adjusted)
    if diff != 0 and adjusted:
        adjusted[-1] = adjusted[-1] + diff
    return adjusted


def smooth_volume_profile_jax(profile: Sequence[float], *, sigma_bins: float = 1.5) -> np.ndarray:
    """
    Optional JAX Gaussian smoothing for intraday volume profiles (``sigma_bins`` in bin units).

    Falls back to a light NumPy moving average when JAX is unavailable.
    """
    p = np.asarray([max(0.0, float(x)) for x in profile], dtype=float)
    if p.size == 0:
        return p
    try:
        import jax
        import jax.numpy as jnp

        x = jnp.asarray(p, dtype=jnp.float32)
        n = int(x.shape[0])
        idx = jnp.arange(n, dtype=jnp.float32)
        sigma = max(float(sigma_bins), 1e-6)

        def smooth_at(i: jnp.ndarray) -> jnp.ndarray:
            w = jnp.exp(-0.5 * ((idx - i) / sigma) ** 2)
            return jnp.dot(w, x) / jnp.maximum(jnp.sum(w), 1e-12)

        y = jax.vmap(smooth_at)(idx)
        out = np.asarray(y, dtype=float)
        s0 = float(np.sum(p))
        s1 = float(np.sum(out))
        if s1 > 0 and s0 > 0:
            out = out / s1 * s0
        return out
    except Exception:  # noqa: BLE001
        k = np.array([0.25, 0.5, 0.25], dtype=float)
        pad = np.pad(p, (1, 1), mode="edge")
        sm = np.convolve(pad, k, mode="valid")
        s0 = float(np.sum(p))
        s1 = float(np.sum(sm))
        if s1 > 0 and s0 > 0:
            sm = sm / s1 * s0
        return sm
