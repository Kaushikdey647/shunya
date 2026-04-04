"""
Time-series operator helpers for alpha research.

Functions accept 1D ``(time,)`` or 2D ``(time, n_series)`` arrays and return
``float32`` JAX arrays. Rolling operators emit ``nan`` until enough lookback
history is available.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

_EPS = 1e-12
_TsRegRetval = Literal["error", "a", "b", "estimate"]


def _as_2d(x: jnp.ndarray | np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def _restore(arr: np.ndarray, squeeze: bool) -> jnp.ndarray:
    out = arr[:, 0] if squeeze else arr
    return jnp.asarray(out, dtype=jnp.float32)


def _validate_window(window: int) -> None:
    if int(window) != window or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")


def _validate_lag(lag: int) -> None:
    if int(lag) != lag or lag < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag!r}")


def tsdelay(x: jnp.ndarray | np.ndarray, lag: int) -> jnp.ndarray:
    _validate_lag(lag)
    arr, squeeze = _as_2d(x)
    out = np.full_like(arr, np.nan, dtype=float)
    if lag == 0:
        out = arr.copy()
    elif lag < arr.shape[0]:
        out[lag:, :] = arr[:-lag, :]
    return _restore(out, squeeze)


def tsdelta(x: jnp.ndarray | np.ndarray, lag: int) -> jnp.ndarray:
    arr, squeeze = _as_2d(x)
    out = arr - np.asarray(tsdelay(arr, lag), dtype=float).reshape(arr.shape)
    return _restore(out, squeeze)


def tssum(x: jnp.ndarray | np.ndarray, window: int) -> jnp.ndarray:
    _validate_window(window)
    arr, squeeze = _as_2d(x)
    out = np.full_like(arr, np.nan, dtype=float)
    n, m = arr.shape
    for j in range(m):
        col = arr[:, j]
        for t in range(window - 1, n):
            vals = col[t - window + 1 : t + 1]
            if np.isfinite(vals).all():
                out[t, j] = float(np.sum(vals))
    return _restore(out, squeeze)


def tsmean(x: jnp.ndarray | np.ndarray, window: int) -> jnp.ndarray:
    return tssum(x, window) / float(window)


def tsstddev(
    x: jnp.ndarray | np.ndarray, window: int, *, eps: float = _EPS
) -> jnp.ndarray:
    _validate_window(window)
    arr, squeeze = _as_2d(x)
    out = np.full_like(arr, np.nan, dtype=float)
    n, m = arr.shape
    for j in range(m):
        col = arr[:, j]
        for t in range(window - 1, n):
            vals = col[t - window + 1 : t + 1]
            if np.isfinite(vals).all():
                std = float(np.std(vals))
                out[t, j] = std if std > eps else 0.0
    return _restore(out, squeeze)


def tszscore(
    x: jnp.ndarray | np.ndarray, window: int, *, eps: float = _EPS
) -> jnp.ndarray:
    arr, squeeze = _as_2d(x)
    mean = np.asarray(tsmean(arr, window), dtype=float).reshape(arr.shape)
    std = np.asarray(tsstddev(arr, window, eps=eps), dtype=float).reshape(arr.shape)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr) & np.isfinite(mean) & np.isfinite(std)
    nonzero = std > eps
    use = valid & nonzero
    out[use] = (arr[use] - mean[use]) / std[use]
    out[valid & ~nonzero] = 0.0
    return _restore(out, squeeze)


def tsrank(x: jnp.ndarray | np.ndarray, window: int) -> jnp.ndarray:
    _validate_window(window)
    arr, squeeze = _as_2d(x)
    out = np.full_like(arr, np.nan, dtype=float)
    n, m = arr.shape
    for j in range(m):
        col = arr[:, j]
        for t in range(window - 1, n):
            vals = col[t - window + 1 : t + 1]
            last = vals[-1]
            if not np.isfinite(last):
                continue
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            less = float(np.sum(finite < last))
            equal = float(np.sum(finite == last))
            if finite.size <= 1:
                out[t, j] = 0.0
            else:
                out[t, j] = (less + 0.5 * (equal - 1.0)) / float(finite.size - 1)
    return _restore(out, squeeze)


def tsregression(
    y: jnp.ndarray | np.ndarray,
    x: jnp.ndarray | np.ndarray,
    window: int,
    lag: int = 0,
    retval: _TsRegRetval = "b",
    *,
    eps: float = _EPS,
) -> jnp.ndarray:
    _validate_window(window)
    _validate_lag(lag)
    if retval not in {"error", "a", "b", "estimate"}:
        raise ValueError("retval must be one of: 'error', 'a', 'b', 'estimate'")
    y2, squeeze = _as_2d(y)
    x2, squeeze_x = _as_2d(x)
    if squeeze != squeeze_x or y2.shape != x2.shape:
        raise ValueError(f"y and x must have matching shapes, got {y2.shape} and {x2.shape}")
    x_lag = np.asarray(tsdelay(x2, lag), dtype=float).reshape(x2.shape)
    out = np.full_like(y2, np.nan, dtype=float)
    n, m = y2.shape
    for j in range(m):
        yc = y2[:, j]
        xc = x_lag[:, j]
        for t in range(window - 1, n):
            wy = yc[t - window + 1 : t + 1]
            wx = xc[t - window + 1 : t + 1]
            mask = np.isfinite(wy) & np.isfinite(wx)
            if int(mask.sum()) < 2:
                continue
            yv = wy[mask]
            xv = wx[mask]
            x_mu = float(np.mean(xv))
            y_mu = float(np.mean(yv))
            varx = float(np.mean((xv - x_mu) ** 2))
            if varx <= eps:
                continue
            cov = float(np.mean((xv - x_mu) * (yv - y_mu)))
            b = cov / varx
            a = y_mu - b * x_mu
            est = a + b * xc[t] if np.isfinite(xc[t]) else np.nan
            err = yc[t] - est if np.isfinite(est) and np.isfinite(yc[t]) else np.nan
            if retval == "a":
                out[t, j] = a
            elif retval == "b":
                out[t, j] = b
            elif retval == "estimate":
                out[t, j] = est
            else:
                out[t, j] = err
    return _restore(out, squeeze)


def humpdecay(x: jnp.ndarray | np.ndarray, hump: float) -> jnp.ndarray:
    if hump < 0:
        raise ValueError(f"hump must be non-negative, got {hump}")
    arr, squeeze = _as_2d(x)
    out = np.full_like(arr, np.nan, dtype=float)
    if arr.shape[0] == 0:
        return _restore(out, squeeze)
    out[0, :] = arr[0, :]
    n, m = arr.shape
    for j in range(m):
        col = arr[:, j]
        for t in range(1, n):
            today = col[t]
            yesterday = col[t - 1]
            if not np.isfinite(today) or not np.isfinite(yesterday):
                out[t, j] = today
                continue
            out[t, j] = today if abs(today - yesterday) > hump else yesterday
    return _restore(out, squeeze)

