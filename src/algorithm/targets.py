"""Shared portfolio target and broker-delta helpers (backtest vs live parity)."""

from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np


def target_usd_universe(
    tickers_in_panel: Sequence[str],
    pass_notionals: Union[jnp.ndarray, np.ndarray, Sequence[float]],
    universe: Sequence[str],
) -> Dict[str, float]:
    """
    Map :meth:`FinStrat.pass_` output (aligned with ``tickers_in_panel``) to a full
    universe dict; missing names get ``0.0``.
    """
    vec = np.asarray(jnp.asarray(pass_notionals), dtype=float).reshape(-1)
    if len(tickers_in_panel) != vec.shape[0]:
        raise ValueError("pass_notionals length must match tickers_in_panel")
    out: Dict[str, float] = {t: 0.0 for t in universe}
    for i, t in enumerate(tickers_in_panel):
        if t in out:
            out[t] = float(vec[i])
    return out


def scale_signed_targets_to_gross_cap(
    targets: Mapping[str, float],
    gross_cap: float,
) -> Dict[str, float]:
    """
    Scale all signed targets by a constant so ``sum(abs(v)) <= gross_cap``.

    If current gross is 0 or already within ``gross_cap``, returns a plain ``dict`` copy.
    """
    if gross_cap < 0:
        raise ValueError(f"gross_cap must be non-negative, got {gross_cap}")
    gross = sum(abs(float(v)) for v in targets.values())
    if gross <= gross_cap or gross == 0.0:
        return {k: float(v) for k, v in targets.items()}
    k = gross_cap / gross
    return {sym: float(val) * k for sym, val in targets.items()}


def broker_deltas(
    targets: Mapping[str, float],
    current_usd: Mapping[str, float],
    universe: Sequence[str],
) -> Dict[str, float]:
    """
    Per-symbol dollar delta: ``target_usd - current_market_value_usd`` (signed).
    """
    return {t: float(targets[t]) - float(current_usd.get(t, 0.0)) for t in universe}


def apply_slippage_to_fill_price(
    price: float,
    *,
    side_is_buy: bool,
    slippage_pct: float,
) -> float:
    """
    Adverse slippage: buys pay more, sells receive less (for symmetry in bps terms).

    ``slippage_pct`` is a non-negative fraction, e.g. ``0.001`` for 10 bps per side.
    """
    if slippage_pct < 0:
        raise ValueError("slippage_pct must be non-negative")
    if side_is_buy:
        return float(price) * (1.0 + slippage_pct)
    return float(price) * (1.0 - slippage_pct)


def apply_group_gross_cap(
    targets: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    max_group_gross_fraction: float,
    on_breach: Literal["rescale", "raise"] = "rescale",
) -> Tuple[Dict[str, float], List[str]]:
    """
    Enforce per-group gross cap as a fraction of total portfolio gross.

    Returns ``(adjusted_targets, breached_groups)``.
    """
    if not (0.0 < max_group_gross_fraction <= 1.0):
        raise ValueError("max_group_gross_fraction must be in (0, 1]")
    if on_breach not in ("rescale", "raise"):
        raise ValueError("on_breach must be 'rescale' or 'raise'")

    adjusted = {k: float(v) for k, v in targets.items()}
    total_gross = sum(abs(v) for v in adjusted.values())
    if total_gross <= 0.0:
        return adjusted, []

    cap_abs = total_gross * max_group_gross_fraction
    group_gross: Dict[str, float] = {}
    group_names: Dict[str, List[str]] = {}
    for sym, val in adjusted.items():
        g = str(groups.get(sym, "UnknownSector"))
        group_gross[g] = group_gross.get(g, 0.0) + abs(val)
        group_names.setdefault(g, []).append(sym)

    breached = sorted([g for g, gv in group_gross.items() if gv > cap_abs])
    if not breached:
        return adjusted, []

    if on_breach == "raise":
        raise ValueError(
            f"Group gross cap breached for groups {breached}; "
            f"cap_abs={cap_abs:.2f}, group_gross={group_gross}"
        )

    for g in breached:
        gg = group_gross[g]
        if gg <= 0.0:
            continue
        k = cap_abs / gg
        for sym in group_names[g]:
            adjusted[sym] *= k
    return adjusted, breached
