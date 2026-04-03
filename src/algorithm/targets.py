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


def apply_group_net_cap(
    targets: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    max_group_net_fraction: float,
    on_breach: Literal["rescale", "raise"] = "rescale",
) -> Tuple[Dict[str, float], List[str]]:
    """
    Enforce per-group absolute net cap as fraction of total gross.
    """
    if not (0.0 < max_group_net_fraction <= 1.0):
        raise ValueError("max_group_net_fraction must be in (0, 1]")
    if on_breach not in ("rescale", "raise"):
        raise ValueError("on_breach must be 'rescale' or 'raise'")

    adjusted = {k: float(v) for k, v in targets.items()}
    total_gross = sum(abs(v) for v in adjusted.values())
    if total_gross <= 0.0:
        return adjusted, []
    cap_abs = total_gross * max_group_net_fraction

    net_by_group: Dict[str, float] = {}
    group_names: Dict[str, List[str]] = {}
    gross_by_group: Dict[str, float] = {}
    for sym, val in adjusted.items():
        g = str(groups.get(sym, "UnknownSector"))
        net_by_group[g] = net_by_group.get(g, 0.0) + val
        gross_by_group[g] = gross_by_group.get(g, 0.0) + abs(val)
        group_names.setdefault(g, []).append(sym)

    breached = sorted([g for g, nv in net_by_group.items() if abs(nv) > cap_abs])
    if not breached:
        return adjusted, []
    if on_breach == "raise":
        raise ValueError(
            f"Group net cap breached for groups {breached}; cap_abs={cap_abs:.2f}, net_by_group={net_by_group}"
        )

    for g in breached:
        net = net_by_group[g]
        if net == 0.0:
            continue
        group_gross = gross_by_group.get(g, 0.0)
        rest_gross = max(total_gross - group_gross, 0.0)
        denom = abs(net) - max_group_net_fraction * group_gross
        if denom <= 0:
            k = cap_abs / abs(net)
        else:
            k = (max_group_net_fraction * rest_gross) / denom
        k = float(np.clip(k, 0.0, 1.0))
        for s in group_names[g]:
            adjusted[s] *= k
    return adjusted, breached


def enforce_turnover_budget(
    targets: Mapping[str, float],
    current_usd: Mapping[str, float],
    *,
    max_turnover_fraction: float,
    on_breach: Literal["rescale", "raise"] = "rescale",
) -> Tuple[Dict[str, float], float, float]:
    """
    Enforce turnover budget where turnover = sum(abs(target-current)).
    """
    if not (0.0 < max_turnover_fraction <= 2.0):
        raise ValueError("max_turnover_fraction must be in (0, 2]")
    if on_breach not in ("rescale", "raise"):
        raise ValueError("on_breach must be 'rescale' or 'raise'")

    adjusted = {k: float(v) for k, v in targets.items()}
    gross_target = sum(abs(v) for v in adjusted.values())
    limit = gross_target * max_turnover_fraction
    turnover = sum(abs(adjusted.get(k, 0.0) - float(current_usd.get(k, 0.0))) for k in adjusted)
    if turnover <= limit or gross_target <= 0.0:
        return adjusted, float(turnover), float(limit)
    if on_breach == "raise":
        raise ValueError(f"Turnover budget breached turnover={turnover:.2f} limit={limit:.2f}")

    k = limit / turnover if turnover > 0 else 1.0
    for sym in adjusted:
        cur = float(current_usd.get(sym, 0.0))
        adjusted[sym] = cur + (adjusted[sym] - cur) * k
    return adjusted, float(turnover), float(limit)


def cap_deltas_by_adv(
    deltas: Mapping[str, float],
    adv_usd: Mapping[str, float],
    *,
    max_adv_fraction: float,
    on_breach: Literal["rescale", "raise"] = "rescale",
) -> Tuple[Dict[str, float], List[str]]:
    """
    Cap order deltas by ADV participation fraction.
    """
    if not (0.0 < max_adv_fraction <= 1.0):
        raise ValueError("max_adv_fraction must be in (0, 1]")
    if on_breach not in ("rescale", "raise"):
        raise ValueError("on_breach must be 'rescale' or 'raise'")
    out = {k: float(v) for k, v in deltas.items()}
    breached: List[str] = []
    for sym, d in out.items():
        adv = float(adv_usd.get(sym, 0.0))
        cap_abs = adv * max_adv_fraction
        if cap_abs <= 0:
            continue
        if abs(d) > cap_abs:
            breached.append(sym)
            if on_breach == "raise":
                raise ValueError(
                    f"ADV cap breached for {sym}: abs(delta)={abs(d):.2f} cap={cap_abs:.2f}"
                )
            out[sym] = cap_abs if d > 0 else -cap_abs
    return out, breached
