"""Share-based reconciliation: target vs settled position and working orders."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence


def required_delta_shares(
    target_shares: Mapping[str, float],
    settled_shares: Mapping[str, float],
    working_buy_shares: Mapping[str, float],
    working_sell_shares: Mapping[str, float],
    universe: Sequence[str],
) -> Dict[str, float]:
    """
    Institutional delta per symbol:

    ``D_k = T_k - (P_k + W_k^buy - W_k^sell)``.

    Positive ``D_k`` implies net buy need; negative implies net sell need.
    """
    out: Dict[str, float] = {}
    for k in universe:
        sym = str(k)
        t_k = float(target_shares.get(sym, 0.0))
        p_k = float(settled_shares.get(sym, 0.0))
        w_b = float(working_buy_shares.get(sym, 0.0))
        w_s = float(working_sell_shares.get(sym, 0.0))
        out[sym] = t_k - (p_k + w_b - w_s)
    return out


def usd_targets_to_share_targets(
    targets_usd: Mapping[str, float],
    prices: Mapping[str, float],
    universe: Sequence[str],
) -> Dict[str, float]:
    """
    Convert vetted USD notionals to *floating* share targets using last prices.

    Callers that require whole-share targets should floor/round with their own
    :class:`~shunya.algorithm.orders.RiskPolicy` (lot size) before submission.
    """
    out: Dict[str, float] = {}
    for sym in universe:
        s = str(sym)
        px = float(prices.get(s, 0.0))
        usd = float(targets_usd.get(s, 0.0))
        if px <= 0.0:
            out[s] = 0.0
        else:
            out[s] = usd / px
    return out
