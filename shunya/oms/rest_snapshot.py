"""REST snapshot helpers to reconcile OMS state when the stream (re)connects."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import Order


_TERMINAL_ORDER_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "failed",
        "done_for_day",
    }
)


def working_shares_from_orders(orders: Sequence[object]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Aggregate remaining broker order quantity into buy vs sell working shares.

    Uses ``qty - filled_qty`` for orders whose status is not a known terminal status.
    """
    buy: DefaultDict[str, float] = defaultdict(float)
    sell: DefaultDict[str, float] = defaultdict(float)
    for o in orders:
        st = str(getattr(o, "status", "") or "").lower()
        if st in _TERMINAL_ORDER_STATUSES:
            continue
        qty = float(getattr(o, "qty", 0.0) or 0.0)
        filled = float(getattr(o, "filled_qty", 0.0) or 0.0)
        rem = max(0.0, qty - filled)
        if rem <= 0.0:
            continue
        sym = str(getattr(o, "symbol", "") or "")
        if not sym:
            continue
        side = str(getattr(o, "side", "") or "").upper()
        if "BUY" in side:
            buy[sym] += rem
        else:
            sell[sym] += rem
    return dict(buy), dict(sell)


def settled_shares_from_positions(positions: Mapping[str, float]) -> Dict[str, float]:
    """Normalize Alpaca ``get_all_positions()`` style maps to float shares."""
    return {str(k): float(v) for k, v in positions.items()}


def snapshot_from_alpaca_client(client: "TradingClient") -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Return ``(settled_shares, working_buy, working_sell)`` from live REST data.

    Settled uses position ``qty``; working uses open orders only.
    """
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    settled: Dict[str, float] = {}
    for p in client.get_all_positions():
        sym = str(getattr(p, "symbol", "") or "")
        if not sym:
            continue
        settled[sym] = float(getattr(p, "qty", 0.0) or 0.0)

    filt = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = client.get_orders(filter=filt)
    wb, ws = working_shares_from_orders(list(orders))
    return settled, wb, ws
