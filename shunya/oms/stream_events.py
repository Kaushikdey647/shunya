"""Broker-neutral events derived from Alpaca trade updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from .fills import ExecutionFill, utc_now

EventKind = Literal["fill", "partial_fill", "canceled", "rejected", "replaced", "new", "unknown"]


@dataclass(frozen=True, slots=True)
class NormalizedTradeEvent:
    """Simplified trade / order lifecycle update for the OMS."""

    kind: EventKind
    symbol: str
    order_id: str
    client_order_id: Optional[str]
    status: str
    event: str
    filled_qty: float
    filled_avg_price: Optional[float]
    qty: Optional[float]
    execution_id: Optional[str]
    timestamp: Optional[datetime] = None


def parent_id_from_client_order_id(client_order_id: Optional[str]) -> Optional[str]:
    """EMS child ids use ``{parent_id}:{slice}[u{n}]``; return the parent root id."""
    if not client_order_id:
        return None
    from shunya.ems.ids import parent_root_from_client_order_id

    return parent_root_from_client_order_id(client_order_id)


def normalize_alpaca_trade_update(data: object) -> NormalizedTradeEvent:
    """Map :class:`alpaca.trading.TradeUpdate` (or duck-typed) to :class:`NormalizedTradeEvent`."""
    ex = getattr(data, "execution_id", None)
    ex_s = str(ex) if ex is not None else None
    ev_raw = getattr(data, "event", None)
    ev = str(ev_raw) if ev_raw is not None else "unknown"
    order = getattr(data, "order", None)
    if order is None:
        return NormalizedTradeEvent(
            kind="unknown",
            symbol="",
            order_id="",
            client_order_id=None,
            status="",
            event=ev,
            filled_qty=0.0,
            filled_avg_price=None,
            qty=None,
            execution_id=ex_s,
            timestamp=getattr(data, "timestamp", None),
        )
    symbol = str(getattr(order, "symbol", "") or "")
    oid = str(getattr(order, "id", "") or "")
    cid = getattr(order, "client_order_id", None)
    cid_s = str(cid) if cid is not None else None
    status = str(getattr(order, "status", "") or "")
    filled_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
    fap = getattr(order, "filled_avg_price", None)
    fap_f = float(fap) if fap is not None else None
    qty_raw = getattr(order, "qty", None)
    qty_f = float(qty_raw) if qty_raw is not None else None
    ts = getattr(data, "timestamp", None)

    ev_l = ev.lower()
    kind: EventKind
    if "fill" in ev_l and "partial" in ev_l:
        kind = "partial_fill"
    elif "fill" in ev_l:
        kind = "fill"
    elif "cancel" in ev_l:
        kind = "canceled"
    elif "reject" in ev_l or "failed" in ev_l:
        kind = "rejected"
    elif "replace" in ev_l:
        kind = "replaced"
    elif "new" in ev_l or "accepted" in ev_l or "pending" in ev_l:
        kind = "new"
    else:
        kind = "unknown"

    return NormalizedTradeEvent(
        kind=kind,
        symbol=symbol,
        order_id=oid,
        client_order_id=cid_s,
        status=status,
        event=ev,
        filled_qty=filled_qty,
        filled_avg_price=fap_f,
        qty=qty_f,
        execution_id=ex_s,
        timestamp=ts,
    )


def execution_fill_from_trade_update(
    data: object,
    *,
    parent_order_id: str,
    fee: float = 0.0,
) -> Optional[ExecutionFill]:
    """
    Build an :class:`~shunya.oms.fills.ExecutionFill` when the update carries execution data.

    Returns None when ``execution_id`` or quantity/price are missing.
    """
    evt = normalize_alpaca_trade_update(data)
    if not evt.execution_id or evt.kind not in ("fill", "partial_fill"):
        return None
    qty = float(getattr(data, "qty", 0.0) or 0.0)
    price = float(getattr(data, "price", 0.0) or 0.0)
    if qty <= 0 or price < 0:
        return None
    order = getattr(data, "order", None)
    side_raw = str(getattr(order, "side", "") or "").upper()
    side = "BUY" if "BUY" in side_raw else "SELL"
    ex = getattr(data, "execution_id", None)
    ex_s = str(ex) if ex is not None else None
    if not ex_s:
        return None
    ts = evt.timestamp if isinstance(evt.timestamp, datetime) else utc_now()
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ExecutionFill(
        trade_id=ex_s,
        parent_order_id=parent_order_id,
        symbol=evt.symbol,
        side=side,
        quantity=qty,
        price=price,
        fee=float(fee),
        ts=ts,
        child_client_order_id=evt.client_order_id,
    )
