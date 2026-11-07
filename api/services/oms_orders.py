"""Manual single-order submission/cancellation against Alpaca, routed through the OMS singleton."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from shunya.algorithm.execution import AlpacaExecutionAdapter
from shunya.ems.ids import child_client_order_id
from shunya.oms.service import ParentIntent

from api.schemas.models import OmsOrderPlaceRequest

if TYPE_CHECKING:
    from api.services.oms_runtime import OMSRuntime
    from api.trade_desk_runtime import TradeDeskRuntime

_log = logging.getLogger(__name__)


class OrderRejected(RuntimeError):
    """Raised when Alpaca rejects a manual order submission (asset validation or broker error)."""


def submit_manual_parent(rt: "TradeDeskRuntime", oms_rt: "OMSRuntime", req: OmsOrderPlaceRequest) -> str:
    """
    Submit one manual parent order to Alpaca and register it with the OMS singleton.

    Returns the new ``parent_id``. Registers the parent in the ledger **before** the
    Alpaca submission call so a near-instant fill delivered over the trade-update
    stream finds an owning parent (``apply_alpaca_trade_update`` silently drops
    updates for unknown parent ids).
    """
    adapter = AlpacaExecutionAdapter(rt.trading_client)
    delta_usd = float(req.quantity) if req.side == "BUY" else -float(req.quantity)
    try:
        adapter.validate_asset(req.symbol, delta_usd=delta_usd)
    except Exception as exc:  # noqa: BLE001
        raise OrderRejected(f"{req.symbol} failed asset validation: {exc}") from exc

    parent_id = uuid.uuid4().hex
    client_order_id = child_client_order_id(parent_id, 0)

    oms_rt.oms.create_parent_order(
        ParentIntent(parent_id=parent_id, symbol=req.symbol, side=req.side, quantity=req.quantity)
    )

    side = OrderSide.BUY if req.side == "BUY" else OrderSide.SELL
    tif = TimeInForce.GTC if req.time_in_force == "GTC" else TimeInForce.DAY

    if req.order_type == "LIMIT":
        order_req: MarketOrderRequest | LimitOrderRequest = LimitOrderRequest(
            symbol=req.symbol,
            qty=req.quantity,
            side=side,
            time_in_force=tif,
            limit_price=round(float(req.limit_price), 2),
            client_order_id=client_order_id,
        )
    else:
        order_req = MarketOrderRequest(
            symbol=req.symbol,
            qty=req.quantity,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
        )

    try:
        order = rt.trading_client.submit_order(order_req)
    except APIError as exc:
        _log.warning("Manual order submission rejected for %s: %s", req.symbol, exc)
        parent = oms_rt.oms.ledger.get_parent(parent_id)
        if parent is not None:
            try:
                parent.broker_reject()
            except Exception:  # noqa: BLE001
                pass
            oms_rt.oms._persist_parent(parent, origin="manual")  # noqa: SLF001
        raise OrderRejected(str(exc)) from exc

    oms_rt.oms.set_broker_order_id(parent_id, str(getattr(order, "id", "") or ""))
    oms_rt.oms.mark_parent_working(parent_id, client_order_id)

    if oms_rt.repo is not None:
        parent = oms_rt.oms.ledger.get_parent(parent_id)
        if parent is not None:
            oms_rt.repo.upsert_parent_snapshot(
                parent_id=parent_id,
                symbol=parent.symbol,
                side=parent.side,
                quantity_ordered=parent.quantity_ordered,
                quantity_filled=parent.quantity_filled,
                state=parent.state,
                client_order_id=parent.client_order_id,
                created_at=_utc_now(),
                order_type=req.order_type,
                limit_price=req.limit_price,
                time_in_force=req.time_in_force,
                broker_order_id=str(getattr(order, "id", "") or ""),
                origin="manual",
            )
    return parent_id


def cancel_manual_parent(rt: "TradeDeskRuntime", oms_rt: "OMSRuntime", parent_id: str) -> bool:
    """
    Request cancellation of a working manual parent at the broker.

    Does not force local state to CANCELED — the Alpaca trade-update stream is the
    single writer of terminal state; this only requests the cancel.
    """
    broker_order_id = oms_rt.oms.get_broker_order_id(parent_id)
    if not broker_order_id:
        raise ValueError(f"No broker order id on file for parent {parent_id}")
    try:
        rt.trading_client.cancel_order_by_id(broker_order_id)
    except APIError as exc:
        raise OrderRejected(str(exc)) from exc
    return True


def _utc_now():
    """Delegate to ``shunya.oms.fills.utc_now`` for persisted timestamps."""
    from shunya.oms.fills import utc_now

    return utc_now()
