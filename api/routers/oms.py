"""Manual order entry, cancel, blotter reads, and live order-updates stream (secured; requires
lifespan Alpaca + OMS runtime)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket
from starlette.websockets import WebSocketDisconnect

from api.routers.trade_desk import require_trade_desk_token
from api.schemas.models import (
    OmsCancelResponse,
    OmsFillOut,
    OmsFillsResponse,
    OmsOrderPlaceRequest,
    OmsOrdersResponse,
    OmsParentOrderOut,
    OmsPositionOut,
    OmsPositionsResponse,
)
from api.services.oms_orders import OrderRejected, cancel_manual_parent, submit_manual_parent
from api.settings import get_settings
from api.trade_desk_runtime import TradeDeskRuntime
from shunya.oms.fills import vwap_avg_price

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/trade", tags=["oms"])


def _trade_desk_runtime(request: Request) -> TradeDeskRuntime:
    rt = getattr(request.app.state, "trade_desk_runtime", None)
    if rt is None:
        raise HTTPException(
            status_code=503,
            detail="Trade desk is disabled. Set SHUNYA_API_ALPACA_ENABLED=1 and APCA API keys.",
        )
    return rt


def _oms_runtime(request: Request):
    rt = getattr(request.app.state, "oms_runtime", None)
    if rt is None:
        raise HTTPException(
            status_code=503,
            detail="OMS runtime is disabled. Set SHUNYA_API_ALPACA_ENABLED=1 and APCA API keys.",
        )
    return rt


def _parent_row_to_out(row: Any, avg_fill_price: Optional[float]) -> OmsParentOrderOut:
    return OmsParentOrderOut(
        parent_id=row.id,
        symbol=row.symbol,
        side=row.side,
        quantity_ordered=row.quantity_ordered,
        quantity_filled=row.quantity_filled,
        state=row.state,
        order_type=row.order_type,
        limit_price=row.limit_price,
        time_in_force=row.time_in_force,
        client_order_id=row.client_order_id,
        broker_order_id=row.broker_order_id,
        avg_fill_price=avg_fill_price,
        origin=row.origin,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _parent_memory_to_out(p: Any, avg_fill_price: Optional[float]) -> OmsParentOrderOut:
    return OmsParentOrderOut(
        parent_id=p.parent_id,
        symbol=p.symbol,
        side=p.side,
        quantity_ordered=p.quantity_ordered,
        quantity_filled=p.quantity_filled,
        state=p.state,
        client_order_id=p.client_order_id,
        avg_fill_price=avg_fill_price,
    )


def _fill_to_out(f: Any) -> OmsFillOut:
    return OmsFillOut(
        trade_id=f.trade_id,
        parent_order_id=f.parent_order_id,
        symbol=f.symbol,
        side=f.side,
        quantity=f.quantity,
        price=f.price,
        fee=f.fee,
        ts=f.ts,
        child_client_order_id=f.child_client_order_id,
    )


@router.post("/orders", response_model=OmsParentOrderOut, dependencies=[Depends(require_trade_desk_token)])
def place_order(
    body: OmsOrderPlaceRequest,
    request: Request,
) -> OmsParentOrderOut:
    """Submits a manual parent order through the OMS runtime and returns ledger state."""
    rt = _trade_desk_runtime(request)
    oms_rt = _oms_runtime(request)
    try:
        parent_id = submit_manual_parent(rt, oms_rt, body)
    except OrderRejected as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    parent = oms_rt.oms.ledger.get_parent(parent_id)
    if parent is None:
        raise HTTPException(status_code=500, detail="Order registered but not found in ledger.")
    fills = oms_rt.repo.list_fills_for_parent(parent_id) if oms_rt.repo is not None else []
    avg = vwap_avg_price(fills) if fills else None
    if oms_rt.repo is not None:
        row = oms_rt.repo.get_parent(parent_id)
        if row is not None:
            return _parent_row_to_out(row, avg)
    return _parent_memory_to_out(parent, avg)


@router.delete(
    "/orders/{parent_id}", response_model=OmsCancelResponse, dependencies=[Depends(require_trade_desk_token)]
)
def cancel_order(parent_id: str, request: Request) -> OmsCancelResponse:
    """Cancels a manual parent order via cancel_manual_parent."""
    rt = _trade_desk_runtime(request)
    oms_rt = _oms_runtime(request)
    parent = oms_rt.oms.ledger.get_parent(parent_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="Unknown parent order id.")
    try:
        cancel_manual_parent(rt, oms_rt, parent_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OrderRejected as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OmsCancelResponse(parent_id=parent_id, state=parent.state, canceled=True)


@router.get("/orders", response_model=OmsOrdersResponse, dependencies=[Depends(require_trade_desk_token)])
def list_orders(
    request: Request,
    state: str = Query(default="all", description="'open' or 'all'"),
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> OmsOrdersResponse:
    """Lists parent orders from Postgres or in-memory ledger with VWAP fills."""
    oms_rt = _oms_runtime(request)
    sym = symbol.strip().upper() if symbol else None

    if oms_rt.repo is not None:
        states = ("PENDING_SUBMIT", "WORKING", "PARTIAL_FILL") if state == "open" else None
        rows = oms_rt.repo.list_parents(states=states, symbol=sym, limit=limit)
        out = []
        for row in rows:
            fills = oms_rt.repo.list_fills_for_parent(row.id)
            out.append(_parent_row_to_out(row, vwap_avg_price(fills) if fills else None))
        return OmsOrdersResponse(orders=out)

    parents = list(oms_rt.oms.ledger.parents.values())
    if sym:
        parents = [p for p in parents if p.symbol == sym]
    if state == "open":
        parents = [p for p in parents if not p.is_terminal()]
    parents = parents[-limit:]
    return OmsOrdersResponse(orders=[_parent_memory_to_out(p, None) for p in parents])


@router.get(
    "/orders/{parent_id}", response_model=OmsParentOrderOut, dependencies=[Depends(require_trade_desk_token)]
)
def get_order(parent_id: str, request: Request) -> OmsParentOrderOut:
    """Fetches one parent order with average fill price when fills exist."""
    oms_rt = _oms_runtime(request)
    fills = oms_rt.repo.list_fills_for_parent(parent_id) if oms_rt.repo is not None else []
    avg = vwap_avg_price(fills) if fills else None
    if oms_rt.repo is not None:
        row = oms_rt.repo.get_parent(parent_id)
        if row is not None:
            return _parent_row_to_out(row, avg)
    parent = oms_rt.oms.ledger.get_parent(parent_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="Unknown parent order id.")
    return _parent_memory_to_out(parent, avg)


@router.get(
    "/orders/{parent_id}/fills",
    response_model=OmsFillsResponse,
    dependencies=[Depends(require_trade_desk_token)],
)
def get_order_fills(parent_id: str, request: Request) -> OmsFillsResponse:
    """Lists fills for a parent order id."""
    oms_rt = _oms_runtime(request)
    if oms_rt.repo is not None:
        fills = oms_rt.repo.list_fills_for_parent(parent_id)
    else:
        fills = [f for f in oms_rt.oms.ledger.fills if f.parent_order_id == parent_id]
    return OmsFillsResponse(fills=[_fill_to_out(f) for f in fills])


@router.get("/fills", response_model=OmsFillsResponse, dependencies=[Depends(require_trade_desk_token)])
def list_fills(
    request: Request,
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> OmsFillsResponse:
    """Lists recent fills optionally filtered by symbol."""
    oms_rt = _oms_runtime(request)
    sym = symbol.strip().upper() if symbol else None
    if oms_rt.repo is not None:
        fills = oms_rt.repo.list_recent_fills(limit=limit, symbol=sym)
    else:
        fills = list(oms_rt.oms.ledger.fills)
        if sym:
            fills = [f for f in fills if f.symbol == sym]
        fills = fills[-limit:]
    return OmsFillsResponse(fills=[_fill_to_out(f) for f in fills])


@router.get(
    "/positions", response_model=OmsPositionsResponse, dependencies=[Depends(require_trade_desk_token)]
)
def list_positions(request: Request) -> OmsPositionsResponse:
    """Merges Alpaca positions with working buy/sell from the OMS ledger."""
    rt = _trade_desk_runtime(request)
    oms_rt = _oms_runtime(request)
    try:
        positions = rt.trading_client.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to fetch positions: {exc}") from exc

    working_buy, working_sell = oms_rt.oms.ledger.working_buy_sell()
    out = []
    for p in positions:
        sym = str(p.symbol)
        out.append(
            OmsPositionOut(
                symbol=sym,
                qty=float(p.qty),
                side=str(p.side.value if hasattr(p.side, "value") else p.side),
                avg_entry_price=float(p.avg_entry_price) if p.avg_entry_price is not None else None,
                market_value=float(p.market_value) if p.market_value is not None else None,
                unrealized_pl=float(p.unrealized_pl) if p.unrealized_pl is not None else None,
                unrealized_plpc=float(p.unrealized_plpc) if p.unrealized_plpc is not None else None,
                current_price=float(p.current_price) if p.current_price is not None else None,
                working_buy_shares=float(working_buy.get(sym, 0.0)),
                working_sell_shares=float(working_sell.get(sym, 0.0)),
            )
        )
    return OmsPositionsResponse(positions=out, paper=bool(oms_rt.paper))


async def _wait_websocket_disconnect(websocket: WebSocket) -> None:
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


@router.websocket("/orders/stream")
async def orders_stream_ws(websocket: WebSocket) -> None:
    """WebSocket order updates after trade-desk token and OMS runtime checks."""
    await websocket.accept()

    settings = get_settings()
    token = settings.trade_desk_token
    provided = websocket.query_params.get("token")
    if not token or not provided or provided != token:
        await websocket.send_json(
            {"type": "error", "code": "unauthorized", "message": "Missing or invalid trade desk token."}
        )
        await websocket.close(code=1008)
        return

    oms_rt = getattr(websocket.app.state, "oms_runtime", None)
    if oms_rt is None:
        await websocket.send_json(
            {
                "type": "error",
                "code": "oms_disabled",
                "message": "OMS runtime is disabled for this API deployment.",
            }
        )
        await websocket.close(code=1008)
        return

    await websocket.send_json({"type": "hello", "schema": 1, "paper": bool(oms_rt.paper)})
    await oms_rt.hub.attach(websocket)
    try:
        await _wait_websocket_disconnect(websocket)
    finally:
        await oms_rt.hub.detach(websocket)
