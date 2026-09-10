# Trade blotter: end-to-end walkthrough

This walks through placing, watching, and canceling a manual order with the **`/trade/blotter`** web UI, backed by the manual OMS API (`api/routers/oms.py`).

## Prerequisites

Same as the [paper trading cycle](paper-cycle-api.md):

1. **`SHUNYA_API_ALPACA_ENABLED=1`** so the API builds Alpaca clients and the manual OMS runtime at startup.
2. Alpaca credentials in the environment (`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, or the `SHUNYA_ALPACA_*` aliases).
3. **`SHUNYA_API_TRADE_DESK_TOKEN`** set to a shared secret on the API.
4. In the UI, open **Settings** and paste the same value so it's applied as the trade-desk token for this browser session (see [Trade → token gating](../ui/trade.md#trade-desk-token-gating)).

## Placing an order

Navigate to **`/trade/blotter`**. The order entry panel takes:

- **Symbol** — ticker, uppercased automatically.
- **Side** — BUY or SELL.
- **Order type** — MARKET or LIMIT.
- **Quantity** — whole shares.
- **Limit price** — only shown/required when order type is LIMIT.

Submit places `POST /trade/orders`. On success you'll see a toast with the assigned state, and the orders query refreshes so the new order appears in the blotter below. On rejection (e.g. bad symbol, market closed, insufficient buying power) you'll see the broker's error message instead — no order is created.

## Understanding the PAPER/LIVE badge

The badge next to "Order entry" is **read-only** — it reflects `SHUNYA_ALPACA_PAPER` on the API (defaults to paper), not a UI setting. There is no way to flip it from the browser. To trade live, an operator must set `SHUNYA_ALPACA_PAPER=false` (with live keys) and restart the API — treat this as a deliberate, deployment-level decision, not something to toggle casually.

## Watching it fill

Once submitted, the order becomes **working** and appears in the **Orders** tab, live-updated over the WebSocket (`WS /trade/orders/stream`) as its state changes — no manual refresh needed. Each execution appears as a line in the **fills tape** alongside the entry form, newest first. If the order fills in slices, the orders table shows the running `quantity_filled` against `quantity_ordered`; the realized average price across those fills is the order's VWAP (see [Working orders, partial fills, and reconciliation](../concepts/oms-ems-and-order-routing.md#working-orders-partial-fills-and-reconciliation)).

## Canceling an order

From the **Orders** tab, use the cancel action on a still-working order — this calls `DELETE /trade/orders/{parent_id}`. The row does **not** immediately flip to canceled: the request only asks the broker to cancel, and the order-status update arrives asynchronously over the Alpaca trade-update stream, which is why the UI may briefly still show it as working.

## Reading the execution tracer

From the **Execution hub** (`/execution`), pick a recent order to open its **tracer** (`/execution/:parentId`). This shows the parent order detail, its full fills list, and (if available) a live L1 bid/ask panel for the order's symbol — all backed by real data (`GET /trade/orders/{id}`, `GET /trade/orders/{id}/fills`), not the mock state older builds of this page used.

## Troubleshooting

- **401 / requests silently fail** — the trade-desk token draft hasn't been applied; re-enter and apply it in Settings, and remember it's cleared when the browser session ends (`sessionStorage`).
- **Everything on the page errors or the badge never appears** — the API likely doesn't have `SHUNYA_API_ALPACA_ENABLED` set, or Alpaca credentials are missing/invalid; check API startup logs.
- **Blotter loads but never updates live** — the WebSocket requires the token as a `?token=` query parameter (`WS /trade/orders/stream?token=...`); if it's missing or wrong the socket will fail to connect and the UI falls back to REST polling only (slower, but orders/positions still eventually refresh).

## See also

- [Trade area (UI)](../ui/trade.md)
- [OMS, EMS, and order routing](../concepts/oms-ems-and-order-routing.md)
- [Risk management and avoiding losses](../concepts/risk-management-and-avoiding-losses.md)
