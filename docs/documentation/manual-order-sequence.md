# Manual order sequence

Happy path for a blotter order (`POST /trade/orders`). This path checks that Alpaca knows the asset, then submits one order. It does not call `PortfolioRiskEngine` and it does not run `EMSParentRunner` or TWAP/VWAP slicing. Those steps belong to `POST /trade/paper/cycle` via `InstitutionalPaperDesk`.

The manual parent id is still formatted as an EMS child id (`{parent_id}:0`) so fill events parse the same way.

```mermaid
sequenceDiagram
    participant UI as Blotter
    participant API as OmsRouter
    participant Svc as submit_manual_parent
    participant Val as AlpacaExecutionAdapter
    participant OMS as PersistingOMS
    participant FSM as ParentOrder
    participant DB as OMSRepository
    participant REST as AlpacaREST
    participant Stream as AlpacaTradeStream
    participant Hub as OrderEventHub
    UI->>API: POST /trade/orders
    API->>Svc: submit_manual_parent
    Svc->>Val: validate_asset
    Svc->>OMS: create_parent_order
    OMS->>FSM: PENDING_SUBMIT
    OMS->>DB: upsert parent
    OMS->>Hub: order event
    Svc->>REST: submit_order
    Svc->>OMS: mark_parent_working
    OMS->>FSM: WORKING
    OMS->>DB: persist snapshot
    REST-->>Stream: trade update
    Stream->>OMS: apply_alpaca_trade_update
    OMS->>FSM: PARTIAL_FILL or FILLED
    OMS->>DB: insert fill
    OMS->>Hub: fill and order events
    Hub-->>UI: WS /trade/orders/stream
```

Position rows on `GET /trade/positions` come from Alpaca `get_all_positions()`, annotated with working buy and sell quantity from the in-memory ledger.

Cancel (`DELETE /trade/orders/{parent_id}`) calls Alpaca `cancel_order_by_id` and leaves the local terminal state to the trade stream.

Startup `rehydrate` reloads open parents and reconciles missed fills from Alpaca REST. That path is not part of the submit sequence above.

See [OMS, EMS, and the live desk](oms-ems.md) for the two OMS instances and [HTTP behavior beyond OpenAPI](../reference/http-beyond-openapi.md) for the socket frames.
