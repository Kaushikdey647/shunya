# Paper trading cycle (HTTP API)

The FastAPI route **`POST /trade/paper/cycle`** runs one **paper** cycle through **`PortfolioRiskEngine`** → **`InstitutionalOMS`** → **`EMSParentRunner`** with the Alpaca trade stream bridge.

## Prerequisites

1. **`SHUNYA_API_ALPACA_ENABLED=1`** (or `true`) so the API builds Alpaca clients at startup.
2. **Alpaca credentials** in the environment: `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, or `SHUNYA_ALPACA_API_KEY_ID` / `SHUNYA_ALPACA_API_SECRET_KEY`.
3. **`SHUNYA_API_TRADE_DESK_TOKEN`** set to a shared secret.
4. On each request, send header **`X-Shunya-Trade-Desk-Token`** with the **same** value as `SHUNYA_API_TRADE_DESK_TOKEN`.

## Request body (summary)

- **`capital`** — notional context for the cycle.
- **`execution_date`** — `YYYY-MM-DD`.
- Either **`use_demo_pcs: true`** (built-in SPY/QQQ stub book) **or** **`targets_usd` + `universe` + `prices`** for a fixed-target run.

Full field list and semantics: [api/README.md — Backtest API](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md) and `api/routers/trade_desk.py` in the repository.

## UI parity

The web app **Trade → Account** and **Settings → Alpaca** use other **`/trade/...`** and **`PATCH /settings/app`** routes with the same trade-desk token when Alpaca is enabled. See [Trade](../ui/trade.md).

## Conceptual background

[Execution: OMS, EMS](../documentation/oms-ems.md); finance view: [OMS, EMS, and order routing](../concepts/oms-ems-and-order-routing.md).
