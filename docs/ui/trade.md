# Trade area (sidebar)

## Portfolios (`/portfolios`, `/portfolios/:id`)

Portfolio workspace: blend configurations and weights. State may combine API data with client-side persistence depending on route coverage; consult `ui/src/hooks/useTradeDesk.ts` and related components for current behavior.

## Live (`/live`)

Cockpit-style strip for live-style monitoring (layout oriented to trading workflows).

## Account (`/trade/account`)

When the API has **`SHUNYA_API_ALPACA_ENABLED`** and keys configured, the UI can call:

- Equity snapshot
- Portfolio history
- Activities
- Account configurations (get/patch)

Send header **`X-Shunya-Trade-Desk-Token`** matching **`SHUNYA_API_TRADE_DESK_TOKEN`** on the API (paste the same value in **Settings** where provided). Without Alpaca + token, these calls fail or are gated by the API.

## Execution (`/execution`, `/execution/:parentId`)

**Execution hub** and **tracer** routes exist in the router; much of the **Trade** desk still uses **mock client state** in **`localStorage`** (`shunya_trade_desk_v1`) until additional live OMS/EMS HTTP APIs are exposed.

## Risk (`/risk`)

**Risk command center** — limits and diagnostics UI backed by mock/local state in the same sense as Execution until server routes are expanded.

## Paper cycle (API)

One-shot institutional paper cycle is available at **`POST /trade/paper/cycle`** (not a full duplicate of the Account page). See [Paper trading cycle (API)](../how-to/paper-cycle-api.md).

## See also

- [Execution: OMS, EMS, live desk](../concepts/oms-ems.md)
- [HTTP API](../http-api.md)
