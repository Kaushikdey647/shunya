# Execution: OMS, EMS, and live desk

## OMS (`shunya.oms`)

The **OMS** layer models institutional parent orders: parent-order FSM, in-memory ledger, share reconciliation versus USD targets, Alpaca stream and REST helpers, and optional **Postgres** persistence (`shunya/oms/db`) with **Alembic** migrations under `alembic/versions/`.

## EMS (`shunya.ems`)

The **EMS** schedules and routes child orders: **`BrokerGateway`** / **`AlpacaBrokerGateway`**, TWAP/VWAP-style schedulers, micro-price limit helpers, and **`EMSParentRunner`** for child lifecycle.

## Alpaca integration

Low-level helpers **`AlpacaExecutionAdapter`** and **`OrderManager`** translate signed USD deltas into orders and cache open-order state across **your** rebalance loop. There is no bundled tick-to-trade runner; you connect these behind a scheduler or service.

**Environment and clients:** use `shunya.integration.alpaca_settings` (`AlpacaRuntimeSettings`, `load_alpaca_settings_from_env`, `build_trading_client`, `build_stock_historical_data_client`, `build_trading_stream`) so credentials and paper mode stay consistent.

Typical keys: `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, optional `SHUNYA_ALPACA_*` aliases, and **`SHUNYA_ALPACA_PAPER`** (defaults to paper).

## `InstitutionalPaperDesk`

**`InstitutionalPaperDesk`** (`shunya.live.desk`) wires **PortfolioConstructionService** (or fixed USD targets) → **`PortfolioRiskEngine`** → **`InstitutionalOMS`** → **`EMSParentRunner`** with **`AlpacaOMSTradeStream`**.

## CLI: `shunya-paper`

From a clone with the appropriate extras:

```bash
uv run shunya-paper paper-cycle --date YYYY-MM-DD [--capital N] [--demo | --pcs-factory module:fn]
```

See the repository **README** for full flags and examples.

## HTTP trade desk

The FastAPI app exposes **`POST /trade/paper/cycle`** for a single paper cycle when Alpaca and the trade-desk token are enabled. See [Paper trading cycle (API)](../how-to/paper-cycle-api.md) and the [HTTP API](../http-api.md) outline.
