# Glossary

Short definitions for terms used across Shunya docs and code.

| Term | Meaning |
|------|---------|
| **Alpha** | A named strategy definition: Python body (`source_code` and/or `import_ref`) plus **FinStrat** JSON config; stored and versioned via **`/alphas`**. |
| **`api_runtime_config`** | Database overlay merged with env for tunable API settings; read via **`GET /settings/app`**, written via **`PATCH /settings/app`**. |
| **`cross_section` (`cs`)** | JAX-friendly cross-sectional ops (`rank`, `zscore`, …) on per-bar vectors. |
| **`DecisionContext`** | Object pinning signal time and data provenance (`yfinance_research` vs `alpaca_bars`). |
| **`finTs`** | Multi-ticker panel builder: OHLCV, indicators, optional sector columns, configurable trading calendar. |
| **`FinBT`** | Backtrader adapter that runs a **FinStrat** on a **`finTs`** panel with commissions, slippage, and rebalance targets. |
| **`FinStrat`** | Binds a panel to `algorithm(ctx) -> weights` with neutralization, decay, truncation, and temporal modes. |
| **`import_ref`** | Allow-listed module attribute path for an alpha callable when not using inline `source_code`. |
| **Index backtest** | **`POST /backtests`** with **`index_code`**: universe from `symbol_index_membership`, Timescale-only OHLCV, benchmark from catalog. |
| **`MarketDataProvider`** | Pluggable historical loader (Yahoo, Alpaca, Tiingo, Timescale, …). |
| **OMS** | Order management system — parent orders, ledger, reconciliation (`shunya.oms`). |
| **EMS** | Execution management — scheduling and broker gateway (`shunya.ems`). |
| **`include_test_period_in_results`** | API flag: when `false`, metrics exclude data from **2025-01-01** onward for a tune-only view. |
| **`PCS`** | **PortfolioConstructionService** — targets from blend configs. |
| **`PortfolioRiskEngine`** | Optional pre-trade risk checks (CVX-backed pieces need **`[risk]`** extra). |
| **Trade-desk token** | Shared secret **`SHUNYA_API_TRADE_DESK_TOKEN`**; client sends **`X-Shunya-Trade-Desk-Token`**. |
| **Worker** | Background process claiming **`api_backtest_jobs`** (and related) from Postgres. |

## See also

- [Concepts: System overview](concepts/overview.md)
- [HTTP API](http-api.md)
- [Python reference](reference/library.md)
