# Data layer: `finTs` and providers

## `finTs`

`finTs` loads one or many tickers and builds a panel (MultiIndex `(Ticker, Date)` in the usual workflow) whose columns include raw **Open / High / Low / Close / Volume** first, then indicators such as `VWAP`, `SMA_50`, `RSI_14`, … (see `shunya.utils.indicators.STRATEGY_FEATURES_LIVE` for default ordering).

Best-effort **yfinance** classification columns are attached when available: **`Sector`**, **`Industry`**, **`SubIndustry`** (with deterministic `Unknown*` fallbacks). These matter for sector/industry neutralization in **FinStrat**.

## `MarketDataProvider`

History loading is abstracted under `shunya.data.providers`:

| Provider | Typical use |
|----------|-------------|
| **`YFinanceMarketDataProvider`** | Default in `finTs`; good for research and demos. |
| **`AlpacaHistoricalMarketDataProvider`** | Broker-aligned bars; **strict** if any requested symbol lacks bars (raises with symbol list). |
| **`TiingoMarketDataProvider`** | Daily EOD from Tiingo; set `SHUNYA_TIINGO_API_KEY` or `TIINGO_API_KEY`. Yahoo-style tickers are mapped for API calls only (e.g. `BRK.B` → `BRK-B`). |
| **`TimescaleMarketDataProvider`** | Read OHLCV from Postgres/Timescale after ingest; same daily contract as Yahoo-backed paths. |

**Contract:** `DatetimeIndex` named `Date`, normalized to **daily** granularity for the equity paths described in the main README.

## `DecisionContext`

`DecisionContext` (`shunya.algorithm.decision`) pins **signal time** and **data provenance** (`yfinance_research` vs `alpaca_bars`) so research and live workflows do not silently mix incompatible sources. Use it when resolving `as_of` against a panel for decisions that must align with execution data.

## Trading-time axis

- **`trading_axis_mode="observed"`** — calendar derived from observed panel rows (legacy-friendly).
- **`trading_axis_mode="canonical"`** — canonical US-equities trading calendar for the selected `BarSpec` (weekend gaps removed from bar progression).
- **`strict_trading_grid=True`** — enforce provider timestamps on the canonical grid with no in-session holes.

These interact with **FinStrat** temporal modes (`bar_step` vs `elapsed_trading_time`); see [FinStrat and FinBT](finstrat-finbt.md).

## Fundamentals

Optional **`TimescaleFundamentalDataProvider`** and related ingest attach fundamentals with the same research patterns as OHLCV; see [Local Timescale](../data_timescale.md) for CLI ingest.

## Further reading

- [Local Timescale](../data_timescale.md) — migrate, ingest, read in `finTs`.
- [Python API reference](../reference/library.md) — generated docstrings for public symbols.
