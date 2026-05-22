# Market data routing and writeback semantics

This document complements Timescale storage docs (`docs/data_timescale.md`).

## Read path vs upstream

- **read_path** (`timescale` | `live_fetch`): how the API materialized the response.
- **upstream_source_id** (`yfinance`, `alpaca_sip`, …): what vendor semantics the OHLCV bytes carry. The `ohlcv_bars.source` column stores this upstream id (not the word `timescale`).

## `best_effort` ingest / writeback

- **UNIQUE** `(symbol_id, ts, interval, source)` prevents one upstream from silently replacing another for the same bar key. To change primary vendor for a series, run an explicit migration or delete the old upstream slice first.
- **Default**: stop after the first upstream that returns a **non-empty** validated window for the HTTP path (no automatic gap-fill across vendors in v1).
- **Deferred writeback** uses the **same** `source` as the live fetch that produced the frame (`yfinance`, `alpaca_sip`, …).

## `ohlcv_bars.metadata` (JSONB)

Migration `016_ohlcv_bars_metadata.sql` adds optional `metadata` for feed, tier, adjustment convention, and session flags without widening the primary key. Upserts pass `NULL` until callers populate JSON.

## Yahoo public adapter (non-router HTTP)

- **`shunya.integration.yahoo_public.YahooPublicAdapter`** centralizes **`build_yfinance_session()`** (or an injected session) for **`yf.download`**, **`yf.screen`**, **`yf.Search`**, and **`yf.Ticker`** on market snapshot / movers / headlines, instruments search and news, instrument dashboard, extended yfinance tables, and the Timescale **`ingest-fundamentals`** per-symbol loop.
- OHLCV-shaped paths that need **repair / validation** stay on **`YFinanceMarketDataProvider`** inside **`shunya.data.market_router`**.

## Stored upstream id constant

- **`STORED_OHLCV_DEFAULT_UPSTREAM_ID`** (`shunya.data.market_data.constants`, value `yfinance`) is the canonical **`ohlcv_bars.source`** for Yahoo-ingested bars. Bootstrap scripts, index OHLCV helpers, **`TimescaleMarketDataProvider`**, **`resolve_market_route`**, and compose bootstrap probes reference this constant.

## FinTs bridge (v1)

- **`shunya.data.market_data.fints_bridge.resolve_market_data_provider`** calls **`resolve_market_route`** on a **`MarketDataRouteContext`** built from **`FinTsRequest`** before constructing **`TimescaleMarketDataProvider`** or **`AlpacaHistoricalMarketDataProvider`**. HTTP-only concerns such as manifest TTL and deferred writeback are **not** replicated in v1.
