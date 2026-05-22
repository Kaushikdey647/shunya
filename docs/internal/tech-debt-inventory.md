# Internal tech debt inventory

Maintainer-facing backlog pointers. Inline `TODO(market-data-router):` marks the highest-priority seams for the market data capability registry and router work.

## Market data sprawl (yfinance outside `YFinanceMarketDataProvider`)

Non-OHLCV Yahoo reads go through **`shunya.integration.yahoo_public.YahooPublicAdapter`** (session injection, single place to extend provenance). **`YFinanceMarketDataProvider`** remains the repair/sanitize path for router OHLCV.

Remaining inline **`yfinance`** usage should be tracked with ripgrep when adding new HTTP surfaces.

## Source string drift (`ohlcv_bars.source`)

**Update:** canonical id is **`STORED_OHLCV_DEFAULT_UPSTREAM_ID`** in **`shunya.data.market_data.constants`**. Bootstrap, index helpers, router, and **`TimescaleMarketDataProvider`** default import it.

## Alpaca historical bars

| Location | Notes |
|----------|------|
| [shunya/data/providers.py](../../shunya/data/providers.py) | `StockBarsRequest` uses `feed=self._data_feed`; ensure ctor/env defaults and future registry `SourceId` stay aligned with `ohlcv_bars.source` on ingest |

## API provenance

**Update:** **`InstrumentOhlcvResponse`** exposes **`provenance`** only (no **`data_source`**).

## FinTs vs HTTP OHLCV

**Update:** **`shunya.data.market_data.fints_bridge.resolve_market_data_provider`** calls **`resolve_market_route`** before constructing providers. HTTP still runs manifest TTL + writeback; FinTs v1 does not.

## Writeback and multi-upstream storage

| Location | Notes |
|----------|------|
| [shunya/data/timescale/ohlcv_writeback.py](../../shunya/data/timescale/ohlcv_writeback.py) | Docstring still yfinance-centric; UNIQUE `(symbol_id, ts, interval, source)` must align with no silent cross-upstream overwrite policy |

## CLI robustness (optional follow-up)

| Location | Notes |
|----------|------|
| [shunya/data/timescale/cli.py](../../shunya/data/timescale/cli.py) | `cmd_ingest_fundamentals`: broad `except` + `pass` on optional yfinance tables—acceptable for CLI but obscures failures |

## Broader `except Exception` (inventory only — not all need FIXME)

Defensive boundaries appear across health checks, kite execution, bootstrap scripts, and workers. Review case-by-case when touching a subsystem; do not blanket-tag with TODO.

Examples: [api/health_checks.py](../../api/health_checks.py), [shunya/algorithm/kite_execution.py](../../shunya/algorithm/kite_execution.py), [scripts/bootstrap_sp100_timescale.py](../../scripts/bootstrap_sp100_timescale.py).

## Follow-up: machine enforcement

Prefer **ruff** / typed narrowing for `BLE001` over hundreds of inline `FIXME` comments once policy is agreed.

## Secondary seams (additional `TODO` / `FIXME` markers)

| Location | Marker | Notes |
|----------|--------|------|
| [scripts/bootstrap_sp100_timescale.py](../../scripts/bootstrap_sp100_timescale.py) | TODO | Third `_OHLCV_SOURCE` copy |
| [api/services/ohlcv_yfinance_backfill.py](../../api/services/ohlcv_yfinance_backfill.py) | TODO | Yahoo-only recovery upsert |
| [api/worker_job.py](../../api/worker_job.py) | TODO | Worker recovery calls Yahoo backfill only |
| [scripts/bootstrap_ts_data.py](../../scripts/bootstrap_ts_data.py) | TODO | Large bootstrap vs CLI/registry consolidation |
| [scripts/gapfill_sp100_universe_metadata.py](../../scripts/gapfill_sp100_universe_metadata.py) | TODO | Hardcoded `source=` strings for delegated CLIs |
| [api/timescale_classifications.py](../../api/timescale_classifications.py) | FIXME | SQL `c.source = 'yfinance'` hides other upstreams |
| [api/db_dashboard.py](../../api/db_dashboard.py) | FIXME | Same for dashboard sector/industry counts |
| [shunya/data/providers.py](../../shunya/data/providers.py) (`fetch_yfinance_classifications`) | TODO | Direct `yf.Ticker` vs shared download adapter |
| [shunya/data/market_router.py](../../shunya/data/market_router.py) | TODO | Hub: migrate callers here |
| [shunya/data/market_data/resolve.py](../../shunya/data/market_data/resolve.py) | TODO | Pure policy entry point for scripts/worker/SQL consumers |
