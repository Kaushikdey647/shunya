# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **API + UI notifications:** **WebSocket** **`/notifications/stream`** (`api.services.notification_hub`, `api.routers.notifications_stream`) sends **`hello`** then **`notification`** JSON (`schema: 1`, `id`, `ts`, `level`, `message`, optional `code` / `title` / `context`); the in-process backtest worker publishes on job failure. The UI opens one socket (reconnect with backoff), shows Mantine notifications (top-right, 10s, close), keeps a capped session history, and lists it from the header bell; **`apiFetch`** HTTP errors are folded into the same client-side list without server broadcast.

- **Notification publishers (CRUD + jobs):** **`api.services.notify_background.schedule_notification`** lets sync routes enqueue **`publish_notification`** via **`BackgroundTasks`**. Backtest lifecycle: **`backtest.queued`** (enqueue), **`backtest.started`** / **`backtest.succeeded`** (worker), existing failure toast; **`backtest.deleted`** and batch-delete summaries. Alpha and universe mutations emit **`alpha.*`** and **`universe.*`** codes; **`PATCH /settings/app`** emits **`settings.app_patched`** with patched key names in **`context`**.

- **Market clock (library + API + UI):** **`shunya.time.market_clock`** exposes US (`America/New_York`) and India (`Asia/Kolkata`) listing-time formatting, **`us_listed_rth_open`**, and **`alpaca_l1_us_equities_stream_allowed`** (aligned with **`SHUNYA_ALPACA_L1_IGNORE_US_RTH`** for dev). **`GET /settings/market-clock`** returns a JSON snapshot. **`/instruments/{symbol}/stream/alpaca-l1`** rejects new sessions outside US regular hours with **`error`** / **`code: us_rth_closed`** unless the ignore flag is set.

- **UI:** header **MarketClockStrip** (US + IN lines) driven by **WebSocket** **`/settings/market-clock/stream`** (server ticks; reconnect with backoff); **Live Data** uses notifications for outside-RTH attempts instead of gating **Connect** from the clock.

### Changed

- **UI — quant research workstation:** **IDE-style** dark canvas (**`#0D1117`** / panels **`#161B22`**, text **`#E6EDF3`**) in resolver + legacy CSS + Mantine `other.dark*`; sidebar matches panels. **Alpha Studio:** **`Grid` ~7/5** split, **`calc`-based** editor height, **`AlphaSourceEditor`** tab bar (**`alpha_signal.py`**) + **unsaved** badge; **`FinStratConfigForm`** compact **inspector** rows (**`xs`** controls). **Universe risk:** lighter **Recharts** grids, borderless metric **Cards**, PCA tooltip **instant** + **open instrument** link + API note for sector; **empty** sector/industry placeholders with **Add tickers** / **Risk** actions + CSV roadmap copy. **Portfolio:** **Live / Risk** as **subtle** header actions; **compact** blend **`SegmentedControl`**; **dense** virtual ledger; toned **Go live** switch. **Instrument** financials/holders: **right-aligned** numeric columns, **tighter** table spacing. **Backtest results** / **movers** / **watchlist:** numeric **tabular** + **right** alignment where applicable. Docs: [`docs/ui/design-tokens.md`](docs/ui/design-tokens.md) persona + table rules.

- **UI — desk polish (dark, charts, data):** **cool-slate** dark surfaces and lighter borders ([`ui/src/mantine/cssVariablesResolver.ts`](ui/src/mantine/cssVariablesResolver.ts), [`ui/src/mantine/theme.ts`](ui/src/mantine/theme.ts), [`ui/src/theme.css`](ui/src/theme.css), [`ui/src/index.css`](ui/src/index.css)); **JetBrains-first** monospace stack + optional webfont ([`ui/index.html`](ui/index.html), [`ui/src/theme/typography.ts`](ui/src/theme/typography.ts)). **Recharts** line series default to **stepped** (`stepAfter`) instead of **`monotone`** where bucketed/discrete; **MacroStrip** uses a subtle **Area** under the sparkline; **L1** spread/mid chart shows **stepped bid/ask/mid** and a wider mid pane; bid/ask **bubble** chart splits **bid-heavy** vs **ask-heavy** scatters; live panel uses a **7/5 grid** and taller **OFI** strip. **Dashboard:** **`ApiErrorAlert`** outline/compact on macro strip; **HealthMiniCard** compact status row + outline errors. **Universes:** sector/industry empty states use **dashed** frames + **skeleton** silhouettes. **Portfolio** workspace: **live cockpit** as **teal light** fill, **risk** as **gray outline**. **Instrument:** underline-style **Tabs**, **compact** financial/holder number formatting ([`ui/src/lib/formatCompact.ts`](ui/src/lib/formatCompact.ts)). **Execution tracer:** **Grid** layout (progress vs microstructure), smaller **Progress**, larger bid/ask readout, static **demo OFI** strip. Docs: [`docs/ui/design-tokens.md`](docs/ui/design-tokens.md).

- **Market clock delivery:** **`GET /settings/market-clock`** remains for point-in-time reads; live updates use **WebSocket** **`/settings/market-clock/stream`** (**`hello`** then **`tick`** frames, same fields as GET). The UI header subscribes to this stream instead of HTTP polling. In-process subscribers use **`api.services.market_clock_hub.create_market_clock_subscription`** / **`release_market_clock_subscription`**. Tick interval **`SHUNYA_MARKET_CLOCK_TICK_SECONDS`** (default **1**, clamped **0.25–60**); a background task in **`api.main`** lifespan drives the hub alongside the backtest worker.

- **Instrument L1 — US RTH:** when **`us_rth_closed`** is returned on **`/instruments/.../stream/alpaca-l1`**, the API also **`publish_notification`** to **`/notifications/stream`**. The Live Data tab no longer shows a static “market closed” banner or disables **Connect** from the clock; it still warns if **`/settings/market-clock`** fails to load.

- **TLS / HTTP clients:** **`SHUNYA_TLS_VERIFY`** is the single switch for certificate verification on **yfinance** paths (`build_yfinance_session`) and **Alpaca** clients built via **`shunya.integration.alpaca_settings`** (REST and trading WebSocket). **Unset** or **`1`** / **`true`** / **`yes`** / **`on`** verifies certificates (new default for Yahoo: strict TLS unless you opt out). **`0`** / **`false`** / **`no`** / **`off`** disables verification (yfinance uses `curl_cffi` with `verify=False` when installed; Alpaca patches `requests` and passes an insecure SSL context to the trading stream). Corporate or Docker dev environments that relied on the old permissive Yahoo default without env should set **`SHUNYA_TLS_VERIFY=0`** explicitly.

- **Instrument OHLCV API:** **`GET /instruments/{symbol}/ohlcv`** responses no longer include **`data_source`**; clients should use **`provenance`** (`read_path`, `upstream_source_id`, `route_rule_id`, …). TypeScript types updated accordingly.

- **Instrument live WebSocket:** **`/instruments/{symbol}/stream/alpaca-bars`** no longer streams minute bars; it returns **`error`** with **`code: deprecated_stream`** and **`replacement_path`** to **`/instruments/{symbol}/stream/alpaca-l1`**, then closes. Clients must use **`alpaca-l1`** for IEX BBO quotes and trades.

- **Alpaca L1 WebSocket:** browser sessions are multiplexed onto a **single** shared **`StockDataStream`** per API key per API process (**`api.services.alpaca_l1_feed_hub`**), with **`SHUNYA_ALPACA_L1_MAX_SYMBOLS`** (default **30**) limiting distinct symbols; **`symbol_limit`** error when exceeded. Multiple uvicorn workers or replicas still open **one stream per process** each unless coordinated externally.

- **FinTs market data:** **`resolve_market_data_provider`** delegates to **`shunya.data.market_data.fints_bridge`** so eligibility matches **`resolve_market_route`** before choosing **`TimescaleMarketDataProvider`** / **`AlpacaHistoricalMarketDataProvider`**. **`resolve_market_route`** accepts **`timescale`** as a FinTs-oriented mode (stored Yahoo upstream in Timescale).

- **UI routing:** **`/`** is a **static landing page** (Matrix-style marketing, no app shell, no API calls from that screen). The research **dashboard** moved to **`/dashboard`** with sidebar label **Dashboard**, command palette entry, and header brand link targeting the desk. Docs: `docs/ui/overview.md`, `docs/ui/research.md`, `docs/ui/keyboard-shortcuts.md`.

- **UI — Data summary (`/data`):** replaced per-ticker coverage heatmap and instruments table with a **missing-coverage pie** (inverted completeness, top ten tickers in the legend); **risk vs log total return** scatter uses **`log_return_pct`** from the API, with **1st–99th percentile winsorization** on each axis for display (tooltips keep raw vol and log return). Docs: `docs/ui/research.md`, `docs/reference/http-api-package.md`.

- **Docker Compose:** the **`api`** service is **image-based** (no bind-mount + runtime **`uv sync`**); **`docker compose up`** builds and runs **TimescaleDB**, a one-shot **`bootstrap`** job, **`api`**, and **`UI`** (see `docs/quickstart.md`). Repo-root **`.env`** is passed into **`api`** via **`env_file`** (**`required: true`**); inline **`DATABASE_URL`** still overrides `.env` so the DB host stays **`timescaledb`** inside the network.
- **Onboarding:** new [Quickstart](https://kaushikdey647.github.io/shunya/quickstart/) doc (`docs/quickstart.md`) with paths for library-only, `./scripts/local-dev-all.sh`, and Docker Compose; MkDocs nav + docs home map row; README “above the fold” bullets and bootstrap copy; aligned “worker” wording with the in-process backtest loop in `api/main.py` across how-to and HTTP API docs; `docker-compose.yml` comment clarifying no separate worker container.
- **READMEs:** root `README.md` and `ui/README.md` trimmed to a short intro, quickstart, single documentation-site link, contributing, and license; detailed material moved to the published docs site.
- **GitHub Pages:** the published site must use **GitHub Actions** (MkDocs `site/` artifact). **Deploy from a branch** runs **Jekyll** on Markdown and drops the Material **sidebar**; the workflow adds **`site/.nojekyll`**. Step-by-step and recovery: **`docs/how-to/documentation-site-github-pages.md`**.
- **`scripts/local-dev-all.sh`:** optional **`--seed-alphas`** (runs `bootstrap_example_alphas.py` after migrate) and **`--help`**.
- **`scripts/bootstrap_example_alphas.py`:** **`--database-url`** for explicit Postgres URL (sets `DATABASE_URL` before API repository imports).
- **Packaging:** the **`api`** package is included in the **`shunya-py`** wheel (`pyproject.toml` → **`[tool.hatch.build.targets.wheel]`** `packages = ["shunya", "api"]`). The **`[api]`** optional extra still installs FastAPI, uvicorn, and related **runtime** libraries required to run **`uvicorn api.main:app`**.

- **UI:** backtest index empty-state hint pointed at a removed script name; it now references `scripts/bootstrap_sp100_timescale.py` / `scripts/bootstrap_ts_data.py`.
- MkDocs math: load **MathJax** via `extra_javascript` and `docs/javascripts/mathjax.js` so **Arithmatex** `\(…\)` / `\[…\]` renders (previously only the extension was enabled).
- MathJax config: use **`ignoreHtmlClass: ".*"`** (per PyMdown Arithmatex docs); **`".*|"`** breaks scanning so TeX stayed visible.
- MkDocs Mermaid: register **`pymdownx.superfences` → `custom_fences` → `mermaid2.fence_mermaid_custom`** so ` ```mermaid ` blocks are not converted to generic Pygments code fences (which prevented **mkdocs-mermaid2** from running).

### Fixed

- **UI — Trade account:** Alpaca-backed queries use an **applied** token only (separate draft input + **Apply token**); React Query no longer refetches on every keystroke or spams notifications.

- **Alpaca L1 hub:** Alpaca **control-stream** messages with **`T: error`** (previously only **`log.error`** in `alpaca-py`) are forwarded to every attached browser as **`error`** with **`code: alpaca_upstream`**. **`T: subscription`** acks are forwarded as **`type: subscription`** with a JSON-safe **`alpaca`** payload. **`ShunyaL1StockDataStream`** overrides **`_dispatch`** to invoke the hub hook before the default handler; **`INFO`** logs when the shared market-data WebSocket is up (**`endpoint=…`**) and on the **first** quote or trade fanout.

- **Alpaca L1 hub:** fatal Alpaca market-data errors (**connection limit exceeded**, **insufficient subscription**) no longer spin a tight reconnect loop in `alpaca-py`; the shared stream exits and browsers receive **`error`** / **`code: alpaca_market_data_stopped`**. **`build_stock_data_stream(..., stream_cls=...)`** injects **`ShunyaL1StockDataStream`**. Subscribe/unsubscribe while the stream is live runs in **`asyncio.to_thread`** so **`run_coroutine_threadsafe(...).result()`** never blocks the uvicorn event loop; **`detach`** yields before Alpaca unsubscribe so a quick re-attach for the same symbol is not torn down by mistake. After teardown, opening a **new** shared Alpaca session waits **`SHUNYA_ALPACA_L1_RECONNECT_COOLDOWN_SEC`** (default **5**; **`0`** in hub unit tests) so Alpaca can release the prior TCP slot; extra browser tabs joining an **already live** stream are not delayed by that wait.

- **UI — Live Data:** IEX L1 mid/spread **lightweight-charts** panes no longer throw **`Value is null`** (black screen) when multiple quotes share the same **UTCTimestamp** after second-level time bucketing; charts dedupe to one sample per UTC second and validate finite BBO fields before ingest.

### Deprecated

- **`YFINANCE_TLS_VERIFY`** — replaced by **`SHUNYA_TLS_VERIFY`** for yfinance and Alpaca together; it is no longer read.

### Removed

- `shunya.streaming` (Alpaca-oriented websocket helpers, micro-bars, rectangular snapshots).
- `StreamingRunner`, `StreamingContextBuilder`, and `StreamingDecision`.
- `FinTrade` (`shunya/algorithm/fintrade.py`).
- `tests/test_streaming_pipeline.py` and `tests/test_fintrade.py`.
- **Instrument live minute bars over WebSocket:** **`/instruments/{symbol}/stream/alpaca-bars`** no longer streams **`bars`** / **`updatedBars`**; use **`alpaca-l1`**.

### Added

- **Instrument Alpaca L1 (IEX):** **`WebSocket /instruments/{symbol}/stream/alpaca-l1`** when **`SHUNYA_API_ALPACA_ENABLED`** and Alpaca keys are configured — **`StockDataStream`** **IEX** **`subscribe_quotes`** + **`subscribe_trades`**, plus trade correction/cancel forwarding; JSON via **`api.services.alpaca_l1_payload`**, router **`api.routers.instrument_l1_stream`**, process-wide multiplexing in **`api.services.alpaca_l1_feed_hub`** (one Alpaca TCP session per key per API process; **`SHUNYA_ALPACA_L1_MAX_SYMBOLS`** default 30). UI **Live Data** tab uses **`LiveL1Provider`**, ring buffers, stepped mid/spread (lightweight-charts **`LineType.WithSteps`**), Recharts bubble + histogram, tape. Stocks and ETFs only.

- **Market data routing:** `shunya.data.market_data` (pure `resolve_market_route` + `MarketRouteDecision`), `shunya.data.market_router.try_timescale_then_live_ohlcv`, explicit Alpaca **`StockBarsRequest.feed`** (`SHUNYA_ALPACA_BAR_FEED` / `bar_feed_upstream`), **`GET /instruments/{symbol}/ohlcv?route=`**, **`provenance`** on instrument OHLCV and market snapshot rows, FinTs **`market_data_provider`** **`alpaca`** / **`best_effort`**, **`FinTsRequest.schema_version`**, migration **`016_ohlcv_bars_metadata.sql`**, **`STORED_OHLCV_DEFAULT_UPSTREAM_ID`** for stored Yahoo **`ohlcv_bars.source`**, and **`shunya.integration.yahoo_public.YahooPublicAdapter`** for non-router Yahoo HTTP reads. Doc: **`docs/market_data_routing.md`**.

- **Docker Compose / bootstrap:** one-shot **`bootstrap`** runs **`migrate`**, **`docker/compose_bootstrap_probe.py`**, then **`scripts/gapfill_sp100_universe_metadata.py`** when OHLCV is already present, or **`scripts/bootstrap_sp100_timescale.py --skip-migrate`**, **`bootstrap_example_alphas.py`**, **`bootstrap_ts_data.py`**, and **`gapfill_sp100_universe_metadata.py`** on a fresh/partial DB. **`api`** **`depends_on`** **`bootstrap`** (**`service_completed_successfully`**). Probe: **`SHUNYA_COMPOSE_BOOTSTRAP_MIN_SP100_BARS`**. **`Dockerfile`** copies **`scripts/`** and **`examples/`**. Docs: **`docs/data_timescale.md`**, **`docs/how-to/bootstrap-scripts.md`**.

- **Scripts:** **`scripts/gapfill_sp100_universe_metadata.py`** — Yahoo backfill for **`symbol_classifications`** + **`fundamentals_daily`** on SP100 members (saved universe overview); skips when counts show no gaps unless **`--force`**.
- **`scripts/diag_alpaca_l1_ws.py`** — CLI WebSocket client for **`/instruments/{symbol}/stream/alpaca-l1`** (prints `hello`, `quote`, `subscription`, `error`, …). See **`docs/reference/http-api-package.md`**.

- **Docker full stack:** repo-root **`Dockerfile`** (FastAPI + `shunya` via **`uv sync --frozen`** with **`api`** and **`timescale`** extras), **`docker/api-entrypoint.sh`** (optional **`RUN_MIGRATIONS=1`** → **`shunya-timescale migrate`** before uvicorn), **`docker/compose-bootstrap.sh`** + **`docker/compose_bootstrap_probe.py`**, **`ui/Dockerfile`** (Vite build + **nginx** proxying **`/api/`** to the **`api`** service), **`.dockerignore`** files, and **`docker-compose.yml`** services **`timescaledb`**, **`bootstrap`**, **`api`**, **`ui`** (UI on host port **8080**, API on **8000**). **`ui/vite.config.ts`** reads **`API_PROXY_TARGET`** for the dev proxy (defaults to **`http://127.0.0.1:8000`**) so a future Vite-in-container setup can target **`http://api:8000`**.

- **API:** **`log_return_pct`** on **`TickerRiskRow`** (and dashboard ticker rows): log total return over the window as **`100 * ln(c_last / c_first)`** when both endpoint closes are strictly positive and finite; otherwise **`null`**. Computed in **`api.risk_metrics.per_bar_return_stats_with_ppy`** and returned from **`POST /data`** and **`GET /data/dashboard`**.

- **UI (keyboard):** **⇧Space** focuses ticker search (replaces ⌘Space for macOS Spotlight compatibility); **⇧↑⇧↓** cycles primary sidebar routes and moves highlights in the command palette, ticker list, and data tables (skipped in modals / inputs / Monaco where applicable); **⌘/Ctrl+Enter** on tables and Alpha Studio **Run backtest**. User doc: `docs/ui/keyboard-shortcuts.md` (MkDocs **Web application** nav). Details in `ui/CHANGELOG.md`.

- **API + UI:** **`GET /universes/{id}/return-analytics`** — Timescale daily OHLCV panel for members: simple and log return **correlation matrices**, **cross-sectional volatility** time series, **PCA** (variance share, PC1 scores, PC1–PC2 loadings scatter), and **HHI / CR5 / CR10** from latest market-cap weights (equal-weight fallback). Universe detail page adds **Risk & structure** tab (period 1y / 2y / 5y).

- **Timescale:** migration **`015_api_universe_sp100.sql`** seeds saved universe **`SP100`** in **`api_universes`** and fills **`api_universe_members`** from **`symbol_index_membership`** where **`index_code = 'SP100'`** (idempotent). Run **`sync-index-memberships`** before or after migrate so membership rows exist; re-run **`migrate`** after sync to attach constituents.

- **Documentation:** [Documentation site & GitHub Pages](https://kaushikdey647.github.io/shunya/how-to/documentation-site-github-pages/) (`docs/how-to/documentation-site-github-pages.md`) — local **`mkdocs`** commands, why the **Material sidebar disappears** when Pages is set to **Deploy from a branch**, and that **GitHub Actions** + the **`Deploy documentation`** workflow (with **`site/.nojekyll`**) is required. Linked from [Install](https://kaushikdey647.github.io/shunya/install/) and the docs home map; MkDocs nav entry under How-to guides.
- **Documentation:** [Reference → HTTP API (overview + `api.main`)](https://kaushikdey647.github.io/shunya/reference/http-api-package/) — merged former top-level **`docs/http-api.md`** into **Reference** (includes [**universe return analytics**](https://kaushikdey647.github.io/shunya/reference/http-api-package/#universe-return-analytics)); added **mkdocstrings** [`api.main`](https://kaushikdey647.github.io/shunya/reference/api-library/) page. **`api`** is included in the **`shunya-py`** wheel (`hatch` `packages`). Docs CI uses **`uv sync --group docs --extra api`**. Cross-links updated across how-tos and UI docs; [Studio → Risk & structure](https://kaushikdey647.github.io/shunya/ui/studio/#risk-structure-tab) documents the tab.
- **Documentation:** embed UI screenshots from `docs/*.png` in the root and `ui/` READMEs, [Web application overview](docs/ui/overview.md), [Research](docs/ui/research.md), [Studio](docs/ui/studio.md), [Trade](docs/ui/trade.md), [Backtest from the web UI](docs/how-to/backtest-from-ui.md), and [Execution: OMS, EMS](docs/documentation/oms-ems.md).
- **Custom universes (Timescale + API + UI):** migration **`014_api_universes.sql`** (`api_universes`, `api_universe_members`, `api_alphas.default_universe_id`); **`/universes`** CRUD, members, flat tickers, summary analytics; **`POST /backtests`** supports **`universe_id`** + **`benchmark_ticker`** (mutually exclusive with **`index_code`**); job list exposes **`universe_id`**; yfinance OHLCV backfill treats index and saved-universe jobs the same; **`POST /trade/paper/cycle`** accepts optional **`universe_resolution_note`** echoed in **`messages`**.
- **Documentation site:** split **Concepts** (quant finance + how it maps to Shunya) vs **Documentation** (code, types, APIs); new pages under `docs/concepts/` and `docs/documentation/`; MkDocs **Mermaid** via `mkdocs-mermaid2-plugin` with vendored `docs/javascripts/mermaid.min.js` for strict/offline-friendly builds; **MathJax** via `pymdownx.arithmatex` and `theme.features: content.math.mathjax`.
- **`scripts/local-dev-all.sh`** — single entrypoint for local dev: prerequisite checks, root **`.env`** bootstrap, Docker **TimescaleDB**, **`uv sync`**, **`shunya-timescale migrate`**, API + Vite UI (see [Local development: API, worker, and UI](docs/how-to/local-dev-api-ui.md)).
- **OMS / EMS (institutional execution split):**
  - `shunya/oms` — share-based reconciliation (`required_delta_shares`), `ParentOrder` FSM via `transitions`, `InMemoryLedger`, `InstitutionalOMS`, Alpaca trade stream bridge (`AlpacaOMSTradeStream`), REST position snapshot (`rest_snapshot`), SQLAlchemy persistence (`shunya/oms/db`), Alembic migration `001_oms_tables`, and `risk_bridge` helpers for `PortfolioRiskEngine` outputs.
  - `shunya/ems` — `BrokerGateway` / `AlpacaBrokerGateway`, TWAP/VWAP schedulers (`twap_slice_quantities`, `vwap_slice_quantities`, optional `smooth_volume_profile_jax`), micro-pricing (`limit_price_for_child`), child `client_order_id` scheme (`child_client_order_id`), and async `EMSParentRunner` (limit submit, timeout, cancel, urgency escalation).
  - Tests: `tests/test_oms.py`, `tests/test_ems.py`, optional Docker `tests/test_oms_db.py`.
- **Pre-trade risk:** `shunya/algorithm/risk_engine.py` — `PortfolioRiskEngine`, `RiskVetConfig` / `RiskVetResult`, `RiskEngineState`, `DrawdownSentinel`, `ShortabilityMode`, optional Ledoit–Wolf / CVXPY integration when the **`[risk]`** extra is installed (`pip install "shunya-py[risk]"` or `uv sync --extra risk`).
- Repo-root **Alembic** layout (`alembic.ini`, `alembic/env.py`, `alembic/versions/001_oms_tables.py`) for OMS SQL migrations alongside existing Timescale CLI migrations.
- Dependencies: `transitions`, `sqlalchemy`; dev: `alembic`, `psycopg[binary]`.
- `PortfolioConstructionService`, `PortfolioConstructionResult`, `TargetBlendConfig`, `AlphaBlendConfig`, `BlendModeKind`, `TickerUniversePolicy`, `StrategyReturnFeed`, and `mark_to_market_strategy_pnl_usd` in `shunya/algorithm/portfolio_manager.py`. Alpha-blend **correlation dampening** applies a **capital haircut** to the master (`active_capital`) instead of renormalizing convictions (previous `z` scaling was a no-op). `PortfolioManager` / `AlphaBlendPortfolioManager` delegate to the shared service; they join `RollingSharpeTracker`, `combine_weighted_targets`, `sum_target_maps`, and `PORTFOLIO_PERF_KEY` in the same module.
- `tests/test_portfolio_manager.py`.
- Broker-neutral open-order snapshot surface:
  - `OpenOrderView`
  - `ExecutionAdapter.list_open_orders()`
  - `AlpacaExecutionAdapter.list_open_orders()`
  - `KiteExecutionAdapter.list_open_orders()`
- Context-based alpha authoring API in `shunya.algorithm.alpha_context`:
  - `AlphaContext` with canonical series fields (`open`, `high`, `low`, `close`, `adj_volume`)
  - `AlphaSeries` wrapper for history tensors
  - namespaced operators via `ctx.ts.*` and `ctx.cs.*`
- New example alpha package under `examples/alphas`:
  - `sma_ratio_50`
  - `mean_reversion_5`
  - `breakout_20`
- New research notebook `examples/alpha_benchmark_oex.ipynb` to compare example alpha returns/correlations/metrics against `^OEX` and against each other.
- `BarIndexPolicy` and `default_bar_index_policy()`: market data indices default to **America/New_York** (session-stable intraday labels across US DST), with optional `timezone`, `naive`, and `daily_anchor` (`"timezone"` vs legacy `"utc"` for daily-like bars).
- `finTs(..., bar_index_policy=...)`, `finTs.bar_index_policy`, and `MarketDataProvider.download(..., bar_index_policy=...)`.
- `bounds_for_validation()` for timezone-consistent `start_date` / `end_date` windows.
- Trading-time axis primitives in `shunya.data.timeframes`:
  - `build_trading_calendar(...)` for canonical US-equities bar grids across minute/hour/day cadences.
  - `timestamp_is_on_trading_grid(...)` and `trading_time_distance(...)` for grid membership and trading-time deltas.
- **BRAIN-style pipeline parity (math + data hygiene):**
  - `finTs.align_universe(...)` — intersect trading calendars, drop or raise on incomplete tickers, dense `(Ticker, Date)` reindex; returns `PanelAlignReport`; sets `_aligned_calendar`.
  - `finTs.get_trading_calendar()` / `finTs.execution_lag_calendar_date(..., lag=)` — calendar-aware **Delay1**-style lag.
  - `PanelAlignReport` (exported from `shunya` / `shunya.data`).
  - `FinStrat`: `decay_mode` (`"ema"` or `"linear"`), `decay_window`, `signal_delay`, `nan_policy`, `panel_date_for_execution()`, linear decay on raw scores.
  - `FinBT`: OHLCV feeds reindexed to a shared calendar; `validate_finite_targets` (default on).
- `cross_section`: finite-safe `neutralize_market` / `neutralize_groups` / `winsorize`; `rank` sorts non-finite last.
- Notebook examples: `vwap_close_rank_backtest_yfinance.ipynb` and `vwap_close_rank_backtest.ipynb` call `align_universe` after `finTs(...)`.
- yfinance-based classification mapping for `Sector`, `Industry`, and `SubIndustry` with deterministic fallback labels.
- New `finTs` controls:
  - `classifications=...`
  - `attach_yfinance_classifications=True`
- Group defaults and validation improvements for neutralization paths in backtest/trading flows.
- Paper-safe execution status observation fields in `OrderAttempt` / `ExecutionReport`:
  - initial/final status
  - fill quantity and average fill price
  - status polling errors
- Optional sector gross cap enforcement in shared target helpers and integration in `FinBT`.
- Session-aware decision-time guardrails:
  - weekend and future-date checks
  - strict same-session option
  - staleness warnings
- Data QA diagnostics in `finTs`:
  - duplicate row detection
  - missing ticker/date coverage checks
  - stale panel checks
  - invalid OHLCV row checks
- New `finTs` controls:
  - `trading_axis_mode` (`"observed"` or `"canonical"`) for calendar/lag helpers.
  - `strict_trading_grid` for off-grid/holey timestamp validation when strict loading is enabled.
- Richer backtest analytics:
  - turnover history and summary metrics
  - concentration metrics
  - group exposure snapshots
- Reconciliation loop and remediation hooks in live/paper trading:
  - `warn_only`
  - `retry_once`
  - `cancel_and_retarget`
- Additional shared constraints:
  - group net caps
  - turnover budget enforcement
  - ADV participation caps
- New documentation:
  - `CONTRIBUTING.md` (contributor guide; previously `CONTRIBUTION.md`)
  - expanded `README.md` sections for controls, diagnostics, and roadmap status

### Changed

- **Breaking:** removed `FinTrade`, the entire `shunya.streaming` package, `StreamingRunner`, and related public exports; use `PortfolioManager` plus `AlpacaExecutionAdapter` / `OrderManager` from your own scheduler.
- `FinStrat` now exposes reusable seams for non-`finTs` runners:
  - `scores_from_context(ctx)`
  - `process_raw_scores(raw_scores, capital, ...)`
- `README.md` documents portfolio construction and explicitly calls out removed streaming / `FinTrade` orchestration.
- **Breaking:** `FinStrat` now executes context-style alpha callables (`algorithm(ctx)`), replacing legacy panel-index authoring (`algorithm(panel)` / `IX_*` flow).
- `FinBT` routes signal execution through context-based alpha evaluation while preserving downstream sizing/neutralization controls.
- **Breaking:** `normalize_history_index` and bundled providers align Yahoo/Alpaca timestamps to `BarIndexPolicy` (default **America/New_York**), not forced UTC-naive. Use `BarIndexPolicy(timezone="UTC")` and `daily_anchor="utc"` to recover older daily-like alignment.
- `validate_core_ohlcv_coverage(..., bar_index_policy=...)` interprets coverage windows in the policy timezone for intraday and daily-like bars.
- **Breaking:** The installable Python package is published on PyPI as **`shunya-py`** (the name `shunya` was already registered by another project). Import the library as **`shunya`** (`from shunya import finTs`, etc.), not `src`.
- `FinStrat.__init__` includes `decay_mode`, `decay_window`, `signal_delay`, `nan_policy`; temporal smoothing requires `tickers` in `pass_` when EMA or multi-day linear decay is active.
- `FinStrat` adds `temporal_mode` (`"bar_step"` or `"elapsed_trading_time"`). Elapsed mode advances decay by trading-time gaps rather than one-step-per-observed-bar.
- `FinStrat.pass_` accepts optional `execution_date`; `FinBT` passes execution timestamps for trading-time-aware decay.
- `FinStrat.panel_at` / `group_labels_at` respect `signal_delay` (execution date → lagged panel date on `get_trading_calendar()`).
- `FinBT._ohlcv_frames` uses `finTs._aligned_calendar` or the intersection of per-ticker indices.
- Public exports updated in `shunya/__init__.py` and `shunya/algorithm/__init__.py` (`PanelAlignReport`, helpers, diagnostics types).

### Documentation

- **Web UI install:** documented **`ui/`** setup (**Node.js 20+**, **`npm ci`**, **`npm run dev`**) in `docs/install.md`, root `README.md` (Requirements), `api/README.md` (under Install), `docs/how-to/local-dev-api-ui.md` (prerequisites + section title), and `docs/ui/overview.md`; `docs/index.md` install row now mentions the web UI.

### Testing

- Added tests:
  - `tests/test_risk_engine.py`
  - `tests/test_portfolio_manager.py`
  - `tests/test_fints_classification.py`
  - `tests/test_data_qa.py`
  - `tests/test_execution_adapter.py`
  - `tests/test_constraints.py`
  - `tests/test_integration_rebalance.py`
  - `tests/test_panel_align.py`
  - `tests/test_brain_pipeline.py`
  - `tests/test_timeframes.py`
  - `tests/test_fints_validation.py`
  - `tests/test_providers.py`
- Expanded tests:
  - `tests/test_decision.py`
  - `tests/test_finbt.py`
  - `tests/test_finstrat.py`
  - `tests/test_targets.py`
  - `tests/test_execution_adapter.py`
  - `tests/test_kite_execution.py`
- Added trading-time coverage:
  - canonical calendar generation and weekend gap handling
  - strict trading-grid validation on off-grid timestamps
  - elapsed-trading-time decay weighting vs bar-step mode
  - intraday lag parity checks across minute/hour bars

### Not Yet Implemented

- No in-repo websocket market-data client; refresh panels or marks on your own schedule.
- `OrderManager` remains an in-memory helper for open-order snapshots unless you mirror it externally; OMS **parent/child** rows are optional via SQLAlchemy + Alembic when you provide `DATABASE_URL` and run migrations — there is still no turnkey multi-process event bus or hosted OMS service in this repository.
