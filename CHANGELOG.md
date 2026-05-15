# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

- **UI routing:** **`/`** is a **static landing page** (Matrix-style marketing, no app shell, no API calls from that screen). The research **dashboard** moved to **`/dashboard`** with sidebar label **Dashboard**, command palette entry, and header brand link targeting the desk. Docs: `docs/ui/overview.md`, `docs/ui/research.md`, `docs/ui/keyboard-shortcuts.md`.

- **UI — Data summary (`/data`):** replaced per-ticker coverage heatmap and instruments table with a **missing-coverage pie** (inverted completeness, top ten tickers in the legend); **risk vs log total return** scatter uses **`log_return_pct`** from the API, with **1st–99th percentile winsorization** on each axis for display (tooltips keep raw vol and log return). Docs: `docs/ui/research.md`, `docs/reference/http-api-package.md`.

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

### Removed

- `shunya.streaming` (Alpaca-oriented websocket helpers, micro-bars, rectangular snapshots).
- `StreamingRunner`, `StreamingContextBuilder`, and `StreamingDecision`.
- `FinTrade` (`shunya/algorithm/fintrade.py`).
- `tests/test_streaming_pipeline.py` and `tests/test_fintrade.py`.

### Added

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
