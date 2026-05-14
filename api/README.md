# Backtest API

Repo-local **FastAPI** service (not part of the published `shunya-py` wheel) for:

- **`/alphas`** — CRUD on alpha definitions: optional **inline** `source_code` (Python saved in `api_alphas`, executed in the worker) or module **`import_ref`** (allow-list: `examples.alphas.<module>:alpha` when not using inline). At least one must be set. If `source_code` is non-empty, it takes precedence at backtest time. Run migration **`003_alpha_source_code.sql`** (via `shunya-timescale migrate`) for the `source_code` column. Stored `finstrat_config` JSON.
- **`/indices`** — List equity indexes from Timescale (`equity_indexes` + membership counts) with **raw index tickers** (e.g. `^GSPC`, `^BFX`) as `benchmark_ticker` for comparison series.
- **`/backtests`** — Enqueue async jobs (`POST`), list/get status, fetch JSON results when `succeeded` (same shape as `FinBT.results(show=False)` plus optional `benchmark`). **`POST` body** always uses the fixed simulation window **`[2020-01-01, 2026-01-01)`** (end exclusive) and **daily** bars; client-supplied `fin_ts.start_date`, `end_date`, and `bar_spec` are overwritten. **`include_test_period_in_results`** (default `false`): when `false`, stored metrics and time series exclude the **test** slice from **2025-01-01** onward (tune-only view). **`POST` with `index_code`**: resolves constituents from `symbol_index_membership`, sets `benchmark_ticker` to the catalog **raw index** symbol, forces **`market_data_provider=timescale`** and **no Yahoo** for the run. By default OHLCV must exist for **every** constituent plus the benchmark; set **`omit_index_members_missing_ohlcv`: true** to drop members with no bars in the window (benchmark still required — ingest e.g. `^NDX` / `^GSPC` if missing).
- **`POST /data`** — Panel NaN counts per ticker/column and per-ticker return / annualized vol / Sharpe / Sortino from `finTs` (Timescale when `DATABASE_URL` is set and `market_data_provider` is `auto` or `timescale`).
- **`GET /data/dashboard`** — Database-wide analytics for a stored `interval` / `source`: reference window `[MIN(ts), MAX(ts)]` over `ohlcv_bars`, per-ticker completeness vs that window (heatmap buckets), aggregated risk/return metrics from stored closes, and completeness histogram bins. Bucket granularity defaults to **`auto`** (chooses day, week, or month so the heatmap stays within `max_buckets`, default **200**); adjacent periods may be merged (logical OR). Optional **`SHUNYA_DASHBOARD_MAX_TICKERS`** caps symbols (alphabetical order).
- **`/instruments/...`** — Search and OHLCV: prefers Timescale when the DB is reachable and coverage is complete; otherwise yfinance (with optional write-back to Timescale).
- **`/market/...`** — Dashboard-oriented market data: **`POST /market/snapshot`** (batched daily OHLCV-derived quotes for macro strip / watchlists), **`GET /market/movers`** (`kind=gainers|losers|active`, Yahoo predefined screeners), **`GET /market/headlines`** (general financial headlines via Yahoo Search). Implemented in `api/services/market_*.py`.
- **`POST /trade/paper/cycle`** — One **paper** cycle: `PortfolioRiskEngine` → `InstitutionalOMS` → `EMSParentRunner` + Alpaca trade stream. Requires **`SHUNYA_API_ALPACA_ENABLED=1`**, Alpaca keys in the environment (`APCA_*` or `SHUNYA_ALPACA_*`), and **`SHUNYA_API_TRADE_DESK_TOKEN`**; send header **`X-Shunya-Trade-Desk-Token`** with that same value. Body: `capital`, `execution_date` (YYYY-MM-DD), either **`use_demo_pcs: true`** (built-in SPY/QQQ stub book) or **`targets_usd` + `universe` + `prices`** for a fixed-target run. See [`api/routers/trade_desk.py`](routers/trade_desk.py).
- **`GET /settings/app`** — Read-only **environment** flags (no secrets) plus **effective runtime tunables** (env defaults merged with the optional DB overlay in `api_runtime_config`) and per-field **`sources`** (`database` vs `environment`). Does not require the trade-desk token.
- **`PATCH /settings/app`** — Merge **non-secret operator tunables** into `api_runtime_config` (same keys as `SHUNYA_API_*` caps such as worker poll interval, serialization caps, Ollama model/timeout, cache TTL). Requires **`DATABASE_URL`** (or equivalent) and migration **`013_api_runtime_config.sql`**. **Auth:** same **`X-Shunya-Trade-Desk-Token`** as **`POST /trade/paper/cycle`** when **`SHUNYA_API_TRADE_DESK_TOKEN`** is set; if that token is unset, returns **503** (writes disabled). Secrets (DB URL, Alpaca keys, `SHUNYA_API_OLLAMA_HOST`, CORS, trade-desk token) are **never** stored in the overlay.

## Install

From the repo root:

```bash
uv sync --extra api --extra timescale
```

### Web UI (optional)

The **`ui/`** React app uses **Node.js 20+** and **npm**, separate from the Python venv. From the repository root (with the API running on port **8000** for dev):

```bash
cd ui
npm ci
npm run dev
```

Vite prints a local URL (default **http://localhost:5173**); dev mode proxies **`/api`** to **`http://127.0.0.1:8000`**. See [`ui/README.md`](../ui/README.md) and the docs **[Quickstart](https://kaushikdey647.github.io/shunya/quickstart/)** / **[Local development: API, worker, and UI](https://kaushikdey647.github.io/shunya/how-to/local-dev-api-ui/)**.

Set `DATABASE_URL` (or `SHUNYA_DATABASE_URL`) to your Postgres URL, apply migrations (including **`013_api_runtime_config.sql`** for `GET`/`PATCH /settings/app`), then start the app:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
shunya-timescale migrate
shunya-timescale sync-index-memberships
uv run python scripts/bootstrap_example_alphas.py
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### TimescaleDB (local, optional)

For a **quick local Timescale / Postgres** stack (Docker-based), you can use Timescale’s installer:

```bash
curl -sL https://tsdb.co/start-local | sh
```

Follow the script’s output for host, port, user, password, and database name, then point `DATABASE_URL` at that instance, for example:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@localhost:PORT/DATABASE'
```

If the installer creates only the default `postgres` database, either use that URL or create an app database (e.g. `createdb shunya`) and set `DATABASE_URL` accordingly. After `DATABASE_URL` is set, run **`shunya-timescale migrate`** once before starting the API.

Without Timescale (or with the DB down), instrument OHLCV and related paths still work via **yfinance**; see **Environment** below for TLS (`curl-cffi` / `YFINANCE_TLS_VERIFY`).

Alternatively, use Docker Compose in this repo (`api` service mounts the repo and can target a `timescaledb` service) if you already use that layout.

## Environment

At process start the API loads a **repo-root `.env`** (if present) via `python-dotenv`, then **pydantic-settings** with prefix **`SHUNYA_API_`** (see [`api/settings.py`](settings.py)). Changing `.env` still requires an **API restart**; values that must update live without restart belong in the **`api_runtime_config`** overlay (via **`PATCH /settings/app`**), not in `.env` edits from clients.

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` / `SHUNYA_DATABASE_URL` | Postgres for API tables + Timescale OHLCV reads |
| `SHUNYA_CORS_ORIGINS` | Optional comma-separated browser origins for cross-origin frontends (e.g. `https://app.vercel.app`). Omit if the UI is same-origin or only server-side clients call the API. |
| `SHUNYA_RUN_TIMESCALE_CONTAINER` | Set to `1` to run Timescale-backed tests via Docker testcontainers when `DATABASE_URL` is unset, and for an **isolated** DB for `test_alphas_crud_and_backtest_job` (ignores shared `DATABASE_URL` for that fixture). |
| `SHUNYA_API_INTEGRATION_DATABASE_URL` | Optional **dedicated** Postgres URL for queue-based API integration tests (`test_alphas_crud_and_backtest_job`). Use when you cannot run Docker testcontainers but must not share the job queue with another process (e.g. a running `uvicorn` on the same `DATABASE_URL`). |
| `SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS` | Set to `1` only with `DATABASE_URL` / `SHUNYA_DATABASE_URL` when **no** other API worker process uses that database; otherwise job-queue tests can flake or fail. Prefer testcontainers or `SHUNYA_API_INTEGRATION_DATABASE_URL`. |
| `SHUNYA_DASHBOARD_MAX_TICKERS` | Optional cap for `GET /data/dashboard` symbol list (positive integer); omit for no cap |
| `SHUNYA_API_ALPACA_ENABLED` | When `true` / `1`, the API builds shared Alpaca clients at startup (requires `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` or `SHUNYA_ALPACA_*` aliases). |
| `SHUNYA_API_TRADE_DESK_TOKEN` | Shared secret for `POST /trade/paper/cycle` and **`PATCH /settings/app`**; required header `X-Shunya-Trade-Desk-Token` must match exactly. |
| `SHUNYA_API_OLLAMA_HOST` | Base URL for Ollama (alpha assist); not stored in DB overlay. |
| `SHUNYA_API_OLLAMA_MODEL` | Default Ollama model id; may be overridden by **`api_runtime_config`** when set via **`PATCH /settings/app`**. |
| `SHUNYA_API_OLLAMA_TIMEOUT_SECONDS` | Default HTTP timeout for Ollama; may be overridden by the DB overlay. |
| `SHUNYA_API_DATABASE_URL` | Optional override (via `pydantic-settings`) |
| `SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS` | Worker poll interval (default `1.0`) |
| `SHUNYA_API_INDEX_OHLCV_BACKFILL_BATCH_SIZE` | Tickers per Yahoo batch when the worker backfills OHLCV after a recoverable index backtest data error (default `40`). |
| `YFINANCE_TLS_VERIFY` | If set to `1` / `true` / `yes` / `on`, yfinance uses default TLS verification instead of the `curl_cffi` session with `verify=False` (useful outside corporate TLS inspection). |

## Optional: prune stored OHLCV to the HTTP backtest window

HTTP backtests use daily bars in **`[2020-01-01, 2026-01-01)`** (end exclusive). To drop older or newer `ohlcv_bars` rows so stored data matches that policy (adjust if your `ts` semantics differ), run SQL once as an operator:

```sql
DELETE FROM ohlcv_bars
WHERE ts < TIMESTAMPTZ '2020-01-01'
   OR ts >= TIMESTAMPTZ '2026-01-01';
```

Re-bootstrap with `scripts/bootstrap_ts_data.py` (defaults use the same window) to refill the canonical range.

## Tests

- Unit: `pytest tests/test_api/ -m "not timescale"`
- DB + HTTP integration: `pytest tests/test_api/ -m timescale` with `DATABASE_URL` set, or `SHUNYA_RUN_TIMESCALE_CONTAINER=1` and Docker (skips if Docker is unavailable).
- **`test_alphas_crud_and_backtest_job`** enqueues `api_backtest_jobs` rows; any other process using the same database URL and running the API worker can **claim the same job** (`FOR UPDATE SKIP LOCKED`). That produces failures whose traceback points at `api/worker.py` even when pytest patches the in-process worker. Use **`SHUNYA_RUN_TIMESCALE_CONTAINER=1`** (isolated Timescale container for that test, even if `DATABASE_URL` is set) or **`SHUNYA_API_INTEGRATION_DATABASE_URL`** pointing at a database **no live API** is connected to. Only if you are sure nothing else is polling the queue: **`SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS=1`** with `DATABASE_URL`.
