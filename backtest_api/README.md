# Backtest API

Repo-local **FastAPI** service (not part of the published `shunya-py` wheel) for:

- **`/alphas`** — CRUD on alpha definitions: optional **inline** `source_code` (Python saved in `api_alphas`, executed in the worker) or module **`import_ref`** (allow-list: `examples.alphas.<module>:alpha` when not using inline). At least one must be set. If `source_code` is non-empty, it takes precedence at backtest time. Run migration **`003_alpha_source_code.sql`** (via `shunya-timescale migrate`) for the `source_code` column. Stored `finstrat_config` JSON.
- **`/indices`** — List equity indexes from Timescale (`equity_indexes` + membership counts) with **raw index tickers** (e.g. `^GSPC`, `^BFX`) as `benchmark_ticker` for comparison series.
- **`/backtests`** — Enqueue async jobs (`POST`), list/get status, fetch JSON results when `succeeded` (same shape as `FinBT.results(show=False)` plus optional `benchmark`). **`POST` body** always uses the fixed simulation window **`[2020-01-01, 2026-01-01)`** (end exclusive) and **daily** bars; client-supplied `fin_ts.start_date`, `end_date`, and `bar_spec` are overwritten. **`include_test_period_in_results`** (default `false`): when `false`, stored metrics and time series exclude the **test** slice from **2025-01-01** onward (tune-only view). **`POST` with `index_code`**: resolves constituents from `symbol_index_membership`, sets `benchmark_ticker` to the catalog **raw index** symbol, forces **`market_data_provider=timescale`** and **no Yahoo** for the run. By default OHLCV must exist for **every** constituent plus the benchmark; set **`omit_index_members_missing_ohlcv`: true** to drop members with no bars in the window (benchmark still required — ingest e.g. `^NDX` / `^GSPC` if missing).
- **`POST /data`** — Panel NaN counts per ticker/column and per-ticker return / annualized vol / Sharpe / Sortino from `finTs` (Timescale when `DATABASE_URL` is set and `market_data_provider` is `auto` or `timescale`).
- **`GET /data/dashboard`** — Database-wide analytics for a stored `interval` / `source`: reference window `[MIN(ts), MAX(ts)]` over `ohlcv_bars`, per-ticker completeness vs that window (heatmap buckets), aggregated risk/return metrics from stored closes, and completeness histogram bins. Bucket granularity defaults to **`auto`** (chooses day, week, or month so the heatmap stays within `max_buckets`, default **200**); adjacent periods may be merged (logical OR). Optional **`SHUNYA_DASHBOARD_MAX_TICKERS`** caps symbols (alphabetical order).
- **`/instruments/...`** — Search and OHLCV: prefers Timescale when the DB is reachable and coverage is complete; otherwise yfinance (with optional write-back to Timescale).

## Install

From the repo root:

```bash
uv sync --extra api --extra timescale
```

Set `DATABASE_URL` (or `SHUNYA_DATABASE_URL`) to your Postgres URL, apply migrations, then start the app:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
shunya-timescale migrate
shunya-timescale sync-index-memberships
uv run python scripts/bootstrap_example_alphas.py
uv run uvicorn backtest_api.main:app --reload --host 127.0.0.1 --port 8000
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

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` / `SHUNYA_DATABASE_URL` | Postgres for API tables + Timescale OHLCV reads |
| `SHUNYA_RUN_TIMESCALE_CONTAINER` | Set to `1` to run Timescale-backed tests via Docker testcontainers when `DATABASE_URL` is unset, and for an **isolated** DB for `test_alphas_crud_and_backtest_job` (ignores shared `DATABASE_URL` for that fixture). |
| `SHUNYA_API_INTEGRATION_DATABASE_URL` | Optional **dedicated** Postgres URL for queue-based API integration tests (`test_alphas_crud_and_backtest_job`). Use when you cannot run Docker testcontainers but must not share the job queue with another process (e.g. a running `uvicorn` on the same `DATABASE_URL`). |
| `SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS` | Set to `1` only with `DATABASE_URL` / `SHUNYA_DATABASE_URL` when **no** other API worker process uses that database; otherwise job-queue tests can flake or fail. Prefer testcontainers or `SHUNYA_API_INTEGRATION_DATABASE_URL`. |
| `SHUNYA_DASHBOARD_MAX_TICKERS` | Optional cap for `GET /data/dashboard` symbol list (positive integer); omit for no cap |
| `SHUNYA_API_DATABASE_URL` | Optional override (via `pydantic-settings`) |
| `SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS` | Worker poll interval (default `1.0`) |
| `YFINANCE_TLS_VERIFY` | If set to `1` / `true` / `yes` / `on`, yfinance uses default TLS verification instead of the `curl_cffi` session with `verify=False` (useful outside corporate TLS inspection). |

## Tests

- Unit: `pytest tests/test_backtest_api/ -m "not timescale"`
- DB + HTTP integration: `pytest tests/test_backtest_api/ -m timescale` with `DATABASE_URL` set, or `SHUNYA_RUN_TIMESCALE_CONTAINER=1` and Docker (skips if Docker is unavailable).
- **`test_alphas_crud_and_backtest_job`** enqueues `api_backtest_jobs` rows; any other process using the same database URL and running the API worker can **claim the same job** (`FOR UPDATE SKIP LOCKED`). That produces failures whose traceback points at `backtest_api/worker.py` even when pytest patches the in-process worker. Use **`SHUNYA_RUN_TIMESCALE_CONTAINER=1`** (isolated Timescale container for that test, even if `DATABASE_URL` is set) or **`SHUNYA_API_INTEGRATION_DATABASE_URL`** pointing at a database **no live API** is connected to. Only if you are sure nothing else is polling the queue: **`SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS=1`** with `DATABASE_URL`.
