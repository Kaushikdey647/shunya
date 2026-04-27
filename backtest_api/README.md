# Backtest API

Repo-local **FastAPI** service (not part of the published `shunya-py` wheel) for:

- **`/alphas`** — CRUD on alpha definitions (`import_ref` allow-list: `examples.alphas.<module>:alpha` only) and stored `finstrat_config` JSON.
- **`/backtests`** — Enqueue async jobs (`POST`), list/get status, fetch JSON results when `succeeded` (same shape as `FinBT.results(show=False)` plus optional `benchmark`).
- **`POST /data`** — Panel NaN counts per ticker/column and per-ticker return / annualized vol / Sharpe / Sortino from `finTs` (Timescale when `DATABASE_URL` is set and `market_data_provider` is `auto` or `timescale`).
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
| `SHUNYA_API_DATABASE_URL` | Optional override (via `pydantic-settings`) |
| `SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS` | Worker poll interval (default `1.0`) |
| `YFINANCE_TLS_VERIFY` | If set to `1` / `true` / `yes` / `on`, yfinance uses default TLS verification instead of the `curl_cffi` session with `verify=False` (useful outside corporate TLS inspection). |

## Tests

- Unit: `pytest tests/test_backtest_api/ -m "not timescale"`
- DB + HTTP integration: `pytest tests/test_backtest_api/test_api_integration.py -m timescale` with `DATABASE_URL` set, or `SHUNYA_RUN_TIMESCALE_CONTAINER=1` and Docker (skips if Docker is unavailable).
