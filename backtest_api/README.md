# Backtest API

Repo-local **FastAPI** service (not part of the published `shunya-py` wheel) for:

- **`/alphas`** — CRUD on alpha definitions (`import_ref` allow-list: `examples.alphas.<module>:alpha` only) and stored `finstrat_config` JSON.
- **`/backtests`** — Enqueue async jobs (`POST`), list/get status, fetch JSON results when `succeeded` (same shape as `FinBT.results(show=False)` plus optional `benchmark`).
- **`POST /data`** — Panel NaN counts per ticker/column and per-ticker return / annualized vol / Sharpe / Sortino from `finTs` (Timescale when `DATABASE_URL` is set and `market_data_provider` is `auto` or `timescale`).

## Install

From the repo root:

```bash
uv sync --extra api --extra timescale
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
shunya-timescale migrate
uv run uvicorn backtest_api.main:app --reload --host 127.0.0.1 --port 8000
```

Or use Docker Compose (`api` service mounts the repo and runs the same commands against the `timescaledb` service).

## Environment

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` / `SHUNYA_DATABASE_URL` | Postgres for API tables + Timescale OHLCV reads |
| `SHUNYA_API_DATABASE_URL` | Optional override (via `pydantic-settings`) |
| `SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS` | Worker poll interval (default `1.0`) |

## Tests

- Unit: `pytest tests/test_backtest_api/ -m "not timescale"`
- DB + HTTP integration: `pytest tests/test_backtest_api/test_api_integration.py -m timescale` with `DATABASE_URL` set, or `SHUNYA_RUN_TIMESCALE_CONTAINER=1` and Docker (skips if Docker is unavailable).
