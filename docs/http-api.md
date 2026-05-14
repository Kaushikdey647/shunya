# HTTP API (FastAPI)

The **`api/`** package is a **repo-local** [FastAPI](https://fastapi.tiangolo.com/) service: alphas, async backtest jobs, instruments, market dashboards, data coverage, optional **trade desk** routes (Alpaca), optional **Ollama**-backed alpha assist, and runtime settings.

It is **not** published inside the `shunya-py` wheel; run it from a clone with the **`api`** extra (and usually **`timescale`**) as described in [Install](install.md).

## Interactive OpenAPI

When the server is running (for example `uv run uvicorn api.main:app --reload`):

- **`/docs`** — Swagger UI
- **`/openapi.json`** — machine-readable OpenAPI schema

There is no checked-in static OpenAPI export; use a running instance or generate a client from `openapi.json`.

## Architecture

- **HTTP process** — FastAPI app (`api.main:app`).
- **Background worker** — by default this is **not** a separate OS process: an **asyncio task** started from the app lifespan polls the Postgres job queue for backtests and related work, using the same **`DATABASE_URL`** as the HTTP handlers. Splitting a dedicated worker process is possible for deployment isolation but is **not** required for local or default Compose setups.

Typical backtest flow:

1. Client **`POST /backtests`** → API inserts a queued job in Postgres.
2. Worker claims the job (`FOR UPDATE SKIP LOCKED` style polling).
3. Worker runs **FinBT** / library code, writes results and status.
4. Client **`GET /backtests/{job_id}`** → API returns status and JSON results when complete.

## Health

- **`GET /healthz`** — fast liveness (**`{"status":"ok"}`**); safe for load balancers and Railway-style probes.
- **`GET /health`** — aggregate readiness: backend settings, Postgres **`SELECT 1`**, a small Yahoo Finance fetch, and **Alpaca** when **`SHUNYA_API_ALPACA_ENABLED`** is set (lightweight **`get_account()`** against the broker API). The **`alpaca`** field is **`skipped`** when the trade desk is disabled at the API layer. Yahoo or Alpaca failures set overall status to **`degraded`**; backend or database errors set **`error`**.

## Route groups (outline)

| Prefix | Purpose |
|--------|---------|
| **`/alphas`** | CRUD on alpha definitions: optional inline **`source_code`** (executed in worker) or module **`import_ref`** (allow-listed). Stored **`finstrat_config`**. |
| **`/indices`** | Equity indexes from Timescale for benchmark / membership flows. |
| **`/backtests`** | Enqueue async jobs, list/get status, fetch JSON results when succeeded (FinBT-shaped payload plus optional benchmark). |
| **`POST /data`** | Panel diagnostics from `finTs` (NaNs, vol, Sharpe, Sortino, …). |
| **`GET /data/dashboard`** | Database-wide coverage heatmaps and risk/return aggregates. |
| **`/instruments/...`** | Search, OHLCV, fundamentals-style panels; prefers Timescale when coverage is complete. |
| **`/market/...`** | **`POST /market/snapshot`**, **`GET /market/movers`**, **`GET /market/headlines`**, … |
| **`/trade/...`** | Alpaca-backed account and paper-cycle routes when enabled (see auth below). |
| **`GET /settings/app`** | Effective runtime tunables (env merged with optional DB overlay). |
| **`PATCH /settings/app`** | Merge non-secret tunables into overlay (requires DB + migration; auth when trade-desk token is set). |

Semantics that affect every client (fixed backtest window, index jobs, `include_test_period_in_results`): authoritative bullet list in **[api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)**.

## Authentication and privileged headers

- **`POST /trade/paper/cycle`** and **`PATCH /settings/app`** (when **`SHUNYA_API_TRADE_DESK_TOKEN`** is set) require header **`X-Shunya-Trade-Desk-Token`** matching that secret exactly.
- **`GET /settings/app`** does not require the trade-desk token (read-only, no secrets in the payload design).

Alpaca routes require **`SHUNYA_API_ALPACA_ENABLED`** and broker keys at API startup.

## Environment variables (compact)

The API loads a repo-root **`.env`** (if present) and **pydantic-settings** with prefix **`SHUNYA_API_`**. Full table and test notes: **[api/README.md — Environment](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md#environment)**.

| Variable | Role |
|----------|------|
| `DATABASE_URL` / `SHUNYA_DATABASE_URL` | Postgres for API tables + Timescale reads |
| `SHUNYA_CORS_ORIGINS` | Comma-separated browser origins when UI is cross-origin |
| `SHUNYA_API_ALPACA_ENABLED` | Enable Alpaca clients and `/trade/...` |
| `SHUNYA_API_TRADE_DESK_TOKEN` | Shared secret for paper cycle and settings patch |
| `SHUNYA_API_OLLAMA_HOST` | Ollama base URL for alpha assist |
| `SHUNYA_API_OLLAMA_MODEL` | Default Ollama model id |
| `SHUNYA_DASHBOARD_MAX_TICKERS` | Optional cap for `/data/dashboard` |
| `YFINANCE_TLS_VERIFY` | Stricter TLS for yfinance when set truthy |

## Related UI

The **[`ui/`](https://github.com/Kaushikdey647/shunya/tree/main/ui)** React app consumes these routes. See [Web application overview](ui/overview.md) and [Configuration](ui/configuration.md).

## Authoritative README

For install commands, bootstrap scripts, migration filenames, and integration-test caveats, keep using **[`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)** as the source of truth.
