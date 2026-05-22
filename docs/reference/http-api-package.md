# HTTP API package (`api/`)

The **`api`** package is a **[FastAPI](https://fastapi.tiangolo.com/)** service: alphas, async backtest jobs, instruments, market dashboards, data coverage, optional **trade desk** routes (Alpaca), optional **Ollama**-backed alpha assist, and runtime settings.

It is **shipped inside the `shunya-py` wheel** alongside **`shunya`**. Runtime dependencies for the server (FastAPI, uvicorn, …) are **not** installed by `pip install shunya-py` alone — use **`pip install "shunya-py[api,timescale]"`** (or **`uv sync --extra api --extra timescale`** from a clone). See [Install](../install.md) and **[`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)** for bootstrap, migrations, and the full environment table.

**Generated code reference** (docstrings in **`api.main`**): [HTTP API — `api.main`](api-library.md).

## Interactive OpenAPI

When the server is running (for example `uv run uvicorn api.main:app --reload`):

- **`/docs`** — Swagger UI (try-it-out)
- **`/redoc`** — ReDoc
- **`/openapi.json`** — machine-readable OpenAPI 3 schema

There is no checked-in static OpenAPI export; generate clients from a running instance or saved `openapi.json`.

```bash
curl -sS "http://127.0.0.1:8000/openapi.json" -o openapi.json
```

Use that file with **[openapi-generator](https://openapi-generator.tech/)**, **[openapi-typescript](https://github.com/drwpow/openapi-typescript)**, or any OpenAPI-aware client generator.

## Architecture

- **HTTP process** — FastAPI app (`api.main:app`).
- **Background worker** — by default **not** a separate OS process: an **asyncio** task started from the app lifespan polls the Postgres job queue for backtests, using the same **`DATABASE_URL`** as the HTTP handlers. A dedicated worker process is optional for deployment isolation.
- **Design diagrams (Mermaid)** — High-level design (HLD), low-level design (LLD), component flows, job sequences, and an ER overview of Timescale/API tables live in **[`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)**. The published docs site enables Mermaid via **`mkdocs.yml`** (`mermaid2` + `pymdownx.superfences`); `api/README.md` is not under `docs/` nav by default—open it in the repository or on GitHub for those sections.

Typical backtest flow:

1. Client **`POST /backtests`** → API inserts a queued job in Postgres.
2. Worker claims the job (`FOR UPDATE SKIP LOCKED` style polling).
3. Worker runs **FinBT** / library code, writes results and status.
4. Client **`GET /backtests/{job_id}`** → API returns status and JSON results when complete.

## Health

- **`GET /healthz`** — fast liveness (**`{"status":"ok"}`**); safe for load balancers and Railway-style probes.
- **`GET /health`** — aggregate readiness: backend settings, Postgres **`SELECT 1`**, a small Yahoo Finance fetch, and **Alpaca** when **`SHUNYA_API_ALPACA_ENABLED`** is set (lightweight **`get_account()`**). The **`alpaca`** field is **`skipped`** when the trade desk is disabled. Yahoo or Alpaca failures set overall status to **`degraded`**; backend or database errors set **`error`**.

## Route groups (outline)

| Prefix | Purpose |
|--------|---------|
| **`/alphas`** | CRUD on alpha definitions: optional inline **`source_code`** (executed in worker) or module **`import_ref`** (allow-listed). Stored **`finstrat_config`**. |
| **`/indices`** | Equity indexes from Timescale for benchmark / membership flows. |
| **`/universes`** | CRUD saved equity universes, membership, flat ticker list, sector/industry + fundamentals summary, **`GET /universes/{id}/return-analytics`**. |
| **`/backtests`** | Enqueue async jobs, list/get status, fetch JSON results when succeeded (FinBT-shaped payload plus optional benchmark). |
| **`POST /data`** | Panel diagnostics from `finTs` (NaNs, vol, Sharpe, Sortino, …). |
| **`GET /data/dashboard`** | Database-wide coverage metrics (per-ticker completeness, bucket flags, histograms), classification counts, and risk metrics for the UI data summary: each ticker includes **`return_pct`**, **`log_return_pct`** (`100 * ln(c_last / c_first)` when valid), annualized vol, Sharpe, and Sortino from stored closes. |
| **`/instruments/...`** | Search, OHLCV (optional **`route`** query: `auto`, `best_effort`, or explicit upstream id; response includes **`provenance`** with **`read_path`** vs **`upstream_source_id`**), fundamentals-style panels; prefers Timescale when coverage is complete. **WebSocket** **`/instruments/{symbol}/stream/alpaca-l1`** — when Alpaca is enabled on the API, **IEX** L1 **quotes** (BBO) and **trades** as JSON (`hello` with `schema: 1` and `channels`, `quote`, `trade`, optional `trade_correction` / `trade_cancel`, `error`). All browser sessions share **one** Alpaca market-data WebSocket per API key in the API process (see **`api.services.alpaca_l1_feed_hub`**); distinct symbols are capped by **`SHUNYA_ALPACA_L1_MAX_SYMBOLS`** (default **30**); over-cap returns **`error`** with **`code: symbol_limit`**. Stocks/ETFs only. The legacy **`/instruments/{symbol}/stream/alpaca-bars`** endpoint returns **`deprecated_stream`** and closes. |
| **`/market/...`** | **`POST /market/snapshot`**, **`GET /market/movers`**, **`GET /market/headlines`**, … |
| **`/trade/...`** | Alpaca-backed account and paper-cycle routes when enabled (see auth below). |
| **`GET /settings/app`** | Effective runtime tunables (env merged with optional DB overlay). |
| **`PATCH /settings/app`** | Merge non-secret tunables into overlay (requires DB + migration; auth when trade-desk token is set). |

Semantics that affect every client (fixed backtest window, **index** vs **saved-universe** jobs, `include_test_period_in_results`): authoritative list in **[`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)**.

## Universe return analytics

- **`GET /universes/{universe_id}/return-analytics`** — Builds an aligned **daily** close panel from **`ohlcv_bars`** for universe members (up to **`max_members`**, default **500**, max **5000**), then returns simple and log **return correlation** matrices, **cross-sectional volatility** of simple returns, **PCA** summaries (explained variance, component scores, loadings for scatter), and **cap-weight concentration** (HHI, CR5, CR10, top holdings) with equal-weight fallback when market caps are missing.

**Query parameters**

| Parameter | Default | Notes |
|-----------|---------|--------|
| **`period`** | `1y` | Same allowlist as instrument OHLCV history: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`. |
| **`interval`** | `1d` | Only **`1d`** is supported. |
| **`source`** | `yfinance` | Provider tag stored on bars (non-empty string, length ≤ 64). |
| **`max_members`** | `500` | Integer **2–5000**; caps how many members enter the panel. |
| **`n_pca_components`** | `5` | Integer **1–15**; SVD truncation for PCA summaries. |

The JSON includes an **`alignment`** string (window, counts, thresholds) so clients can interpret sparse panels. The **Risk & structure** tab on the universe detail page uses this route; see [Studio → Risk & structure tab](../ui/studio.md#risk-structure-tab).

## Authentication and privileged headers

- **`POST /trade/paper/cycle`** and **`PATCH /settings/app`** (when **`SHUNYA_API_TRADE_DESK_TOKEN`** is set) require header **`X-Shunya-Trade-Desk-Token`** matching that secret exactly.
- **`GET /settings/app`** does not require the trade-desk token (read-only, no secrets in the payload design).

Alpaca routes require **`SHUNYA_API_ALPACA_ENABLED`** and broker keys at API startup.

## Environment variables (compact)

The API loads a repo-root **`.env`** (if present) and **pydantic-settings** with prefix **`SHUNYA_API_`**. Full table: **[`api/README.md` — Environment](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md#environment)**.

| Variable | Role |
|----------|------|
| `DATABASE_URL` / `SHUNYA_DATABASE_URL` | Postgres for API tables + Timescale reads |
| `SHUNYA_CORS_ORIGINS` | Comma-separated browser origins when UI is cross-origin |
| `SHUNYA_API_ALPACA_ENABLED` | Enable Alpaca clients and `/trade/...` |
| `SHUNYA_API_TRADE_DESK_TOKEN` | Shared secret for paper cycle and settings patch |
| `SHUNYA_API_OLLAMA_HOST` | Ollama base URL for alpha assist |
| `SHUNYA_API_OLLAMA_MODEL` | Default Ollama model id |
| `SHUNYA_DASHBOARD_MAX_TICKERS` | Optional cap for `/data/dashboard` |
| `SHUNYA_TLS_VERIFY` | **Unset** or truthy: verify TLS for yfinance and Alpaca clients from `shunya.integration.alpaca_settings`. Falsy (`0` / `false` / `no` / `off`): disable verification (dev only). |

## Source layout (`api/`)

| Path | Role |
|------|------|
| **`api/main.py`** | `create_app()`, CORS, router includes, **`/health`** / **`/healthz`**, lifespan (worker task, trade desk runtime). |
| **`api/routers/`** | HTTP route modules (`alphas`, `backtests`, `universes`, `instruments`, `data`, `market`, `trade_desk`, `app_settings`, `indices`, …). |
| **`api/services/`** | Business logic used by routers (Timescale queries, market snapshots, universe analytics, …). |
| **`api/repositories/`** | Postgres access patterns (jobs queue, universes CRUD, …). |
| **`api/schemas/`** | Pydantic request/response models shared with routers. |
| **`api/settings.py`** | **`SHUNYA_API_*`** pydantic-settings surface. |
| **`api/worker.py`** | Default async backtest job loop (in-process with uvicorn unless you split processes). |

## Related UI

The **[`ui/`](https://github.com/Kaushikdey647/shunya/tree/main/ui)** React app consumes these routes. See [Web application overview](../ui/overview.md) and [Configuration](../ui/configuration.md).

## Related

- [Reference overview](index.md)
- [Local development: API, worker, and UI](../how-to/local-dev-api-ui.md)
