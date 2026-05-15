# HTTP API package (`api/`)

The **`api/`** directory is a **repo-local [FastAPI](https://fastapi.tiangolo.com/)** application. It is **not** shipped inside the **`shunya-py`** PyPI wheel. Install and run it from a clone with the **`api`** (and usually **`timescale`**) extras — see [Install](../install.md) and [`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md) on GitHub.

## Where to read what

| Need | Document |
|------|-----------|
| Route groups, auth headers, env overview | [HTTP API](../http-api.md) (this docs site) |
| Backtest semantics, migrations, full env table, trade desk | [`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md) (repository) |
| **Per-route schemas, query params, try-it-out** | **OpenAPI** from a running instance (below) |

There is **no** checked-in static OpenAPI export in the repository; the schema is always derived from the live app.

## OpenAPI and Swagger UI

With the API running (for example `uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`):

| URL | Purpose |
|-----|---------|
| **`/docs`** | **Swagger UI** — interactive explorer |
| **`/redoc`** | **ReDoc** — alternate read-only layout |
| **`/openapi.json`** | Machine-readable **OpenAPI 3** schema (full paths, components, security) |

Download the schema without a browser:

```bash
curl -sS "http://127.0.0.1:8000/openapi.json" -o openapi.json
```

Use that file with **[openapi-generator](https://openapi-generator.tech/)**, **[openapi-typescript](https://github.com/drwpow/openapi-typescript)**, or any OpenAPI-aware client generator. Regenerate when you upgrade FastAPI or change route signatures.

## Source layout (`api/`)

High-level map (paths relative to repository root):

| Path | Role |
|------|------|
| **`api/main.py`** | `create_app()`, CORS, router includes, **`/health`** / **`/healthz`**, lifespan (worker task, trade desk runtime). |
| **`api/routers/`** | HTTP route modules (`alphas`, `backtests`, `universes`, `instruments`, `data`, `market`, `trade_desk`, `app_settings`, `indices`, …). |
| **`api/services/`** | Business logic used by routers (Timescale queries, market snapshots, universe analytics, …). |
| **`api/repositories/`** | Postgres access patterns (jobs queue, universes CRUD, …). |
| **`api/schemas/`** | Pydantic request/response models shared with routers. |
| **`api/settings.py`** | **`SHUNYA_API_*`** pydantic-settings surface. |
| **`api/worker.py`** | Default async backtest job loop (in-process with uvicorn unless you split processes). |

Exception handling, health aggregation, and Alpaca/trade wiring live alongside these modules; use your IDE or GitHub file search for specifics.

## Related

- [Reference overview](index.md) — Python package (`shunya`) vs HTTP API (`api`)
- [Local development: API, worker, and UI](../how-to/local-dev-api-ui.md)
