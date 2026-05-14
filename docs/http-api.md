# HTTP API (FastAPI)

The **`api/`** package is a **repo-local** [FastAPI](https://fastapi.tiangolo.com/) service: alphas, async backtest jobs, instruments, market dashboards, data coverage, optional **trade desk** routes (Alpaca), and optional **Ollama**-backed alpha assist.

It is **not** published on PyPI inside the `shunya-py` wheel; run it from a clone with the `api` (and usually `timescale`) extras.

## Authoritative README

See **[`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)** in the repository for:

- Route list and semantics
- Environment variables (`DATABASE_URL`, `SHUNYA_API_*`, Alpaca keys, CORS, trade-desk token, Ollama, …)
- How the background worker and DB migrations fit together

## Interactive OpenAPI

When the server is running (for example `uv run uvicorn api.main:app --reload`), open:

- **`/docs`** — Swagger UI
- **`/openapi.json`** — OpenAPI schema

There is no bundled static OpenAPI export in this documentation site yet; use a running instance or the linked README.

## Related UI

The **[`ui/`](https://github.com/Kaushikdey647/shunya/tree/main/ui)** directory is the React client (Alpha Studio, backtests, dashboards, trade surfaces). Point it at your API base URL and configure CORS on the API as documented in `api/README.md`.
