# Local development: API, worker, and UI

## One-shot bootstrap

From the repository root:

```bash
./scripts/local-dev-all.sh
```

The script checks for Docker, `uv`, and Node.js 20+, creates or augments a root **`.env`** with a default **`DATABASE_URL`** when none is set, starts the **TimescaleDB** service from **`docker compose`**, runs **`uv sync`** (dev, API, Timescale extras), applies **`shunya-timescale migrate`**, starts **`uvicorn`** on port **8000**, installs UI dependencies with **`npm ci`** when **`ui/node_modules`** is missing, then runs **`npm run dev`** in **`ui/`**. Stopping the UI (Ctrl+C) stops the API process started by the script.

**Backtest jobs** run **in the same process as `uvicorn`**: the API lifespan starts an asyncio task that polls Postgres and executes queued backtests. You do **not** need a separate worker daemon for normal development (Docker Compose runs the same layout: one **`api`** container, no extra `worker` service).

For step-by-step control, use the sections below instead.

Run the **FastAPI** service first, then the **`ui/`** Vite dev server. The UI health checks and proxied `/api` calls expect the API to be reachable.

**Prerequisites:** **Python** with [`uv`](https://docs.astral.sh/uv/) (or your usual tool) for sections 1–3; **Node.js 20+** and **npm** to install and run the UI in sections 4–5. See also [Install](../install.md) (Web UI section).

## 1. Verify Python dependencies

From the repository root:

```bash
uv sync --extra dev --extra api --extra timescale
```

Optional: copy `.env.example` to `.env` at the repo root and set `DATABASE_URL`, `SHUNYA_API_*`, etc. Restart the API after changing `.env`.

## 2. Database migrations

With `DATABASE_URL` (or `SHUNYA_DATABASE_URL`) set:

```bash
shunya-timescale migrate
```

Many features (alpha CRUD, backtest queue, `/data/dashboard`, `PATCH /settings/app` overlay) expect Postgres. Without a database, some **yfinance**-backed routes still work; see [Timescale first-run checklist](timescale-checklist.md).

## 3. Start the API

**Docker Compose** (Timescale + API on `http://127.0.0.1:8000`):

```bash
docker compose up -d
docker compose exec api uv run shunya-timescale migrate
curl -sSf http://127.0.0.1:8000/healthz
```

**Local uvicorn:**

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Smoke test: `curl -sSf http://127.0.0.1:8000/healthz` (expect HTTP 200). Interactive docs: `http://127.0.0.1:8000/docs`.

### CORS

If the UI runs on another origin (for example `http://localhost:5173`) **without** the Vite proxy, set on the API:

`SHUNYA_CORS_ORIGINS=http://localhost:5173` (exact origin, no trailing slash; comma-separate multiple).

### Alpha Studio AI (optional)

Set **`SHUNYA_API_OLLAMA_HOST`** (for example `http://127.0.0.1:11434`) and optionally **`SHUNYA_API_OLLAMA_MODEL`**. Model and timeout can also be tuned via **`PATCH /settings/app`** when the database overlay and trade-desk token are configured (see [api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)).

## 4. Install and start the UI

From the repository root (first time in `ui/`, use **`npm ci`** so the lockfile is respected):

```bash
cd ui
npm ci
npm run dev
```

Open the URL Vite prints (default **http://localhost:5173**). In development, **`ui/vite.config.ts`** proxies **`/api`** to **`http://127.0.0.1:8000`**, so the API should listen on port **8000** unless you change the proxy.

## 5. Production UI build

Set **`VITE_API_BASE`** at **build** time if the UI is not served behind the same host as `/api`. See [Web application: Configuration](../ui/configuration.md).

## See also

- [HTTP API](../http-api.md) — route groups and OpenAPI.
- [Backtest from the web UI](backtest-from-ui.md).
