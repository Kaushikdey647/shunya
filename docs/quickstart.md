# Quickstart

Pick **one** path. Prerequisites are called out in each section.

!!! note "Backtest jobs and the API"

    When you run **`uv run uvicorn api.main:app`**, the FastAPI app starts an **in-process async loop** that claims and runs backtest jobs from Postgres ([`api/main.py` lifespan](https://github.com/Kaushikdey647/shunya/blob/main/api/main.py)). You do **not** need a second terminal running a separate “worker” daemon for normal local development. [Docker Compose](https://github.com/Kaushikdey647/shunya/blob/main/docker-compose.yml) runs **`uvicorn`** in the **`api`** image the same way; the same embedded loop applies.

---

## Path A — Python library only

Use this if you want **`shunya`** in notebooks or scripts without the HTTP API or UI.

```bash
pip install "shunya-py[dev]"
# or from a clone:
uv sync --extra dev
uv run pytest
```

Minimal simulation; deeper narrative in [Concepts](concepts/index.md) and [Signals: FinStrat and FinBT](documentation/finstrat-finbt.md):

```python
import jax.numpy as jnp
from shunya import FinBT, FinStrat, finTs

fts = finTs("2023-01-01", "2024-01-01", ["AAPL", "MSFT", "NVDA"])

def alpha(ctx) -> jnp.ndarray:
    signal = ctx.close / ctx.ts.mean(ctx.close, 50)
    return ctx.cs.rank(signal)

fs = FinStrat(fts, alpha, neutralization="sector", truncation=0.02)
bt = FinBT(fs, fts, cash=100_000.0, commission=0.0005, slippage_pct=0.0005).run()
print(bt.results(show=False)["metrics"])
```

---

## Path B — Full stack from a clone (API + UI)

Fastest path on macOS/Linux with **Docker**, **uv**, and **Node.js 20+**:

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
./scripts/local-dev-all.sh
```

The script creates **`.env`** from [`.env.example`](https://github.com/Kaushikdey647/shunya/blob/main/.env.example) if missing, starts **TimescaleDB**, migrates, runs **uvicorn** on port **8000**, then **Vite** in **`ui/`**. See [Local development: API, worker, and UI](how-to/local-dev-api-ui.md) for manual steps and Ctrl+C behavior.

**Optional database seeding** (OHLCV for index backtests, example alpha rows): use the repo **`scripts/`** helpers and the ordering in [Bootstrap scripts (API + UI + DB)](how-to/bootstrap-scripts.md). Ingest is not run by **`local-dev-all.sh`** by default (network-heavy); **`--seed-alphas`** only inserts **`api_alphas`** examples after migrate.

**Smoke checks**

```bash
curl -sSf http://127.0.0.1:8000/healthz    # expect HTTP 200
```

With **Docker Compose** (Path C), you can also check **`curl -sSf http://127.0.0.1:8080/api/healthz`** (through the UI nginx proxy).

- **API docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **UI:** with **`local-dev-all.sh`**, the URL Vite prints (default [http://localhost:5173](http://localhost:5173)); with **Docker Compose**, open [http://localhost:8080](http://localhost:8080)

**First five minutes in the UI**

1. Open **Studio** → create or open an alpha, or use a bootstrapped example if you ran bootstrap scripts.
2. Open **Backtests** → enqueue a job → open the job detail when it succeeds.  
   Walkthrough: [Backtest from the web UI](how-to/backtest-from-ui.md).

---

## Path C — Docker Compose (Timescale + API + UI)

Use this if you want the **full stack in containers** without installing **uv** or **Node.js** on the host (Docker only).

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
cp .env.example .env   # required for api env_file (Alpaca, Ollama, etc. optional inside the file)
docker compose up --build
```

Repo-root **`.env`** is loaded into the **`api`** container via **`env_file`** in [docker-compose.yml](https://github.com/Kaushikdey647/shunya/blob/main/docker-compose.yml) (Compose 2.24+; **`required: true`**). **`DATABASE_URL`** is still set in Compose to **`timescaledb:5432`**, so it overrides a host-style **`localhost`** URL from `.env` — use that override for Docker networking.

**Bootstrap service:** Compose starts a one-shot **`bootstrap`** container before **`api`**. It runs **`shunya-timescale migrate`**, then — if the DB already has enough SP100 OHLCV (see [Local Timescale](data_timescale.md#docker-compose-bootstrap)) — runs **`scripts/gapfill_sp100_universe_metadata.py`** so universe **SP100** has sector/industry and **`fundamentals_daily`** rows for the overview. Otherwise it runs **`scripts/bootstrap_sp100_timescale.py --skip-migrate`**, **`scripts/bootstrap_example_alphas.py`**, **`scripts/bootstrap_ts_data.py`**, then **`gapfill_sp100_universe_metadata.py`** again as a safety net. The first full OHLCV path can take a long time. Set **`SHUNYA_COMPOSE_AUTO_BOOTSTRAP=0`** to run **migrations only** in that container (no ingest). The **`api`** container still runs **`migrate`** on each start when **`RUN_MIGRATIONS=1`** so new SQL migrations apply on upgrades.

- **UI (nginx + static build):** [http://localhost:8080](http://localhost:8080) — the browser calls **`/api/...`** on the same origin; nginx proxies to FastAPI.
- **API (direct):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and **`curl -sSf http://127.0.0.1:8000/healthz`**, or through the UI proxy: **`curl -sSf http://127.0.0.1:8080/api/healthz`**.

The **`api`** image runs **`shunya-timescale migrate`** before **`uvicorn`** when **`RUN_MIGRATIONS=1`** (set by default in [docker-compose.yml](https://github.com/Kaushikdey647/shunya/blob/main/docker-compose.yml)); this is intentional so **schema** upgrades apply even when the **`bootstrap`** container skipped **data** ingest. Set **`RUN_MIGRATIONS=0`** on the **`api`** service if you prefer to migrate manually, then run:

```bash
docker compose run --rm api uv run shunya-timescale migrate
```

The **`api`** service runs **`uvicorn`** only; the **backtest worker loop still runs inside that process** (no separate `worker` container). Images are built from the repo-root **`Dockerfile`** (API) and **`ui/Dockerfile`** (UI).

---

## Manual `.env` (without the bootstrap script)

If you start the API yourself:

```bash
cp .env.example .env   # repo root; edit DATABASE_URL and any SHUNYA_API_* keys
```

Then follow [Install](install.md) and [Local development: API, worker, and UI](how-to/local-dev-api-ui.md).

---

## Where to go next

| Goal | Link |
|------|------|
| Install options and extras | [Install](install.md) |
| Timescale ingest | [Timescale checklist](how-to/timescale-checklist.md), [Local Timescale](data_timescale.md) |
| Finance concepts | [Concepts](concepts/index.md) |
| Code-oriented APIs | [Documentation](documentation/system-overview.md) |
| HTTP routes | [HTTP API (overview)](reference/http-api-package.md) |
