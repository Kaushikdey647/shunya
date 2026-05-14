# Quickstart

Pick **one** path. Prerequisites are called out in each section.

!!! note "Backtest jobs and the API"

    When you run **`uv run uvicorn api.main:app`**, the FastAPI app starts an **in-process async loop** that claims and runs backtest jobs from Postgres ([`api/main.py` lifespan](https://github.com/Kaushikdey647/shunya/blob/main/api/main.py)). You do **not** need a second terminal running a separate “worker” daemon for normal local development. [Docker Compose](https://github.com/Kaushikdey647/shunya/blob/main/docker-compose.yml) also runs only **`uvicorn`**; the same embedded loop applies.

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

**Smoke checks**

```bash
curl -sSf http://127.0.0.1:8000/healthz    # expect HTTP 200
```

- **API docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **UI:** URL printed by Vite (default [http://localhost:5173](http://localhost:5173))

**First five minutes in the UI**

1. Open **Studio** → create or open an alpha, or use a bootstrapped example if you ran bootstrap scripts.
2. Open **Backtests** → enqueue a job → open the job detail when it succeeds.  
   Walkthrough: [Backtest from the web UI](how-to/backtest-from-ui.md).

---

## Path C — Docker Compose (API + Timescale only)

Use this if you want the API in Docker without running **`local-dev-all.sh`** on the host.

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
docker compose up -d
docker compose exec api uv run shunya-timescale migrate
curl -sSf http://127.0.0.1:8000/healthz
```

The **`api`** service runs **`uvicorn`** only; the **backtest worker loop still runs inside that process** (no separate `worker` container). The compose file is [docker-compose.yml](https://github.com/Kaushikdey647/shunya/blob/main/docker-compose.yml).

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
| HTTP routes | [HTTP API](http-api.md) |
