# System overview

**Shunya** is a Python-first stack for **systematic equity research**: multi-ticker OHLCV panels, JAX-friendly alpha code (**FinStrat** / `cross_section`), **backtrader** simulation (**FinBT**), portfolio construction, optional pre-trade risk, institutional-style **OMS** / **EMS** building blocks, optional **TimescaleDB** storage, and a repo-local **FastAPI** service with a **React** web app in `ui/`.

## Components

| Layer | Role |
|-------|------|
| **`shunya` (PyPI `shunya-py`)** | Library: `finTs`, `FinStrat`, `FinBT`, portfolio services, OMS/EMS, providers, operators. |
| **`api/`** | FastAPI app (Python package **`api`** is in the **`shunya-py`** wheel; install **`[api]`** extra to run it): alphas, async backtests, data/instruments/market routes, optional trade desk and Ollama assist. |
| **`ui/`** | Vite + React client: Alpha Studio, backtests, dashboards, trade surfaces (some live, some mock until more HTTP routes exist). |
| **Postgres / Timescale** | Optional durable OHLCV, API job queue, dashboard aggregates, runtime config overlay. |

## How the pieces connect

```text
Browser (React ui) --HTTP JSON--> FastAPI (api)
FastAPI + worker ----import----> shunya (library)
FastAPI + worker ----SQL-------> Postgres / Timescale
shunya ------------providers--> yfinance / Alpaca / Tiingo / Timescale
```

- The **UI** calls the **API** over JSON (in dev, Vite proxies `/api` to uvicorn).
- The **API** and **worker** orchestrate backtests and persistence; they import **`shunya`** for simulation and data access.
- **`finTs`** and providers read from **yfinance**, **Alpaca**, **Tiingo**, or **Timescale** depending on configuration.

## Layering and ports

For how adapters, ports, and API error mapping stay decoupled from engines, see [ADR 001: layering](../adr/001-architecture-layers.md).

## Where to read next

- [Data layer: finTs and providers](fints-providers.md) — data contracts and provenance.
- [Signals: FinStrat and FinBT](finstrat-finbt.md) — signal to simulation.
- [HTTP API (overview)](../reference/http-api-package.md) — route groups and OpenAPI.
- [Glossary](../glossary.md) — short definitions of terms used across the docs.
- [Concepts (finance)](../concepts/index.md) — quantitative background and how it maps to this stack.
