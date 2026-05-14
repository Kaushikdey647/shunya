# Shunya

![Shunya banner](banner.png)

**Shunya** is a Python stack for **systematic equity research**: multi-ticker OHLCV panels (`finTs`), JAX-friendly **FinStrat** / `cross_section` alphas, **FinBT** (backtrader) backtests, portfolio construction, optional **PortfolioRiskEngine**, **OMS** / **EMS** modules, optional **TimescaleDB** storage, and a repo-local **FastAPI** service consumed by the **[`ui/`](https://github.com/Kaushikdey647/shunya/tree/main/ui)** web app.

## Where to go next

| Topic | Page |
|--------|------|
| Install and extras | [Install](install.md) |
| Timescale ingest and `DATABASE_URL` | [Local Timescale](data_timescale.md) |
| FastAPI routes and env | [HTTP API](http-api.md) |
| Public Python API (docstrings) | [Reference](reference/index.md) |
| Architecture decision (layers) | [ADR 001](adr/001-architecture-layers.md) |

**PyPI:** [`shunya-py`](https://pypi.org/project/shunya-py/) — `pip install shunya-py` (optional extras: `timescale`, `api`, `risk`, …).

**Repository:** [github.com/Kaushikdey647/shunya](https://github.com/Kaushikdey647/shunya) — full narrative, quick start, and roadmap live in the root [README](https://github.com/Kaushikdey647/shunya/blob/main/README.md).
