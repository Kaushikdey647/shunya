# Shunya

![Shunya banner](banner.png)

**Shunya** is a Python stack for **systematic equity research**: multi-ticker OHLCV panels (`finTs`), JAX-friendly **FinStrat** / `cross_section` alphas, **FinBT** (backtrader) backtests, portfolio construction, optional **PortfolioRiskEngine**, **OMS** / **EMS** modules, optional **TimescaleDB** storage, a repo-local **FastAPI** service, and a **React** web app in **`ui/`**.

**PyPI:** [`shunya-py`](https://pypi.org/project/shunya-py/) — `pip install shunya-py` (optional extras: `timescale`, `api`, `risk`, …).

**Repository:** [github.com/Kaushikdey647/shunya](https://github.com/Kaushikdey647/shunya) — short intro and quick commands in the root [README](https://github.com/Kaushikdey647/shunya/blob/main/README.md); this site holds the full guides.

## Documentation map

| I want to… | Page |
|------------|------|
| **Start in a few minutes** (library vs API vs Docker) | [Quickstart](quickstart.md) |
| Install the library, docs toolchain, or web UI | [Install](install.md) |
| Learn terms quickly | [Glossary](glossary.md) |
| **Quant finance** view (alphas, portfolios, OMS/EMS, pipeline, fundamentals, design tips) | [Concepts](concepts/index.md) |
| **Code and APIs** (`finTs`, `FinStrat`, PCS, operators, stack overview) | [Documentation](documentation/system-overview.md), [ADR 001](adr/001-architecture-layers.md) |
| Run API + UI locally (embedded backtest loop in uvicorn) | [How-to → Local dev](how-to/local-dev-api-ui.md) |
| Bootstrap Timescale | [How-to → Timescale checklist](how-to/timescale-checklist.md), [Local Timescale](data_timescale.md) |
| Run a backtest from the browser | [How-to → Backtest from UI](how-to/backtest-from-ui.md) |
| Call the paper cycle HTTP route | [How-to → Paper cycle](how-to/paper-cycle-api.md) |
| Configure Alpha Studio AI / DSL | [How-to → Alpha Studio](how-to/alpha-studio-ai-dsl.md) |
| Use the web app by area (with UI screenshots) | [Web application](ui/overview.md) |
| Browse FastAPI route groups | [HTTP API](http-api.md) |
| Read auto-generated Python API | [Reference → Library](reference/library.md) |
| Alpha Vantage integration | [Alpha Vantage](alpha-vantage.md) |

## Contributing

Architecture and coding guidelines: [`CONTRIBUTING.md`](https://github.com/Kaushikdey647/shunya/blob/main/CONTRIBUTING.md) in the repository.
