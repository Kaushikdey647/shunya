# Shunya

![Shunya banner](banner.png)

**Shunya** is a Python stack for **systematic equity research**: multi-ticker OHLCV panels (`finTs`), JAX-friendly **FinStrat** / `cross_section` alphas, **FinBT** (backtrader) backtests, portfolio construction, optional **PortfolioRiskEngine**, **OMS** / **EMS** modules, optional **TimescaleDB** storage, a repo-local **FastAPI** service, and a **React** web app in **`ui/`**.

**PyPI:** [`shunya-py`](https://pypi.org/project/shunya-py/) — `pip install shunya-py` (optional extras: `timescale`, `api`, `risk`, …).

**Repository:** [github.com/Kaushikdey647/shunya](https://github.com/Kaushikdey647/shunya) — full narrative, quick start, and roadmap in the root [README](https://github.com/Kaushikdey647/shunya/blob/main/README.md).

## Documentation map

| I want to… | Page |
|------------|------|
| Install the library or docs toolchain | [Install](install.md) |
| Learn terms quickly | [Glossary](glossary.md) |
| Understand architecture and data flow | [Concepts → System overview](concepts/overview.md), [ADR 001](adr/001-architecture-layers.md) |
| Learn `finTs`, providers, calendars | [Concepts → finTs and providers](concepts/fints-providers.md) |
| Learn FinStrat / FinBT | [Concepts → FinStrat and FinBT](concepts/finstrat-finbt.md) |
| Learn PCS and risk | [Concepts → Portfolio and risk](concepts/portfolio-risk.md) |
| Learn OMS / EMS / paper desk | [Concepts → OMS and EMS](concepts/oms-ems.md) |
| Learn operator modules | [Concepts → Operators](concepts/operators.md) |
| Run API + worker + UI locally | [How-to → Local dev](how-to/local-dev-api-ui.md) |
| Bootstrap Timescale | [How-to → Timescale checklist](how-to/timescale-checklist.md), [Local Timescale](data_timescale.md) |
| Run a backtest from the browser | [How-to → Backtest from UI](how-to/backtest-from-ui.md) |
| Call the paper cycle HTTP route | [How-to → Paper cycle](how-to/paper-cycle-api.md) |
| Configure Alpha Studio AI / DSL | [How-to → Alpha Studio](how-to/alpha-studio-ai-dsl.md) |
| Use the web app by area | [Web application](ui/overview.md) |
| Browse FastAPI route groups | [HTTP API](http-api.md) |
| Read auto-generated Python API | [Reference → Library](reference/library.md) |
| Alpha Vantage integration | [Alpha Vantage](alpha-vantage.md) |

## Contributing

Architecture and coding guidelines: [`CONTRIBUTING.md`](https://github.com/Kaushikdey647/shunya/blob/main/CONTRIBUTING.md) in the repository.
