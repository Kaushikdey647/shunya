# Reference

Generated pages use [mkdocstrings](https://mkdocstrings.github.io/) and Google-style docstrings. OpenAPI from a running server remains the catalog of every path, query, and body field.

## Python

| Page | What it covers |
|------|----------------|
| [Package root](library.md) | `shunya.__all__` |
| [Algorithms and backtesting](algorithm.md) | `FinStrat`, `FinBT`, portfolio construction, risk, operators |
| [Data access](data.md) | `finTs`, providers, calendars, fundamentals |
| [Market data contracts](market-data.md) | Route types and `resolve_market_route` |
| [Execution](execution.md) | OMS, EMS, paper desk |
| [Shared contracts](contracts.md) | Schemas, `ErrorCode`, market clock |
| [Ports and adapters](ports-adapters.md) | Backtest and cross-section ports, Backtrader / NumPy / JAX adapters |

Narrative for those types is under [Documentation](../documentation/system-overview.md) and [Concepts](../concepts/index.md).

## HTTP

| Page | What it covers |
|------|----------------|
| [Overview](http-api-package.md) | Route groups, health, auth, environment |
| [`api.main`](api-library.md) | App factory and lifespan |
| [Settings and errors](http-settings-errors.md) | `SHUNYA_API_*` settings object and error re-exports |
| [Models](http-models.md) | Selected request and response models |
| [Beyond OpenAPI](http-beyond-openapi.md) | Job readback, notification codes, WebSocket frames |

Per-endpoint schemas: `/docs` or `/openapi.json` on a running API. Bootstrap and the full environment table: [`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md).
