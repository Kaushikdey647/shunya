# Reference

## Python package (`shunya`)

The installable PyPI package **`shunya-py`** ships the **`shunya`** and **`api`** namespaces. Auto-generated API pages are built from **docstrings** with [mkdocstrings](https://mkdocstrings.github.io/).

- **[Python package (`shunya`)](library.md)** — generated from `import shunya` (same symbols as [`__init__.py` `__all__`](https://github.com/Kaushikdey647/shunya/blob/main/shunya/__init__.py) at the package root).

Submodules under `shunya.*` are not given separate top-level pages in this first revision; expand types on the library page or jump to source on GitHub from the docstring links where enabled.

For narrative explanations of major types (`finTs`, FinStrat, PCS, OMS/EMS), see **[Concepts (finance)](../concepts/index.md)**. For **APIs, types, and configuration**, see **[Documentation](../documentation/system-overview.md)**.

## HTTP API (`api`)

The **`api`** package is **included in the `shunya-py` wheel** alongside **`shunya`**. Install optional extras **`api`** and usually **`timescale`** so FastAPI, uvicorn, and database clients are available at runtime.

- **[HTTP API (overview)](http-api-package.md)** — architecture, health routes, route-group table, universe return analytics, authentication, compact environment table, OpenAPI URLs, **`curl`** export, and **`api/`** source layout.
- **[HTTP API (`api.main`)](api-library.md)** — generated docstrings for **`create_app`**, **`app`**, **`run`**, and related symbols in **`api.main`**.

For per-endpoint request/response models and query parameters, use **Swagger** (`/docs`) or **`openapi.json`** from a running server. Deep semantics (backtest window, migrations, full env table): **[`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)**.
