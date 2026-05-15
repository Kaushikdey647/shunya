# HTTP API (`api.main`)

Generated from **docstrings** in **`api.main`** with [mkdocstrings](https://mkdocstrings.github.io/). The **`api`** package is included in the **`shunya-py`** wheel; install **`[api]`** (and usually **`[timescale]`**) to run **`uvicorn api.main:app`**. Overview, route table, and environment summary: [HTTP API (overview)](http-api-package.md).

Per-route request/response models live on the routers and in **`api.schemas`**; use **`/openapi.json`** from a running server for the complete OpenAPI document.

::: api.main
    options:
      show_submodules: false
      heading_level: 2
      members_order: source
      filters:
        - "!^_"
