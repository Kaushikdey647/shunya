# Reference

## Python package (`shunya`)

The installable PyPI package **`shunya-py`** ships the **`shunya`** namespace only. Auto-generated API pages are built from **docstrings** with [mkdocstrings](https://mkdocstrings.github.io/).

- **[Python package (`shunya`)](library.md)** — generated from `import shunya` (same symbols as [`__init__.py` `__all__`](https://github.com/Kaushikdey647/shunya/blob/main/shunya/__init__.py) at the package root).

Submodules under `shunya.*` are not given separate top-level pages in this first revision; expand types on the library page or jump to source on GitHub from the docstring links where enabled.

For narrative explanations of major types (`finTs`, FinStrat, PCS, OMS/EMS), see **[Concepts (finance)](../concepts/index.md)**. For **APIs, types, and configuration**, see **[Documentation](../documentation/system-overview.md)**.

## HTTP API (`api`)

The **`api/`** tree is **not** part of the PyPI wheel; it is developed and deployed from this repository.

- **[HTTP API package (`api/`)](http-api-package.md)** — how **`api/`** maps to FastAPI, where narrative docs vs **`api/README.md`** vs **OpenAPI** live, **`/docs`**, **`/redoc`**, **`/openapi.json`**, and a **`curl`** example for exporting the schema.
- **[HTTP API](../http-api.md)** — route-group outline, authentication, and environment summary on this site.

For per-endpoint request/response models and query parameters, use **Swagger** (`/docs`) or **`openapi.json`** from a running server; those stay in sync with `api/routers` and **`api/schemas`** automatically.
