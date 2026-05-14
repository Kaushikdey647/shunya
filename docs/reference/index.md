# Reference

## Python package (`shunya`)

The installable PyPI package **`shunya-py`** ships the **`shunya`** namespace only. Auto-generated API pages are built from **docstrings** with [mkdocstrings](https://mkdocstrings.github.io/).

- **[Python package (`shunya`)](library.md)** — generated from `import shunya` (same symbols as [`__init__.py` `__all__`](https://github.com/Kaushikdey647/shunya/blob/main/shunya/__init__.py) at the package root).

Submodules under `shunya.*` are not given separate top-level pages in this first revision; expand types on the library page or jump to source on GitHub from the docstring links where enabled.

For narrative explanations of major types (`finTs`, FinStrat, PCS, OMS/EMS), see **[Concepts (finance)](../concepts/index.md)**. For **APIs, types, and configuration**, see **[Documentation](../documentation/system-overview.md)**.

## HTTP API (`api`)

The **`api/`** tree is **not** part of the PyPI wheel; it is developed and deployed from this repository (see [HTTP API](../http-api.md)). Route-level reference is the running service **OpenAPI** at `/docs` / `/openapi.json` when uvicorn is up.
