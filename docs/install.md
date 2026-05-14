# Install

Requires **Python 3.12+**.

## From PyPI

```bash
pip install shunya-py
```

### Optional extras

| Extra | Purpose |
|-------|---------|
| `timescale` | Postgres/Timescale client (`psycopg`), CLI `shunya-timescale` |
| `api` | FastAPI + uvicorn stack for the repo-local HTTP service |
| `risk` | CVX-backed `PortfolioRiskEngine` helpers |
| `notebook` | Jupyter-friendly additions |
| `kite` | Kite Connect market / execution adapters |

Examples:

```bash
pip install "shunya-py[timescale]"
pip install "shunya-py[api,timescale]"
```

## From a clone (uv)

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
uv sync --extra dev --extra timescale
```

Add `--extra api` when you need the FastAPI app and worker. Use `uv run …` so the project venv is picked up (for example `uv run pytest`, `uv run uvicorn api.main:app`).

## Docs site locally

```bash
uv sync --group dev --group docs
uv run mkdocs serve
```

Use `--group docs` only if you only need MkDocs (for example in CI). Then open the URL MkDocs prints (usually `http://127.0.0.1:8000`).

## Related documentation

- [Glossary](glossary.md) — terminology
- [Concepts](concepts/overview.md) — library architecture and major types
- [How-to: Local dev](how-to/local-dev-api-ui.md) — API, worker, and `ui/` together
- [HTTP API](http-api.md) — FastAPI route outline
- [Web application](ui/overview.md) — React client areas and configuration
