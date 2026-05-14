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

## Web UI (`ui/`)

The **React + Vite** app in **`ui/`** is not published on PyPI; install it from a **clone** of this repository alongside (or after) the Python environment above.

Requires **Node.js 20+** and **npm** (see [`ui/package.json`](https://github.com/Kaushikdey647/shunya/blob/main/ui/package.json) in the repo).

From the repository root:

```bash
cd ui
npm ci
npm run dev
```

Open the URL Vite prints (default **http://localhost:5173**). Start the **FastAPI** service on **port 8000** first so health checks and proxied API calls succeed; in dev, [`ui/vite.config.ts`](https://github.com/Kaushikdey647/shunya/blob/main/ui/vite.config.ts) forwards **`/api`** to **`http://127.0.0.1:8000`**.

**Production:** `npm run build`, then `npm run preview` or `npm run start`. Set **`VITE_API_BASE`** at **build** time when the UI and API are served from different origins.

More detail: [`ui/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/ui/README.md), [How-to: Local dev](how-to/local-dev-api-ui.md), root [README — Service and UI setup](https://github.com/Kaushikdey647/shunya/blob/main/README.md#service-and-ui-setup).

## Docs site locally

```bash
uv sync --group dev --group docs
uv run mkdocs serve
```

Use `--group docs` only if you only need MkDocs (for example in CI). Then open the URL MkDocs prints (usually `http://127.0.0.1:8000`).

MkDocs **Mermaid** diagrams use a vendored copy at **`docs/javascripts/mermaid.min.js`** (for `mkdocs build --strict` without relying on outbound URL checks). If that file is missing, download the same path from [cdn.jsdelivr](https://cdn.jsdelivr.net/npm/mermaid@10.4.0/dist/mermaid.min.js) before building.

## Related documentation

- [Glossary](glossary.md) — terminology
- [Concepts](concepts/index.md) — quantitative finance and how it maps to Shunya
- [Documentation](documentation/system-overview.md) — library stack, types, and APIs
- [How-to: Local dev](how-to/local-dev-api-ui.md) — API, worker, and `ui/` together
- [HTTP API](http-api.md) — FastAPI route outline
- [Web application](ui/overview.md) — React client areas and configuration
