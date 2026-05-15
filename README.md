# shunya

![Shunya](docs/banner.png)

[![PyPI](https://img.shields.io/pypi/v/shunya-py.svg)](https://pypi.org/project/shunya-py/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-526F9A?logo=githubpages&logoColor=white)](https://kaushikdey647.github.io/shunya/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/package_manager-uv-6f42c1.svg)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

**Shunya** is a Python toolkit for **systematic equity research**: multi-ticker panels (`finTs`), JAX-friendly alphas (**FinStrat**, **FinBT**), portfolio construction, optional risk and broker helpers, plus a repo-local **FastAPI** service and **React** app in **`ui/`** (Alpha Studio, backtests, dashboards).

Home dashboard (macro strip, movers, recent backtests) when the UI is pointed at a running API:

![Shunya web UI — home dashboard](docs/dashboard.png)

Long-form guides, concepts, HTTP API outline, Timescale, and the generated Python reference are on the documentation site:

**[kaushikdey647.github.io/shunya](https://kaushikdey647.github.io/shunya/)**

Begin at **[Quickstart](https://kaushikdey647.github.io/shunya/quickstart/)** (library-only vs full stack vs Docker Compose).

## Quickstart

**Install the library from PyPI**

```bash
pip install "shunya-py[dev]"
```

**Clone: API + Timescale + UI** (needs Docker, [uv](https://docs.astral.sh/uv/), Node.js 20+)

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
./scripts/local-dev-all.sh
```

Smoke-check the API: `curl -sSf http://127.0.0.1:8000/healthz`. Open the URL Vite prints (default **http://localhost:5173**). Queued backtests run **in the same process as uvicorn** ([`api/main.py`](api/main.py)); you do not need a second “worker” process for normal local use.

After migrate, optional **database seeding** (example alphas, OHLCV ingest) uses the repo **`scripts/`** helpers; see the docs guide **[Bootstrap scripts (API + UI + DB)](https://kaushikdey647.github.io/shunya/how-to/bootstrap-scripts/)** or [`scripts/README.md`](scripts/README.md). Example: `./scripts/local-dev-all.sh --seed-alphas` inserts bundled **`api_alphas`** rows for **`GET /alphas`**.

**Clone: run tests**

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
uv sync --extra dev
uv run pytest
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for architecture, extension patterns, and coding guidelines.

## License

Licensed under the **MIT License** — see [`LICENSE`](LICENSE).
