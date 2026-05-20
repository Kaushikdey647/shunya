# Timescale first-run checklist

Use this checklist alongside the full guide [Local Timescale](../data_timescale.md).

## Prerequisites

- Docker (or another Postgres with Timescale)
- Python: `pip install 'shunya-py[timescale]'` or from a clone `uv sync --extra timescale` (and `--extra api` if you run FastAPI)

## Steps

1. **Set `DATABASE_URL`** (or `SHUNYA_DATABASE_URL`), for example:

   `postgresql://postgres:postgres@localhost:5432/shunya`

2. **Start Postgres** — for example `docker compose up -d` from the repo root (starts **TimescaleDB**, **API**, and **UI**; the API image applies migrations on startup unless **`RUN_MIGRATIONS=0`**).

3. **Apply migrations:**

   ```bash
   shunya-timescale migrate
   ```

   Equivalent: `python -m shunya.data.timescale.cli migrate`

   If you use **Docker Compose** for the **API** service, migrations normally run automatically when the **`api`** container starts (**`RUN_MIGRATIONS=1`** in [docker-compose.yml](https://github.com/Kaushikdey647/shunya/blob/main/docker-compose.yml)). Use the commands above when running **`shunya-timescale`** from the host against **`localhost:5432`**, or after disabling auto-migrate.

4. **Ingest OHLCV** (Yahoo-backed loader, normalized like default `finTs`):

   ```bash
   shunya-timescale ingest-ohlcv --symbols "AAPL MSFT" --start 2020-01-01 --end 2024-01-01
   ```

5. **Optional — fundamentals** (repo on `PYTHONPATH` for `examples.yfinance_fundamental_provider`):

   `shunya-timescale ingest-fundamentals --symbols AAPL --start 2020-01-01 --end 2024-01-01`

6. **Optional — classifications** (sector/industry snapshot):

   `shunya-timescale ingest-classifications --symbols "AAPL MSFT"`

7. **API-only — index memberships** (if you use index backtests via `POST /backtests` with `index_code`):

   ```bash
   shunya-timescale sync-index-memberships
   ```

   (Also referenced in [api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md) bootstrap.)

For **repo `scripts/`** ingest (SP100, full index union, example alphas) and ordering with the API + UI, see [Bootstrap scripts (API + UI + DB)](bootstrap-scripts.md).

## Per-invocation override

Pass **`--database-url ...`** to `shunya-timescale` subcommands if you do not want to rely on the environment for a single run.

## Read panels from Timescale in code

See the example in [Local Timescale](../data_timescale.md#reading-in-fints) using `TimescaleMarketDataProvider` and optional `TimescaleFundamentalDataProvider`.
