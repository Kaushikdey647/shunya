# Timescale first-run checklist

Use this checklist alongside the full guide [Local Timescale](../data_timescale.md).

## Prerequisites

- Docker (or another Postgres with Timescale)
- Python: `pip install 'shunya-py[timescale]'` or from a clone `uv sync --extra timescale` (and `--extra api` if you run FastAPI)

## Steps

1. **Set `DATABASE_URL`** (or `SHUNYA_DATABASE_URL`), for example:

   `postgresql://postgres:postgres@localhost:5432/shunya`

2. **Start Postgres** — for example `docker compose up -d` from the repo root.

3. **Apply migrations:**

   ```bash
   shunya-timescale migrate
   ```

   Equivalent: `python -m shunya.data.timescale.cli migrate`

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

## Per-invocation override

Pass **`--database-url ...`** to `shunya-timescale` subcommands if you do not want to rely on the environment for a single run.

## Read panels from Timescale in code

See the example in [Local Timescale](../data_timescale.md#reading-in-fints) using `TimescaleMarketDataProvider` and optional `TimescaleFundamentalDataProvider`.
