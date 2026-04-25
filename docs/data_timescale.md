# Local TimescaleDB market store

Optional durable layer for OHLCV bars, fundamentals (EAV), and symbol classifications. Research code keeps using `MarketDataProvider` / `FundamentalDataProvider`; the database is the canonical copy after ingest, while Yahoo and other APIs remain **loaders only**.

## Prerequisites

- Docker (for the database)
- Python extra: `pip install 'shunya-py[timescale]'` (installs `psycopg` with binary wheels)

## Connection

Set **`DATABASE_URL`** (or **`SHUNYA_DATABASE_URL`**) to a Postgres URL, for example:

`postgresql://postgres:postgres@localhost:5432/shunya`

Do not commit secrets; use compose defaults locally or your own credentials.

## Bootstrap

1. Start the database:

   `docker compose up -d`

2. Apply SQL migrations (packaged under `shunya/data/timescale/migrations/`):

   `shunya-timescale migrate`

   or:

   `python -m shunya.data.timescale.cli migrate`

3. Ingest OHLCV (Yahoo path, normalized the same way as `finTs` defaults):

   `shunya-timescale ingest-ohlcv --symbols "AAPL MSFT" --start 2020-01-01 --end 2024-01-01`

4. Optional: fundamentals (requires repo root on `PYTHONPATH` for `examples.yfinance_fundamental_provider`):

   `shunya-timescale ingest-fundamentals --symbols AAPL --start 2020-01-01 --end 2024-01-01`

5. Optional: sector / industry snapshot:

   `shunya-timescale ingest-classifications --symbols "AAPL MSFT"`

Override the DSN per invocation with `--database-url ...` (the CLI also sets `DATABASE_URL` for the process).

## Reading in `finTs`

Use the DB-backed providers with the same contracts as Yahoo:

```python
import os
from shunya.data import finTs
from shunya.data.timescale import TimescaleMarketDataProvider, TimescaleFundamentalDataProvider

os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/shunya"

fts = finTs(
    ["AAPL", "MSFT"],
    "2020-01-01",
    "2024-01-01",
    market_data=TimescaleMarketDataProvider(),
    fundamental_data=TimescaleFundamentalDataProvider(),  # optional
)
```

Technicals (`SMA_*`, `RSI_*`, …) are still computed in memory from stored OHLCV, same as the live Yahoo path.

## Tests

- Default `pytest` runs only unit tests (no DB).
- Integration tests are marked **`timescale`**:
  - With a running DB: set `DATABASE_URL` and run `pytest -m timescale`.
  - With Docker and no local DB: `SHUNYA_RUN_TIMESCALE_CONTAINER=1 pytest -m timescale` (pulls the Timescale image on first run).

## Migrations directory override

Advanced: set **`SHUNYA_MIGRATIONS_DIR`** to a directory of `*.sql` files if you fork the schema.
