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

   When you run the API via **Docker Compose** in this repo, the **`api`** container runs **`migrate`** on startup by default (**`RUN_MIGRATIONS=1`**); you can still run the commands above from the host against **`localhost:5432`** when the **`timescaledb`** service is up.

3. Ingest OHLCV (Yahoo path, normalized the same way as `finTs` defaults):

   `shunya-timescale ingest-ohlcv --symbols "AAPL MSFT" --start 2020-01-01 --end 2024-01-01`

4. Optional: fundamentals (requires repo root on `PYTHONPATH` for `examples.yfinance_fundamental_provider`):

   `shunya-timescale ingest-fundamentals --symbols AAPL --start 2020-01-01 --end 2024-01-01`

5. Optional: sector / industry snapshot:

   `shunya-timescale ingest-classifications --symbols "AAPL MSFT"`

Override the DSN per invocation with `--database-url ...` (the CLI also sets `DATABASE_URL` for the process).

### Docker Compose bootstrap

On **`docker compose up`**, a one-shot **`bootstrap`** service (same image as **`api`**) runs **before** **`api`**:

1. **`shunya-timescale migrate`**
2. If **`SHUNYA_COMPOSE_AUTO_BOOTSTRAP=0`**, it stops after migrate (no Yahoo traffic).
3. Otherwise it runs **`docker/compose_bootstrap_probe.py`**. If the probe exits **0**, **full OHLCV ingest scripts are skipped**, but **`scripts/gapfill_sp100_universe_metadata.py`** still runs so **`symbol_classifications`** and **`fundamentals_daily`** stay filled for the **SP100** universe overview.
4. If the probe exits **1**, it runs **`scripts/bootstrap_sp100_timescale.py --skip-migrate`**, then **`scripts/bootstrap_example_alphas.py`**, then **`scripts/bootstrap_ts_data.py`**, then **`scripts/gapfill_sp100_universe_metadata.py`** again as a safety net (idempotent upserts if a prior run stopped after OHLCV).

**Population heuristic:** at least one daily (**`interval = '1d'`**) OHLCV row for benchmark **`^OEX`** with **`source = 'yfinance'`**, and at least **`SHUNYA_COMPOSE_BOOTSTRAP_MIN_SP100_BARS`** distinct **SP100** members with at least one such bar (default **50**). This matches **`default_bar_spec()`** → **`bar_spec_to_interval_key`** and **`bootstrap_sp100_timescale`’s** Yahoo source label.

**Why `api` still migrates:** skipping **`migrate`** when data exists is unsafe if a later release adds SQL migrations. **`RUN_MIGRATIONS=1`** on **`api`** keeps schema upgrades automatic; SQL files are written to be re-runnable.

**`bootstrap_ts_data.py`** ingests the **union of PyTickerSymbols catalog indices**, not SP100-only; first boot can be very slow.

## Reading in `finTs`

Use the DB-backed providers with the same contracts as Yahoo:

```python
import os
from shunya.data import finTs
from shunya.data.timescale import TimescaleMarketDataProvider, TimescaleFundamentalDataProvider

os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/shunya"

fts = finTs(
    "2020-01-01",
    "2024-01-01",
    ["AAPL", "MSFT"],
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
- FastAPI HTTP API integration (`tests/test_api/test_api_integration.py`) uses the same marker and skips if Docker is unavailable when using the testcontainer path.

## Migrations directory override

Advanced: set **`SHUNYA_MIGRATIONS_DIR`** to a directory of `*.sql` files if you fork the schema.
