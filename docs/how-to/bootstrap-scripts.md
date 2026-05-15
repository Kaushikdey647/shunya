# Bootstrap scripts (API + UI + database)

Use these **after** Postgres/Timescale is reachable and **`shunya-timescale migrate`** has been applied (see [Local development: API, worker, and UI](local-dev-api-ui.md) or `./scripts/local-dev-all.sh`). The scripts live under **`scripts/`** in the repository; run them from the **repo root** with **`uv run python …`** so imports resolve.

## Order of operations

1. **Database up** — e.g. Docker Compose **`timescaledb`**, or any Postgres URL in **`DATABASE_URL`** / **`SHUNYA_DATABASE_URL`**.
2. **Migrations** — `uv run shunya-timescale migrate` (creates API tables, OHLCV schema, etc.).
3. **Optional market data** — pick **one** path below (or neither if you only need CRUD and Yahoo-backed routes).
4. **Optional example alphas** — `scripts/bootstrap_example_alphas.py` so **`GET /alphas`** lists the bundled examples for Studio and backtests.

`./scripts/local-dev-all.sh` covers steps 1–2 and starts the API + Vite UI. Pass **`--seed-alphas`** to also run step 4. Step 3 is always manual (ingest can take a long time and hits external data providers).

## `local-dev-all.sh`

From the repo root:

```bash
./scripts/local-dev-all.sh              # DB + migrate + API + UI
./scripts/local-dev-all.sh --seed-alphas   # same, then example alphas
./scripts/local-dev-all.sh --help
```

`--seed-alphas` runs **`scripts/bootstrap_example_alphas.py`** with the same **`DATABASE_URL`** the script exports from **`.env`**.

## `bootstrap_example_alphas.py`

Inserts rows into **`api_alphas`** from **`examples.alphas.ALL_ALPHAS`** (idempotent: skips names that already exist). The UI’s alpha list calls **`GET /alphas`** on this database.

```bash
uv sync --extra api --extra timescale   # if not already (local-dev-all does this)
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/shunya
uv run python scripts/bootstrap_example_alphas.py
uv run python scripts/bootstrap_example_alphas.py --dry-run
uv run python scripts/bootstrap_example_alphas.py --only sma_ratio_20,mean_reversion_20
uv run python scripts/bootstrap_example_alphas.py --database-url "$DATABASE_URL"
```

## `bootstrap_sp100_timescale.py`

Focused ingest for **S&P 100** index workflows: migrations (unless **`--skip-migrate`**), PyTickerSymbols index sync, daily OHLCV for constituents + benchmark **`^OEX`**, quarterly fundamentals and yfinance classifications, then coverage checks. Good default when you mainly care about **SP100** backtests and dashboards without downloading the full PyTickerSymbols union.

```bash
uv sync --extra timescale
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/shunya
uv run python scripts/bootstrap_sp100_timescale.py --dry-run
uv run python scripts/bootstrap_sp100_timescale.py
```

See **`python scripts/bootstrap_sp100_timescale.py --help`** for **`--start`**, **`--end`**, **`--strict`**, and skips.

## `bootstrap_ts_data.py`

Larger ingest: daily OHLCV for the **union of all indices** resolved via PyTickerSymbols, with incremental passes, fundamentals, classifications, and **`symbol_index_membership`** sync. Default date window is aligned with the HTTP backtest policy documented in the API (**`[2020-01-01, 2026-01-01)`** end exclusive); override with **`--start`** / **`--end`**.

```bash
uv sync --extra timescale
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/shunya
uv run python scripts/bootstrap_ts_data.py --dry-run
uv run python scripts/bootstrap_ts_data.py
```

Requires the repo clone ( **`examples/yfinance_fundamental_provider`** ). See **`--help`** for **`--full`**, **`--single-pass`**, and rate-limit knobs.

## Environment

- **`DATABASE_URL`** or **`SHUNYA_DATABASE_URL`** (some scripts also accept **`--database-url`**).
- Optional **`YFINANCE_TLS_VERIFY=1`** for strict TLS when using yfinance paths (see API README).

## See also

- [Timescale first-run checklist](timescale-checklist.md)
- [Local Timescale](../data_timescale.md)
- [Quickstart](../quickstart.md) — Path B full stack
