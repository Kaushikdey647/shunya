# Create and run a backtest from the web UI

This walkthrough assumes [Local dev: API, worker, and UI](local-dev-api-ui.md) is already working (API on port 8000, UI on Vite’s port, database migrated if you use the job queue).

## 1. Open Alpha Studio

In the sidebar under **Studio**, choose **Studio** (hub) or go to `/studio`. Open an existing alpha workspace (`/studio/:alphaId`) or create one via **New** (`/studio/new`).

## 2. Edit the alpha

- **Inline Python** — stored `source_code` in the API database; executed in the **worker** at backtest time when present (takes precedence over `import_ref` when non-empty).
- **`import_ref`** — allow-listed module path such as `examples.alphas.<module>:alpha` when not using inline code.

FinStrat JSON config is stored as **`finstrat_config`** on the alpha record.

## 3. Enqueue a backtest

Use the Studio UI control to create a **backtest job** (POST to `/backtests`). The worker picks up jobs from the database queue.

## 4. Follow progress

Under **Studio → Backtests** (`/backtests`), open the list and then a job detail page (`/backtests/:jobId`) for status, logs, and (when `succeeded`) charts and metrics.

## HTTP semantics (what the UI relies on)

From the authoritative [api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md):

- **`POST /backtests`** uses a **fixed** simulation window **`[2020-01-01, 2026-01-01)`** (end exclusive) and **daily** bars. Client-supplied `fin_ts.start_date`, `end_date`, and `bar_spec` are **overwritten** by the server.
- **`include_test_period_in_results`** (default `false`) — when `false`, stored metrics and time series **exclude** the **test** slice from **2025-01-01** onward (tune-only view in dashboards).
- **`POST` with `index_code`** — resolves constituents from `symbol_index_membership`, sets benchmark to the catalog raw index symbol, forces **`market_data_provider=timescale`** and **no Yahoo**. You can set **`omit_index_members_missing_ohlcv`: true** to drop members without bars; the benchmark must still have data.

## Troubleshooting

- **Job stuck / failed** — confirm the **API process** (`uvicorn api.main:app`) is running with the same **`DATABASE_URL`** you use from the UI (the in-process worker loop runs inside that process; there is no separate worker binary to start by default). Check API logs for claim/run errors.
- **Queue test flakes** — if you also run pytest against the same DB, see the “job queue” warnings in [api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md).

## See also

- [Studio](../ui/studio.md) — UI features (Monaco, lint, results).
- [HTTP API](../http-api.md) — route outline.
