# Studio area (sidebar)

## Alpha Studio hub (`/studio`)

List and navigate alpha definitions stored via **`/alphas`**.

## New alpha (`/studio/new`)

Create a record with name, optional **`import_ref`**, optional inline **`source_code`**, and **FinStrat** JSON configuration.

## Workspace (`/studio/:alphaId`)

- **Monaco editor** for Python alpha body when using inline storage.
- **Lint** and **assist** call FastAPI (optional Ollama on the server).
- **Inline DSL** hints from `ui/src/alphaEditor/alphaDslCatalog.ts` (`ctx`, `ts`, `cs`, `fun`, `jnp`).
- **Enqueue backtest** from the workspace; results render below the editor when the job completes (metrics strip, optional AI review of numbers, Recharts / lightweight-charts tearsheets).

See [Alpha Studio: AI assist and DSL](../how-to/alpha-studio-ai-dsl.md) for Ollama env and DSL roots.

## Backtests (`/backtests`, `/backtests/new`, `/backtests/:jobId`)

- **List** — job status and navigation to detail.
- **New** — configure and submit **`POST /backtests`**.
- **Detail** — logs, charts, and payload when `succeeded`.

HTTP semantics (fixed date window, `include_test_period_in_results`, index runs): [Backtest from the web UI](../how-to/backtest-from-ui.md).

## See also

- [FinStrat and FinBT](../concepts/finstrat-finbt.md)
- [HTTP API](../http-api.md)
