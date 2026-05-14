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
- **Default universe** — optional saved equity universe (`default_universe_id` on **`/alphas`**) used by the portfolio workspace union and as a convenience when configuring runs.

See [Alpha Studio: AI assist and DSL](../how-to/alpha-studio-ai-dsl.md) for Ollama env and DSL roots.

## Universes (`/universe`, `/universe/:id`)

Create and maintain saved universes (**`/universes`** API): members, sector/industry breakdown, fundamentals summary. Use them from **Backtests → New** (saved universe + benchmark) or add symbols from **Instrument** detail.

## Backtests (`/backtests`, `/backtests/new`, `/backtests/:jobId`)

- **List** — job status and navigation to detail.
- **New** — configure and submit **`POST /backtests`** (index universe, saved **`universe_id`** + **`benchmark_ticker`**, or explicit ticker list per API rules).
- **Detail** — logs, charts, and payload when `succeeded`.

HTTP semantics (fixed date window, `include_test_period_in_results`, index runs): [Backtest from the web UI](../how-to/backtest-from-ui.md).

## See also

- [FinStrat and FinBT](../documentation/finstrat-finbt.md) (code); [Alphas, metrics](../concepts/alphas-metrics-and-evaluation.md) (finance)
- [HTTP API](../http-api.md)
