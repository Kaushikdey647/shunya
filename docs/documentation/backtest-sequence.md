# Asynchronous backtest sequence

Happy path for a UI or API backtest. The worker loop starts in the API process lifespan. It is not a separate OS process unless you deploy one yourself.

`GET /backtests/{job_id}` returns status and `result_summary`. The chart payload is `GET /backtests/{job_id}/result` after `succeeded`.

```mermaid
sequenceDiagram
    participant UI as BacktestUI
    participant API as BacktestsRouter
    participant Norm as normalize_backtest_create
    participant Resolve as UniverseOrIndex
    participant Repo as BacktestRepository
    participant DB as Postgres
    participant Worker as backtest_worker_loop
    participant Job as execute_claimed_backtest_job
    participant Runner as run_backtest_job
    participant Panel as build_fin_ts
    participant Alpha as resolve_alpha_for_backtest
    participant Strat as FinStrat
    participant Engine as BacktraderBacktestEngine
    participant BT as FinBT
    UI->>API: POST /backtests
    API->>Norm: fixed daily window
    Norm->>Resolve: index or universe members
    Resolve->>Repo: insert queued job
    Repo->>DB: api_backtest_jobs
    API-->>UI: job id
    Worker->>Repo: claim_next_queued_job
    Repo->>DB: status running
    Worker->>Job: execute_claimed_backtest_job
    Job->>Runner: run_backtest_from_payload
    Runner->>Panel: finTs via market-data route
    Runner->>Alpha: inline source or import_ref
    Alpha->>Strat: algorithm
    Runner->>Engine: run_backtest
    Engine->>BT: run then results
    BT-->>Runner: metrics and series
    Runner-->>Job: serialized payload and summary
    Job->>Repo: mark_job_succeeded
    Repo->>DB: result_payload and result_summary
    UI->>API: GET /backtests/{job_id}
    API-->>UI: status and result_summary
    UI->>API: GET /backtests/{job_id}/result
    API-->>UI: full JSON result
```

Notifications on this path use `backtest.queued`, `backtest.started`, and `backtest.succeeded`. A failed job stores `error_code` from `shunya.errors.ErrorCode` and publishes that same code.

Library-only `FinBT` calls skip the queue. See [Signals: FinStrat and FinBT](finstrat-finbt.md) and [Ports and adapters](../reference/ports-adapters.md).
