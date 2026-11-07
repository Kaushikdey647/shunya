# HTTP behavior beyond OpenAPI

Swagger documents paths, query parameters, and JSON bodies. This page records lifecycle and socket behavior that a schema listing does not.

## Authentication

When `SHUNYA_API_TRADE_DESK_TOKEN` is set, `POST /trade/paper/cycle`, manual blotter routes under `/trade/orders`, and `PATCH /settings/app` require `X-Shunya-Trade-Desk-Token`. The blotter WebSocket takes the same secret as the `token` query parameter. `GET /settings/app` stays readable without that header.

## Backtest jobs

`POST /backtests` normalizes the window to `[2020-01-01, 2026-01-01)` and daily bars, resolves an index or saved universe when those fields are set, and inserts a `queued` row.

The default worker is an asyncio task in the same uvicorn process. It claims a row with `FOR UPDATE SKIP LOCKED`, runs `BacktraderBacktestEngine` (which calls `FinBT`), and stores both a summary and the full payload.

| Call | Body |
|------|------|
| `GET /backtests/{job_id}` | Status, error fields, and `result_summary`. |
| `GET /backtests/{job_id}/result` | Full serialized result after `succeeded`. Other statuses return `BACKTEST_RESULT_NOT_READY`. |
| `GET /backtests/{job_id}/logs` | Captured log lines for that job. |

The sequence is [Asynchronous backtest sequence](../documentation/backtest-sequence.md).

## Notification codes

`/notifications/stream` emits `hello`, then `notification` frames (`schema: 1`). Lifecycle codes:

- `backtest.queued`, `backtest.started`, `backtest.succeeded`, `backtest.deleted`, `backtest.batch_deleted`
- On failure, `code` is the domain `ErrorCode` string (for example `BACKTEST_JOB_EXECUTION_ERROR` or `ALPHA_NOT_FOUND`), not a fixed `backtest.failed`
- `alpha.created`, `alpha.updated`, `alpha.deleted`
- `universe.created`, `universe.updated`, `universe.deleted`, and member-change codes
- `settings.app_patched`
- `us_rth_closed` when an Alpaca L1 socket is rejected outside the US regular session

Subscribers are in-process. A second uvicorn worker does not see the first worker's notifications.

## Order stream

`WS /trade/orders/stream` receives `{"type":"order",...}` and `{"type":"fill",...}` from `OrderEventHub`. Manual placement does not run `PortfolioRiskEngine` or EMS slicing. See [Manual order sequence](../documentation/manual-order-sequence.md).

## Market clock stream

`WS /settings/market-clock/stream` sends `hello`, then `tick` frames with the same fields as `GET /settings/market-clock`.
