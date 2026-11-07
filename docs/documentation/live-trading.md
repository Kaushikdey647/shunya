# Daily live trading

Shunya trades US equities once a day with **market-on-open (OPG)** orders on Alpaca. Signals use bars through the close of session **T**. Orders are sent that evening and fill in the opening auction of the next session **N**. That matches FinBT, where orders placed on bar T fill at T+1's open.

```
Local machine                                   Railway
UI ─► local API ─(SHUNYA_LIVE_URL + token)────► api.live_main (status / books / kill)
        │                                       cron: shunya-live tick (every 15 min)
        └─ fallback kill with local APCA_* keys   Postgres (live_* tables) ─► Alpaca
```

## Components

| Module | Role |
|---|---|
| `shunya/live/runner.py` | `tick()`: decides what is due, submits and reconciles |
| `shunya/live/signals.py` | FinStrat replay over the lookback window, so decay matches the backtest |
| `shunya/live/broker.py` | Alpaca access; every submit re-checks the kill switch |
| `shunya/live/control.py` | Sticky kill switch: `halt` / `cancel` / `flatten` |
| `shunya/live/store.py` | `live_*` tables (`shunya/live/migrations`) |
| `api/live_main.py`, `api/routers/live.py` | Hosted API (`X-Shunya-Live-Token`) |
| `api/routers/live_bridge.py` | Local proxy, deploy, and kill fallback |
| `ui/src/pages/LivePage.tsx` | Dashboard (polls every 15 s) |

## Timing

All times come from Alpaca's calendar, so holidays, half-days and DST need no special handling.

- **Submit window:** `T.close + 4h15m` to `N.open − 8 min`, which is 20:15 to 09:22 ET on a normal day. Alpaca accepts OPG orders from 19:00 to 09:28 ET; the later start gives late prints and corrections time to settle into the daily bar.
- **Reconcile:** from `N.open + 15 min`. Fills are recorded, split across sleeves, and checked against Alpaca positions.
- **Idempotency:** each book and session gets one `live_runs` row. Order ids are `shy-{book}-{YYYYMMDD}-{SYMBOL}`, so a retry can't double-submit. Failed runs are retried up to 3 times inside the window.
- **Blocking:** a new submission waits while an earlier run is still unreconciled.

## Market data

The runner reads **Alpaca IEX** daily bars (`SHUNYA_ALPACA_BAR_FEED=iex`, the default), **split- and dividend-adjusted** (`SHUNYA_ALPACA_BAR_ADJUSTMENT=all`, the default; Alpaca's own API default is unadjusted).

- **IEX is one venue.** Its closes can differ slightly from the official close, and its volume is a few percent of the consolidated tape. Deploying an alpha that reads volume shows a warning.
- **Gaps:** thinly traded names can have days without IEX prints. Up to 5% of a universe may be missing; names without a bar on the signal date are skipped. Beyond that the run fails and auto-halts.
- **Backtest parity:** ingest the same bars into Timescale and point backtests at them:

  ```bash
  shunya-timescale ingest-ohlcv --provider alpaca --symbols-from-db --start 2016-01-01 --end 2026-11-05
  # then in .env:
  SHUNYA_STORED_OHLCV_SOURCE=alpaca_iex
  ```

  Rows are stored with `source=alpaca_iex` next to the yfinance rows. The Yahoo backfill that retries failed backtests is skipped while the stored source isn't `yfinance`.
- **Re-ingest after splits.** Adjusted history is anchored at ingest time, so after a split or dividend the stored history is stale. Re-run the full-window ingest periodically (for example weekly); it upserts in place.

## Books and sleeves

- **Strategy version:** a frozen alpha source, FinStrat config and ticker list. The hosted runner never reads research tables.
- **Book:** a capital fraction of account equity plus a `risk_config` (`max_drawdown`, `max_single_name_fraction`, `max_rejections`, `max_equity_jump`). v1 allows one active book per account.
- **Sleeve:** a strategy version with a weight in its book. Statuses:
  - `active`: trades on its signal.
  - `paused`: holds its current shares.
  - `draining`: target 0.
  - `retired`: ignored.
  - `ramp_days` phases in a new sleeve's weight.
- **Netting:** sleeve targets are summed, vetted by `PortfolioRiskEngine`, rounded to whole shares, and netted against Alpaca positions and working orders.
- **v1 limits:**
  - Long-only.
  - No fundamentals (`fun.*`).
  - Neutralization `none` / `market` only.
  - FinBT post-pass caps (sector, turnover, ADV) are not applied live.

## Kill switch

| Level | Effect |
|---|---|
| `halt` | Sticky flag; no new orders |
| `cancel` | `halt` plus cancel all open orders (queued OPG orders included) |
| `flatten` | `cancel` plus close all positions (after hours, the closes queue for the next open) |

- **Order of operations:** the flag is written first, then the broker is called. Broker steps still run if the database write fails.
- **Fail closed:** if the flag can't be read, the runner treats it as halted.
- **Re-arming:** only `resume` clears a kill.
- **Dashboard fallback:** the dashboard's kill tries the hosted service first. If that fails, it cancels or flattens directly at Alpaca with the local keys.
- **Automatic `cancel`:**
  - drawdown beyond `max_drawdown`
  - equity moving more than `max_equity_jump` between runs
  - stale bars
  - too many rejections
  - sleeve ledger not matching Alpaca positions

## Setup

1. Create a Railway Postgres and two services from this repo:
   - **web**, with config path `deploy/railway/live-web.toml`
   - **cron**, with `deploy/railway/live-cron.toml`
2. Set the service variables on both: `SHUNYA_LIVE_TOKEN`, `DATABASE_URL`, `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `SHUNYA_ALPACA_PAPER=1`, and optionally `SHUNYA_LIVE_HEARTBEAT_URL`.
3. Add `SHUNYA_LIVE_URL` and `SHUNYA_LIVE_TOKEN` to the local `.env`, along with `SHUNYA_API_ALPACA_ENABLED=1` (this enables the kill fallback).
4. In the UI, open **Live**, create a book, then deploy an alpha + universe as a sleeve.
5. Check before the first evening with `shunya-live plan --book <id>`, a dry run that prints targets and orders.

CLI: `shunya-live migrate | tick [--now ISO] | plan --book ID | halt --level cancel --reason ... | resume --reason ... | status`.
