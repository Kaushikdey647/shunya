# Shunya: project rules

These rules are binding for all work in this repo. They override older docs/TODO items that contradict them.

## Agenda (Alpaca-only migration)

1. **Broker:** trading goes through Alpaca only (`alpaca-py`). No Kite or other brokers.
2. **Data:** the only market data source is Alpaca **IEX** bars. No yfinance, Tiingo, Alpha Vantage or Kite data.
   Remove the yfinance bypasses: curl_cffi impersonation, `verify=False` sessions added for Yahoo, rate-limit sleeps, the repair logic, and the Yahoo backfill retries.
   `SHUNYA_TLS_VERIFY` stays only where Alpaca itself needs it behind corporate TLS inspection.
3. **Execution:** a scheduled cron job (Railway) aggregates the signals of every deployed alpha and trades them in one of two modes:
   - `open`: the signal uses session T's full daily bar. Orders are market-on-open (OPG) and fill at session T+1's opening auction.
   - `close`: the signal uses session T's bar as of the pre-close cutoff. Orders are market-on-close (CLS) and fill at session T's closing auction.
   - Each alpha declares its mode. Its backtest uses the dataset that matches that mode, so backtest and live see the same data shape.
   - Algorithms run on a **daily timeframe at minimum**. There is no realtime trading: no WebSocket/SSE streaming, OMS/EMS child-order slicing, trade desk or intraday bar specs. Alpaca is accessed over REST only.
   - The scheduled job is the daily cron runner (`shunya-live tick`).
4. **Symbol master:** comes from SEC `company_tickers.json`, with a boolean `symbols.is_alpaca_tradable` (indexed) set from Alpaca `/v2/assets`.
   - Non-tradable entities are shown in ticker info only.
   - Every trading or Alpaca-related path filters on the flag: universes, algorithm inputs, ingest, viewing and indexing historical bars, and the runner.
   - Classification comes from `data.sec.gov/submissions/CIK{10-digit}.json`: `sic`, `sicDescription` and `ownerOrg` go into the `sic_codes` table. `filings` is ignored.
   - Reference data is loaded by a one-time job, re-runnable for new tickers. Backtests and order cycles read it from the DB only and never call external APIs per run or iteration.
5. **Storage and compute:** the Timescale schema, ingest, finTs and the JAX/AlphaContext tensors are designed around Alpaca bars (vwap, trade_count, IEX gaps, corporate actions), not around Yahoo.

## Workflow

- Work is tracked as GitHub issues under milestones in `Kaushikdey647/shunya`. Tackle them one at a time, in milestone order.
- After each issue, stop and report to the user: what was fixed, how, what was tested, and what still needs a live Alpaca check. Wait for go-ahead before starting the next one.
- There is no live Alpaca access from the dev machine. Use fakes/fixtures that mirror real alpaca-py response shapes. Mark anything only verifiable live as `needs-paper-verification` in the issue.
- Commit per issue on a feature branch and reference the issue (`Fixes #N`). Don't push or open PRs without asking.
- Tests: `uv run pytest -q --ignore=tests/test_timescale_integration.py`. Tests that need Postgres fail locally without a DB; that is expected.
- Delete dead code; don't leave compatibility shims for removed providers. Update docs in the same change as the code.
