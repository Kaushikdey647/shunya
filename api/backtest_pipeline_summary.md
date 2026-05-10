# Backtest pipeline (conceptual)

1. **Data** — `FinTsRequest` loads an OHLCV (+ optional fundamentals) panel for tickers and a date window. Bar grid and provider flags control strictness.

2. **`alpha(ctx)`** — Your function returns **per-ticker scores** on the bar grid (same shape convention as `ctx.close`: time × tickers). The runtime builds `AlphaContext` with OHLCV, attached features, `ts` / `cs` / `fun`.

3. **FinStrat** — Turns scores into **targets** (weights) under decay, signal delay, NaN policy, temporal mode, and neutralization (none / market / sector / industry). Constraints cap gross/leverage and turnover-style budgets where configured.

4. **FinBT** — Simulates execution with cash, commission, slippage, sector/group caps, and participation limits. Output includes **metrics** (e.g. return, Sharpe, drawdown summaries) and series used for charts.

5. **Interpretation** — Poor metrics can come from signal quality, overfitting to noise, wrong `cs` vs `ts` usage, lookahead in fundamentals, or portfolio constraints clipping the signal. Use metrics together, not a single headline number.
