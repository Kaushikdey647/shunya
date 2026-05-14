# Alphas, metrics, and evaluation

This page is about **finance and measurement**. For **`FinStrat` / `FinBT` APIs** and code knobs, see [Signals: FinStrat and FinBT](../documentation/finstrat-finbt.md).

## What is an alpha?

In systematic equity research, an **alpha** is a **repeatable rule** that assigns each asset \(i\) at time \(t\) a **score** (or forecast) \(s_{i,t}\) using only information available at \(t\). Scores are meant to **rank or weight** names so that, **in expectation**, higher-scored names outperform lower-scored names over a horizon you care about (often implicitly the next rebalance, day, or week).

- **Economic content:** the alpha encodes a **hypothesis** (e.g. “cheap stocks on a value proxy outperform”) or a **statistical regularity** (e.g. “12-1 momentum continues short term”).
- **Portfolio link:** scores become **portfolio weights** (dollar-neutral long/short or long-only) after **normalization** (e.g. cross-sectional rank or z-score) and **risk overlays** (sector neutral, caps). In Shunya, **`FinStrat.pass_`** turns your `algorithm(ctx)` output into tradable targets subject to `neutralization`, `truncation`, `max_single_weight`, etc.

## How does an alpha “generate returns”?

Returns come from **exposure** to rewarded sources of risk or mispricing, not from the formula alone:

1. **Signal** \(s_{i,t}\) ranks assets.
2. **Weights** \(w_{i,t}\) allocate capital (long winners, short or underweight losers in a long-short book).
3. **Realized PnL** is \(\sum_i w_{i,t-1}\, r_{i,t}\) plus **costs** (commissions, bid-ask, market impact).

Shunya **`FinBT`** simulates that path in **backtrader** with explicit **`commission`** and **`slippage_pct`**, so you separate **gross signal quality** from **net implementable** performance.

## How we evaluate an alpha

Good practice combines **statistical** and **economic** checks:

- **In-sample vs out-of-sample:** tune on one window; report on a **held-out** period (the HTTP API can exclude a trailing test slice via `include_test_period_in_results`; see [api/README](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)).
- **Stability:** decaying performance often means the effect is spurious or crowded.
- **Capacity and turnover:** high turnover erodes net returns after costs.

In Shunya, **`FinBT(...).run()`** then **`results = bt.results(show=False)`** yields metrics and series you can also surface in the **web UI** after an async job completes.

## Common metrics (with formulas)

Notation: simple returns \(r_t\), risk-free \(r_f\) (often 0 in daily research), portfolio return \(R_t = \sum_i w_{i,t-1} r_{i,t}\). Use \(\mu(\cdot)\) and \(\sigma(\cdot)\) for sample mean and standard deviation over the evaluation window.

### Sharpe ratio

Risk-adjusted return vs volatility:

\[
\text{SR} = \frac{\mu(R_t - r_f)}{\sigma(R_t)}
\]

Annualize with \(\sqrt{252}\) for daily data when reporting conventionally.

### Sortino ratio

Penalizes **downside** volatility only. Let \(d_t = \min(0, R_t - \text{MAR})\) for a minimum acceptable return MAR (often 0):

\[
\text{Sortino} = \frac{\mu(R_t - r_f)}{\sigma(d_t)}
\]

### Maximum drawdown

Peak-to-trough loss on **cumulative wealth** \(W_t = \prod_{k \le t}(1+R_k)\):

\[
\text{MDD} = \min_t \left( \frac{W_t - \max_{s \le t} W_s}{\max_{s \le t} W_s} \right)
\]

### Hit rate

Fraction of periods with \(R_t > 0\) (or vs benchmark); useful but **not** sufficient alone.

### Turnover

One common definition (with weights summing to 1 long-only or gross leverage \(L\) long-short):

\[
\text{Turnover}_t = \frac{1}{2}\sum_i |w_{i,t} - w_{i,t-1}|
\]

High turnover raises **implementation shortfall**.

### Information coefficient (IC)

Cross-sectional **Spearman or Pearson** correlation between signal at \(t\) and **forward** return over \(h\):

\[
\text{IC}_t = \text{corr}\big(s_{i,t},\, r_{i,t+1:t+h}\big)
\]

Stable positive mean IC suggests predictive content; check **IC decay** across horizons.

### Math in this site

Inline math uses `\(` … `\)` and display math uses `\[ … \]` (see [pymdownx Arithmatex](https://facelessuser.github.io/pymdown/extensions/arithmatex/) with `generic: true`).

## Where this appears in Shunya

| Idea | Shunya surface |
|------|----------------|
| Alpha as vector signal | `algorithm(ctx) -> jnp.ndarray` in **`FinStrat`** ([docs](../documentation/finstrat-finbt.md)) |
| Simulation + costs | **`FinBT`**, `commission`, `slippage_pct` |
| Metrics / tearsheets | `bt.results(show=False)`; HTTP backtest JSON; UI charts |

## Further reading

- [Alpha design: rank, z-score, gates](alpha-design-rank-zscore-and-gates.md)
- [Pipeline: alpha to execution](pipeline-alpha-to-execution.md)
