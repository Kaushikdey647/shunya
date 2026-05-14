# Signals: `FinStrat` and `FinBT`

## `FinStrat`

`FinStrat(fin_ts, algorithm, ...)` binds a **`finTs`** panel to a callable:

```python
def algorithm(ctx) -> jnp.ndarray:
    ...
```

The context **`ctx`** exposes:

- Base series: `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.adj_volume`
- Time-series helpers: `ctx.ts.*` (for example `ctx.ts.mean(ctx.close, 50)`)
- Cross-sectional helpers: `ctx.cs.*` (for example `ctx.cs.rank(signal)`)

### Knobs (BRAIN-like)

- **`decay`** — per-name EMA on raw scores; pass `tickers=` into `pass_` where applicable.
- **`truncation`** — cross-sectional winsorize.
- **`neutralization`** — `"market"`, `"none"`, `"sector"` (demean within `Sector`), `"industry"` (within `Industry`), or `"group"` (caller-supplied groups / `group_column` on **FinBT**).
- **`max_single_weight`** — cap on single-name weight.

For `neutralization="group"`, pass **`group_column`** on **FinBT** (defaults to `"Sector"` when omitted). `"sector"` / `"industry"` require the corresponding columns on `fin_ts.df`.

### Temporal modes

- **`temporal_mode="bar_step"`** — advance decay one step per bar.
- **`temporal_mode="elapsed_trading_time"`** — advance decay by trading-time distance; **FinBT** passes execution timestamps so this works without extra wiring.

## `FinBT`

`FinBT(fin_strat, fin_ts, ...)` runs the same **FinStrat** on the same **`fin_ts`** instance inside **backtrader**, rebalancing to **`pass_`** dollar targets each bar.

- Call **`run()`** to execute; it resets **FinStrat** decay state as implemented.
- Pass **`commission`** (broker rate) and optional **`slippage_pct`** (adverse percent via backtrader’s `set_slippage_perc`).
- Helpers such as **`broker_deltas`** / **`target_usd_universe`** in `shunya.algorithm.targets` mirror how live orders diff targets vs positions.

## Minimal example

```python
import jax.numpy as jnp
from shunya import FinBT, FinStrat, finTs

fts = finTs("2023-01-01", "2024-01-01", ["AAPL", "MSFT", "NVDA"])

def alpha(ctx) -> jnp.ndarray:
    sma_50 = ctx.ts.mean(ctx.close, 50)
    signal = ctx.close / sma_50
    return ctx.cs.rank(signal)

fs = FinStrat(
    fts,
    alpha,
    neutralization="sector",
    truncation=0.02,
)

bt = FinBT(fs, fts, cash=100_000.0, commission=0.0005, slippage_pct=0.0005).run()
results = bt.results(show=False)
```

## Further reading

- [Operators](operators.md) — `cross_section`, `logical`, `time_series`, `group_ops` used inside `algorithm`.
- [finTs and providers](fints-providers.md) — panel construction and calendars.
- [Python API reference](../reference/library.md).
