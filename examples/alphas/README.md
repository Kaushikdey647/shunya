# Example Alphas

This folder contains context-style alpha examples for `FinStrat`.

## Contract

Each alpha function follows:

```python
def alpha(ctx) -> jnp.ndarray:
    ...
```

Where `ctx` provides:

- `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.adj_volume`
- `ctx.features["Some_Field"]` or `ctx.feature("Some_Field")` for attached fundamentals
- `ctx.ts.*` time-series operators
- `ctx.cs.*` cross-sectional operators

## Included

- `sma_ratio_50`: trend via `close / SMA(50)`
- `mean_reversion_5`: short-term reversion via negative 5-bar deviation
- `breakout_20`: 20-bar momentum via delayed-close ratio
- `fundamental_value_quality`: combine price trend with ROE and P/E
- `volume_price_trend_20`: trend weighted by relative volume

## Usage

```python
from shunya import FinStrat, finTs
from examples.alphas import sma_ratio_50

fts = finTs("2023-01-01", "2024-01-01", ["AAPL", "MSFT", "NVDA"])
fs = FinStrat(fts, sma_ratio_50, neutralization="market")
```

## Fundamentals

When `finTs(..., attach_fundamentals=True, ...)` is used, attached fields become available in
the alpha context through `ctx.features[...]` and `ctx.feature(...)`.

```python
def alpha(ctx):
    roe = ctx.feature("Return_On_Equity")
    pe = ctx.feature("Price_To_Earnings")
    signal = ctx.ts.zscore(roe, 4) - ctx.ts.zscore(pe, 4)
    return ctx.cs.rank(signal)
```

