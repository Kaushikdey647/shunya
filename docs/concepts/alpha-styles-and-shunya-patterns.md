# Alpha styles and Shunya patterns

Each subsection gives **finance intuition** and a **minimal pattern** using the same surfaces as Alpha Studio (`ctx`, `ts`, `cs`, `fun`, `jnp`) and library helpers where noted. For API details see [Signals: FinStrat and FinBT](../documentation/finstrat-finbt.md) and [Operators](../documentation/operators.md).

## Momentum (trend following)

**Idea:** assets that rose over a medium horizon tend to continue outperforming **short term** (subject to crashes and crowding).

```python
def alpha(ctx):
    ret20 = ctx.close / ts.delay(ctx.close, 20) - 1.0
    return cs.rank(ret20)
```

Use `cs.rank(-ret20)` if your convention wants **higher = stronger past decline** (reversal flavor) — recall `cross_section.rank` is **increasing** in its argument ([operators](../documentation/operators.md)).

## Reversal (mean reversion)

**Idea:** short-term stretched moves **snap back** relative to peers.

```python
def alpha(ctx):
    z = ts.zscore(ctx.close - ts.mean(ctx.close, 20), 60)
    return cs.rank(-z)
```

## Conditional (regime / gate)

**Idea:** apply a signal **only when a filter** holds (e.g. volatility regime). In Alpha Studio you typically gate with **`jnp.where`** on arrays aligned with the cross section:

```python
def alpha(ctx):
    mom = ctx.close / ts.delay(ctx.close, 60) - 1.0
    vol = ts.std(ctx.close, 20)
    vol_med = ts.mean(vol, 60)
    return jnp.where(vol > vol_med, cs.rank(mom), 0.0)
```

In full Python **`FinStrat`** algorithms (notebooks, modules), prefer **`shunya.algorithm.logical.trade_when`** for readable entry/exit logic ([operators](../documentation/operators.md)).

## Fundamental (quality / value)

**Idea:** ratios and statements from financial reports predict **long-horizon** returns or risk; cross-sectionally compare peers.

```python
def alpha(ctx):
    # fun.* aligns with shunya.data.fundamentals (see fundamentals doc)
    fcfy = fun.Free_Cash_Flow_Yield
    return cs.zscore(fcfy)
```

## Technical (price/volume features)

**Idea:** patterns in **OHLCV** (trend, range, volume confirmation) proxy for supply/demand.

```python
def alpha(ctx):
    rsi_like = 100.0 - 100.0 / (1.0 + ts.mean(jnp.maximum(ctx.close - ctx.low, 1e-9), 14) / ts.mean(jnp.maximum(ctx.high - ctx.close, 1e-9), 14))
    return cs.rank(rsi_like)
```

(Production RSI often uses smoother Wilder smoothing; this illustrates **composition** with `ts.mean` and `ctx` extremes.)

## Combining styles

Many products blend **slow fundamental** with **fast technical** filters. In Shunya, combine inside `algorithm` with `jnp.where`, `logical.trade_when`, or add a second sleeve via **`AlphaBlendConfig`** at the PCS layer ([portfolios doc](portfolios-construction-and-pms.md)).

## Further reading

- [Fundamentals for alphas](fundamentals-for-alphas.md)
- [Alpha design: rank, z-score, gates](alpha-design-rank-zscore-and-gates.md)
