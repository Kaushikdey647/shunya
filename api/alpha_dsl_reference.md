# Alpha DSL (body editor)

Inside `alpha(ctx)` the saved module injects `ts = ctx.ts`, `cs = ctx.cs`, `fun = ctx.fun`. Use **OHLCV** via `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.adj_volume`. Named features: `ctx.feature("Name")`.

## Time series (`ts`)

| Call | Meaning |
|------|---------|
| `ts.delay(x, lag)` | Lag series |
| `ts.delta(x, lag)` | Difference |
| `ts.sum(x, window)` | Rolling sum |
| `ts.mean(x, window)` | Rolling mean |
| `ts.std(x, window)` | Rolling stdev |
| `ts.zscore(x, window)` | Rolling z-score |
| `ts.rank(x, window)` | Rolling rank over time |
| `ts.regression(y, x, window, lag, retval)` | TS regression; `retval` in `b\|a\|r\|t` |
| `ts.humpdecay(x, hump)` | Hump decay |

`x` is `AlphaSeries` or JAX array shaped `(time, n_tickers)`.

## Cross-section (`cs`)

Operates on **latest bar** snapshot.

| Call | Meaning |
|------|---------|
| `cs.rank(x)` | Cross-section rank |
| `cs.zscore(x)` | CS z-score |
| `cs.scale(x, target=1.0)` | Rescale gross |
| `cs.sign(x)` | Sign |
| `cs.winsorize(x, tail)` | Winsorize tails |
| `cs.neutralize_market(x)` | Market-neutral |
| `cs.neutralize_groups(x, group_ids)` | Group-neutral |

## Fundamentals (`fun`)

`fun.Revenue`, `fun.Net_Income`, … (statement fields) and `fun.Market_Cap`, … (daily fields). Same as `ctx.feature("ColumnName")`. Aliases work too: camelCase (`fun.revenue`, `fun.debtToEquity`), lower_snake (`fun.net_income`), or UPPER_SNAKE — all resolve to the canonical panel column names.

## JAX (`jnp`)

`jnp.array`, `zeros`, `ones`, `sqrt`, `log`, `exp`, `where`, `abs`, `mean`, `std`, … — prefer vectorized ops over Python loops over tickers.
