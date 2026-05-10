# Alpha platform context (for AI assist)

Alphas implement a function body (saved inside a module template) that returns **per-ticker scores** on a bar grid. The runtime injects `ts`, `cs`, and `fun` as aliases for time-series ops, cross-section ops, and fundamentals on `AlphaContext`.

- **Cross-section ops (`cs`)** operate on the **latest bar** snapshot (e.g. `cs.rank(x)` ranks tickers at the current date).
- **Time-series ops (`ts`)** roll over full **(time, ticker)** history (e.g. `ts.mean(ctx.close, 20)`).
- **Scores** feed portfolio construction in backtests: signals are turned into weights under finstrat constraints (neutralization, decay, NaN policy, etc.). Mis-using `cs` vs `ts` is a common pitfall (ranking a level vs ranking changes).
- **Fundamentals (`fun`)** are sparse vs daily prices; mixing statement-period fields with daily fields without alignment awareness can leak lookahead bias if you combine them naively with price signals. Cross-sectional ranking of aligned fundamentals (e.g. `cs.rank(fun.Revenue)`) is a normal pattern and is not inherently wrong.
- **JAX**: prefer vectorized ops; avoid Python loops over tickers.

When reviewing or assisting on an alpha, apply fundamental-alignment and vectorization advice **only** if the body actually uses `fun` or an explicit loop over tickers; otherwise those points are not relevant.

This appendix is static product context; combine it with the alpha name/description supplied by the user.
