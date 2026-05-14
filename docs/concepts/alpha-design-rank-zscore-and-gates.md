# Alpha design: rank, z-score, and gates

For **API listings** of `cross_section`, `logical`, and `time_series`, see [Operators](../documentation/operators.md).

## Cross-sectional rank

**When to use:** you want a **long-short** or **relative value** book and only care about **ordering** within the universe each day. Ranks dampen outliers and make signals comparable across names with different units.

**Math:** for scores \(s_i\) at a bar, rank maps to order statistics (Shunya uses a JAX-friendly `rank` that is **increasing** in \(s_i\): **larger** score → **larger** rank value).

**In Shunya:** `ctx.cs.rank(x)` or `cross_section.rank`. To **buy losers / sell winners** (contrarian), rank **`-signal`** instead of `signal`.

## Cross-sectional z-score

**When to use:** you want **standardized** deviations from the cross-sectional mean — “how many sigmas away from peers?” Good for combining signals on **similar scales** after winsorizing tails.

\[
z_i = \frac{s_i - \mu(s)}{\sigma(s)}
\]

**Caveat:** if \(\sigma(s) \approx 0\) (whole market flat on that feature), z-scores blow up — pair with **`truncation`** on **`FinStrat`** or winsorize first (`cs.winsorize`).

## Raw values (levels, spreads, yields)

**When to use:** the **unit is economically meaningful** and comparable across names without ranking (e.g. everyone’s **dividend yield** in percent). Also used **inside** time-series steps before a final `cs.rank` (e.g. `ctx.close / ts.mean(ctx.close, 50)`).

**Risk:** raw signals can be dominated by **extreme microcaps** or bad data; combine with **`truncation`**, liquidity filters, or **`max_single_weight`**.

## Gates: `trade_when` and `jnp.where`

**When to use:** **regime** switching (on/off), **entry/exit** rules, or **hysteresis**. Example patterns:

- **Volatility filter:** only hold momentum when vol &lt; median vol.
- **Earnings blackout:** zero exposure around earnings if you had a calendar feature (not shown here).

**Alpha Studio:** `jnp.where(cond, signal, 0.0)` is explicit and fast to read. **Library Python:** `logical.trade_when` expresses richer entry/exit pairs.

## Neutralization choice (brief)

- **`market`** — dollar-neutral vs the cross section; classic long-short equity.
- **`sector` / `industry`** — removes **structural** bets you do not want to pay for; pairs names within the same group.
- **`group`** — custom baskets (e.g. peer sets).

See **`neutralization`** on **`FinStrat`** in [Signals: FinStrat and FinBT](../documentation/finstrat-finbt.md).

## Further reading

- [Alphas, metrics, and evaluation](alphas-metrics-and-evaluation.md)
- [Alpha styles and Shunya patterns](alpha-styles-and-shunya-patterns.md)
