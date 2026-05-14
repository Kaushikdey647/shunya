/**
 * Single catalog for completion + inline suggest (keep in sync with
 * ``shunya/data/fundamentals.py`` for fun.* names).
 */
export type AlphaDslEntry = { label: string; insertText: string; detail: string; inline?: boolean }

const FUN_STMT = [
  'Revenue',
  'Net_Income',
  'EPS_Diluted',
  'Operating_Cash_Flow',
  'Free_Cash_Flow',
  'Total_Assets',
  'Total_Equity',
  'Total_Debt',
  'Current_Ratio',
  'Gross_Margin',
  'Operating_Margin',
  'Return_On_Assets',
  'Return_On_Equity',
  'Debt_To_Equity',
  'Free_Cash_Flow_Yield',
  'Price_To_Earnings',
] as const
const FUN_DAILY = [
  'Market_Cap',
  'Enterprise_Value',
  'Trailing_PE',
  'Forward_PE',
  'PEG_Ratio',
  'Price_To_Book',
  'Dividend_Yield',
  'Beta',
  'Shares_Outstanding',
] as const

const funEntries: AlphaDslEntry[] = [
  ...FUN_STMT.map((fn) => ({
    label: `fun.${fn}`,
    insertText: `fun.${fn}`,
    detail: `AlphaSeries (statement); ctx.feature('${fn}')`,
    inline: true,
  })),
  ...FUN_DAILY.map((fn) => ({
    label: `fun.${fn}`,
    insertText: `fun.${fn}`,
    detail: 'AlphaSeries (daily)',
    inline: true,
  })),
]

/** Entries safe for grey inline suffix (no tab-stop snippets). */
export const ALPHA_DSL_INLINE_ENTRIES: readonly AlphaDslEntry[] = [
  // Bare ctx/ts/cs/fun/jnp roots omitted — use ctx.*, ts.*, cs.*, fun.*, jnp.* (body editor injects modules).
  { label: 'ctx.open', insertText: 'ctx.open', detail: 'AlphaSeries OHLCV', inline: true },
  { label: 'ctx.high', insertText: 'ctx.high', detail: 'AlphaSeries', inline: true },
  { label: 'ctx.low', insertText: 'ctx.low', detail: 'AlphaSeries', inline: true },
  { label: 'ctx.close', insertText: 'ctx.close', detail: 'AlphaSeries', inline: true },
  { label: 'ctx.adj_volume', insertText: 'ctx.adj_volume', detail: 'AlphaSeries', inline: true },
  { label: 'ctx.n_tickers', insertText: 'ctx.n_tickers', detail: 'int', inline: true },
  { label: 'ctx.feature_names', insertText: 'ctx.feature_names', detail: 'tuple', inline: true },
  { label: 'ctx.feature', insertText: 'ctx.feature("name")', detail: 'named feature', inline: true },
  { label: 'ts.delay', insertText: 'ts.delay(x, lag)', detail: 'TSDelay', inline: true },
  { label: 'ts.delta', insertText: 'ts.delta(x, lag)', detail: 'delta', inline: true },
  { label: 'ts.sum', insertText: 'ts.sum(x, window)', detail: 'rolling sum', inline: true },
  { label: 'ts.mean', insertText: 'ts.mean(x, window)', detail: 'rolling mean', inline: true },
  { label: 'ts.std', insertText: 'ts.std(x, window)', detail: 'rolling std', inline: true },
  { label: 'ts.zscore', insertText: 'ts.zscore(x, window)', detail: 'rolling z', inline: true },
  { label: 'ts.rank', insertText: 'ts.rank(x, window)', detail: 'rolling rank', inline: true },
  {
    label: 'ts.regression',
    insertText: 'ts.regression(y, x, window, lag, retval)',
    detail: 'TS regression',
    inline: true,
  },
  { label: 'ts.humpdecay', insertText: 'ts.humpdecay(x, hump)', detail: 'hump decay', inline: true },
  { label: 'cs.rank', insertText: 'cs.rank(x)', detail: 'CS rank', inline: true },
  { label: 'cs.zscore', insertText: 'cs.zscore(x)', detail: 'CS z', inline: true },
  { label: 'cs.scale', insertText: 'cs.scale(x, target)', detail: 'scale', inline: true },
  { label: 'cs.sign', insertText: 'cs.sign(x)', detail: 'sign', inline: true },
  { label: 'cs.winsorize', insertText: 'cs.winsorize(x, tail)', detail: 'winsorize', inline: true },
  { label: 'cs.neutralize_market', insertText: 'cs.neutralize_market(x)', detail: 'neutral', inline: true },
  {
    label: 'cs.neutralize_groups',
    insertText: 'cs.neutralize_groups(x, group_ids)',
    detail: 'group neutral',
    inline: true,
  },
  ...funEntries,
  { label: 'jnp.array', insertText: 'jnp.array(a)', detail: 'array', inline: true },
  { label: 'jnp.zeros', insertText: 'jnp.zeros(shape)', detail: 'zeros', inline: true },
  { label: 'jnp.ones', insertText: 'jnp.ones(shape)', detail: 'ones', inline: true },
  { label: 'jnp.sqrt', insertText: 'jnp.sqrt(x)', detail: 'sqrt', inline: true },
  { label: 'jnp.log', insertText: 'jnp.log(x)', detail: 'log', inline: true },
  { label: 'jnp.exp', insertText: 'jnp.exp(x)', detail: 'exp', inline: true },
  { label: 'jnp.where', insertText: 'jnp.where(cond, x, y)', detail: 'where', inline: true },
  { label: 'jnp.abs', insertText: 'jnp.abs(x)', detail: 'abs', inline: true },
  { label: 'jnp.mean', insertText: 'jnp.mean(x)', detail: 'mean', inline: true },
  { label: 'jnp.std', insertText: 'jnp.std(x)', detail: 'std', inline: true },
]
