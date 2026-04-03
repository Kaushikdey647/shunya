# alpaca-2

Small Python stack for **multi-ticker equity panels**, **JAX alpha functions** (WorldQuant BRAIN–style processing), and **backtrader** backtests. Data comes from `yfinance`; features include OHLCV plus technicals from `finta`.

## Layout

| Package | Role |
|--------|------|
| `src.data` | `finTs` — download OHLCV, build MultiIndex `(Ticker, Date)` frame, attach engineered columns |
| `src.algorithm` | `FinStrat` (alpha pipeline), `FinBT` (backtrader wrapper), `cross_section` (rank, zscore, winsorize, neutralization) |
| `src.utils` | `indicators` — column namespaces (`COL`, `IX`, `IX_LIVE`), strategy feature lists, helpers |

Public re-exports from `src`:

```python
from src import FinBT, FinStrat, cross_section, finTs, indicators
```

## Core ideas

1. **`finTs`** loads one or many tickers and produces a dataframe whose live columns include raw **Open / High / Low / Close / Volume** first, then indicators (`SMA_50`, `RSI_14`, …). See `indicators.STRATEGY_FEATURES_LIVE` for the full default ordering.

2. **`FinStrat(fin_ts, algorithm, ...)`** binds the panel to a JAX callable `algorithm(panel) -> (n_stocks,)`. Optional BRAIN-like knobs:
   - `decay` (per-name EMA on raw scores — pass `tickers=` into `pass_`)
   - `truncation` (cross-sectional winsorize)
   - `neutralization`: `"market"`, `"none"`, or `"group"` (with `group_ids`)
   - `max_single_weight`, `jit_algorithm`
   - **`panel_columns`** — restrict which columns `panel_at` loads (e.g. `indicators.STRATEGY_PANEL_OHLCV_ONLY`) so OHLC-only alphas trade from the first bar instead of waiting on `SMA_200` warm-up

3. **`FinBT(fin_strat, fin_ts, ...)`** runs the same `FinStrat` on the same `fin_ts` instance in backtrader, rebalancing to `pass_` dollar targets each bar. Call `reset_pipeline_state` is handled at `run()` for decay state.

4. **`cross_section`** — JIT-friendly helpers: `rank`, `zscore`, `winsorize`, `neutralize_market`, `neutralize_groups`. `rank(x)` is increasing in `x` (smallest → ~0, largest → ~1); use `rank(-x)` to flip.

## Notebook

`test.ipynb` walks through `finTs` → `FinStrat` (CLV × volume style alpha) → `FinBT`.

## Requirements

- Python **≥ 3.12**
- Main libraries: `jax`, `pandas`, `yfinance`, `backtrader`, `finta`, `matplotlib`, … (see `pyproject.toml`)

Install (e.g. with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

## Note on live data

`indicators` defines a lookahead column `Future_1d_Ret`; for real-time style panels use `panel_at(..., live=True)` and `IX_LIVE` / `STRATEGY_FEATURES_LIVE` so that column is excluded.
