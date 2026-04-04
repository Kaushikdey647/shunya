# alpaca-2

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/package_manager-uv-6f42c1.svg)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![Data](https://img.shields.io/badge/data-yfinance-informational.svg)](https://pypi.org/project/yfinance/)
[![Broker](https://img.shields.io/badge/broker-alpaca--py-orange.svg)](https://github.com/alpacahq/alpaca-py)

Small Python stack for **multi-ticker equity panels**, **JAX alpha functions** (WorldQuant BRAIN–style processing), and **backtrader** backtests. Data comes from `yfinance`; features include OHLCV plus technicals from `finta`.

See `CONTRIBUTION.md` for architecture details, extension patterns, and coding guidelines.

## Layout

| Package | Role |
|--------|------|
| `src.data` | `finTs` — download OHLCV, build MultiIndex `(Ticker, Date)` frame, attach engineered columns |
| `src.algorithm` | `FinStrat` (alpha pipeline), `FinBT` (backtrader), `FinTrade` (Alpaca live/paper orders), `cross_section` (rank, zscore, winsorize, neutralization) |
| `src.utils` | `indicators` — column namespaces (`COL`, `IX`, `IX_LIVE`), strategy feature lists, helpers |

Public re-exports from `src`:

```python
from src import (
    DecisionContext,
    FinBT,
    FinStrat,
    FinTrade,
    cross_section,
    finTs,
    indicators,
)
```

## Core ideas

1. **`finTs`** loads one or many tickers and produces a dataframe whose live columns include raw **Open / High / Low / Close / Volume** first, then indicators (`VWAP`, `SMA_50`, `RSI_14`, …). It also attaches best-effort yfinance classification columns: **`Sector`**, **`Industry`**, **`SubIndustry`** (with deterministic `Unknown*` fallbacks). See `indicators.STRATEGY_FEATURES_LIVE` for the full default ordering.

2. **`FinStrat(fin_ts, algorithm, ...)`** binds the panel to a JAX callable `algorithm(panel) -> (n_stocks,)`. Optional BRAIN-like knobs:
   - `decay` (per-name EMA on raw scores — pass `tickers=` into `pass_`)
   - `truncation` (cross-sectional winsorize)
   - `neutralization`: `"market"`, `"none"`, or `"group"` (with `group_ids`, often from `Sector`/`Industry`/`SubIndustry`)
   - `max_single_weight`, `jit_algorithm`
   - **`panel_columns`** — restrict which columns `panel_at` loads (e.g. `indicators.STRATEGY_PANEL_OHLCV_ONLY`) so OHLC-only alphas trade from the first bar instead of waiting on `SMA_200` warm-up

3. **`FinBT(fin_strat, fin_ts, ...)`** runs the same `FinStrat` on the same `fin_ts` instance in backtrader, rebalancing to `pass_` dollar targets each bar. `run()` resets `FinStrat` decay state. Pass **`commission`** (broker rate) and optional **`slippage_pct`** (adverse percent via backtrader’s `set_slippage_perc`). For `neutralization="group"`, `group_column` defaults to `"Sector"` (or set `"Industry"` / `"SubIndustry"`). `broker_deltas` / `target_usd_universe` in `src.algorithm.targets` mirror how live orders diff targets vs positions.

4. **`FinTrade(fin_strat, ...)`** uses the Alpaca Trading API (`alpaca-py`): `run(tradecapital, fin_ts)` builds the panel (latest date by default), runs `pass_`, diffs targets vs open positions, and submits **market DAY** orders by **notional**. Set `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, or pass `api_key` / `secret_key`. `dry_run=True` still fetches positions but does not submit. Temporal `decay` state is **not** reset between `run` calls (unlike `FinBT.run`). For `neutralization="group"`, `group_column` defaults to `"Sector"` (or set `"Industry"` / `"SubIndustry"`). Optional controls include sector gross/net caps, turnover budget, ADV participation caps, and post-submit reconciliation policies. The return value is an `ExecutionReport` dataclass; use `ExecutionReport.as_dict()` if you need a JSON-serializable summary.

5. **`DecisionContext`** (`src.algorithm.decision`) pins **signal time** and **data provenance** (`yfinance_research` vs `alpaca_bars`) so live logic does not silently mix “Yahoo’s last bar” with “submit now.”  Pass `decision=DecisionContext(as_of=..., data_source=...)` into `FinTrade.run`, or set `as_of=` explicitly; otherwise the last date in the panel index is used.

6. **`MarketDataProvider`** (`src.data.providers`) abstracts history loading: default `YFinanceMarketDataProvider` in `finTs`, optional `AlpacaHistoricalMarketDataProvider` for broker-aligned panels and parity checks vs Yahoo.

7. **`cross_section`** — JIT-friendly helpers: `rank`, `zscore`, `scale`, `sign`, `winsorize`, `neutralize_market`, `neutralize_groups`. `rank(x)` is increasing in `x` (smallest → ~0, largest → ~1); use `rank(-x)` to flip.

## Operator helpers

- `src.algorithm.logical`
  - `trade_when(condition, alpha, otherwise, exit_condition=...)`
  - `if_else`, `logical_and`, `logical_or`, `logical_not`
- `src.algorithm.time_series`
  - `tsdelta`, `tsdelay`, `tssum`, `tsmean`, `tsrank`, `tszscore`, `tsstddev`
  - `tsregression(y, x, window, lag, retval)` with `retval in {"error", "a", "b", "estimate"}`
  - `humpdecay`
- `src.algorithm.group_ops`
  - `group_rank`, `group_zscore`, `group_mean`, `group_neutralize`

```python
import jax.numpy as jnp
from src.algorithm import cross_section, group_ops, logical, time_series

signal = cross_section.zscore(jnp.array([1.0, 2.0, 3.0]))
gated = logical.trade_when(signal > 0, signal, 0.0)
```

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

```python
import jax.numpy as jnp
from src import FinBT, FinStrat, finTs
from src.utils import indicators as ind

fts = finTs("2023-01-01", "2024-01-01", ["AAPL", "MSFT", "NVDA"])

def alpha(panel: jnp.ndarray) -> jnp.ndarray:
    close = panel[:, ind.IX_LIVE.CLOSE]
    vol = panel[:, ind.IX_LIVE.VOLUME]
    return jnp.log1p(vol) * close

fs = FinStrat(
    fts,
    alpha,
    neutralization="group",   # defaults to Sector in FinBT/FinTrade when applicable
    truncation=0.02,
)

bt = FinBT(fs, fts, cash=100_000.0, commission=0.0005, slippage_pct=0.0005).run()
results = bt.results(show=False)
print(results["metrics"])
```

## Notebook

`test.ipynb` walks through `finTs` → `FinStrat` (CLV × volume style alpha) → `FinBT`.

## Requirements

- Python **≥ 3.12**
- Main libraries: `jax`, `pandas`, `yfinance`, `backtrader`, `finta`, `matplotlib`, … (see `pyproject.toml`)

Install (e.g. with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

## Classification and sector controls

- `finTs` attaches `Sector`, `Industry`, `SubIndustry` with deterministic fallbacks:
  - `UnknownSector`, `UnknownIndustry`, `UnknownSubIndustry`
- For group-neutral alphas:
  - set `FinStrat(..., neutralization="group")`
  - `FinBT(..., group_column=...)` and `FinTrade(..., group_column=...)` default to `Sector`
- For pre-trade concentration controls:
  - `FinTrade.run(..., sector_gross_cap_fraction=0.30, sector_cap_mode="rescale")`
  - `FinBT(..., sector_gross_cap_fraction=0.30, sector_cap_mode="rescale")`
- For net/turnover/participation controls:
  - `group_net_cap_fraction=...` (fraction of portfolio gross)
  - `turnover_budget_fraction=...` (fraction of target gross turnover)
  - `adv_participation_fraction=...` (caps order deltas by ADV participation)

## Decision-time guards and reconciliation

- Decision-time validation (`FinTrade.run`) now supports:
  - `decision_enforce_weekday=True`
  - `decision_strict_same_session=False`
  - `decision_max_staleness_days=7`
- Execution reconciliation supports:
  - `reconcile_after_submit=True`
  - `reconciliation_policy in {"warn_only", "retry_once", "cancel_and_retarget"}`
  - `reconciliation_tolerance_notional=...`

`ExecutionReport` includes:

- submit attempts + optional remediation attempts
- status observation fields (initial/final status, fill info)
- post-trade current exposures and residual deltas

## Note on live data

`indicators` defines a lookahead column `Future_1d_Ret`; for real-time style panels use `panel_at(..., live=True)` and `IX_LIVE` / `STRATEGY_FEATURES_LIVE` so that column is excluded.

## Risks: Yahoo vs Alpaca, margin, paper vs live

- **yfinance vs Alpaca:** Yahoo-adjusted history, time zones, and corporate-action handling can differ from Alpaca’s tape. Treat research PnL on Yahoo-only panels as indicative; for execution alignment prefer `AlpacaHistoricalMarketDataProvider` and `DecisionContext(data_source="alpaca_bars")`, or reconcile closes explicitly before trusting live notionals.
- **Shorting and margin:** Negative target notionals imply shorts. Alpaca requires margin, borrow availability, and `shortable` assets; orders can reject if the account is cash-only or the name is not shortable. The execution layer warns on non-shortable names but does not guarantee borrow.
- **Paper vs live checklist:** Confirm `paper=True`/`False` on `TradingClient`, that keys are scoped and never committed to git, use `dry_run=True` for rehearsal, set `require_market_open=True` when you must avoid after-hours submits, and read `ExecutionReport.warnings` (buying-power cap, Yahoo parity note, etc.) on every run.

## Development tests

```bash
uv sync --extra dev
uv run pytest
```

## Roadmap status

- P0 completed: yfinance classifications, group defaults, order-status observation, sector gross cap.
- P1 completed: decision/session guards, panel QA diagnostics, richer backtest diagnostics.
- P2 completed: reconciliation loop + remediation hooks, net/turnover/ADV constraints, integration tests.

## Documentation

- Main usage and behavior: `README.md`
- Contributor and architecture guide: `CONTRIBUTION.md`
