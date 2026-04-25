# shunya

[![PyPI](https://img.shields.io/pypi/v/shunya-py.svg)](https://pypi.org/project/shunya-py/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/package_manager-uv-6f42c1.svg)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![Data](https://img.shields.io/badge/data-yfinance-informational.svg)](https://pypi.org/project/yfinance/)
[![Broker](https://img.shields.io/badge/broker-alpaca--py-orange.svg)](https://github.com/alpacahq/alpaca-py)

Small Python stack for **multi-ticker equity panels**, **JAX alpha functions** (WorldQuant BRAIN-style processing), **backtrader** backtests, and an early **tick-to-trade streaming foundation**. Historical data is provider-driven (`yfinance` by default, optional Alpaca bars); features include OHLCV plus technicals from `finta`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for architecture details, extension patterns, and coding guidelines.

Optional **local TimescaleDB** for canonical OHLCV and fundamentals is documented in [`docs/data_timescale.md`](docs/data_timescale.md) (`pip install 'shunya-py[timescale]'`, `docker compose`, migration and ingest CLIs).

## Layout

| Package | Role |
|--------|------|
| `shunya.data` | `finTs` — download OHLCV, build MultiIndex `(Ticker, Date)` frame, attach engineered columns |
| `shunya.algorithm` | `FinStrat` (context-based alpha pipeline), `FinBT` (backtrader), `FinTrade` (Alpaca live/paper orders), `cross_section` (rank, zscore, winsorize, neutralization) |
| `shunya.streaming` | Event/tick plumbing for Alpaca-style streaming: normalized market events, per-symbol FIFO buffers, micro-bar aggregation, universe/subscription helpers, and rectangular snapshots for alpha evaluation |
| `shunya.utils` | `indicators` — column namespaces (`COL`, `IX`, `IX_LIVE`), strategy feature lists, helpers |

Common imports from `shunya` (illustrative):

```python
from shunya import (
    DecisionContext,
    FinBT,
    FinStrat,
    FinTrade,
    cross_section,
    finTs,
    indicators,
)
```

The canonical set of symbols re-exported at the package root is `__all__` in [`shunya/__init__.py`](shunya/__init__.py) (for example `PanelQADiagnostics`, `YFinanceMarketDataProvider`, `StreamingRunner`, `OrderManager`, target helpers, `logical`, `time_series`, and timestamp helpers).

## Core ideas

1. **`finTs`** loads one or many tickers and produces a dataframe whose live columns include raw **Open / High / Low / Close / Volume** first, then indicators (`VWAP`, `SMA_50`, `RSI_14`, …). It also attaches best-effort yfinance classification columns: **`Sector`**, **`Industry`**, **`SubIndustry`** (with deterministic `Unknown*` fallbacks). See `indicators.STRATEGY_FEATURES_LIVE` for the full default ordering.

2. **`FinStrat(fin_ts, algorithm, ...)`** binds the panel to a context callable `algorithm(ctx) -> (n_stocks,)`, where `ctx` exposes:
   - base series: `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.adj_volume`
   - time-series operators: `ctx.ts.*` (for example, `ctx.ts.mean(ctx.close, 50)`)
   - cross-sectional operators: `ctx.cs.*` (for example, `ctx.cs.rank(signal)`)

   Optional BRAIN-like knobs:
   - `decay` (per-name EMA on raw scores — pass `tickers=` into `pass_`)
   - `truncation` (cross-sectional winsorize)
   - `neutralization`: `"market"`, `"none"`, or `"group"` (with `group_ids`, often from `Sector`/`Industry`/`SubIndustry`)
   - `max_single_weight`

3. **`FinBT(fin_strat, fin_ts, ...)`** runs the same `FinStrat` on the same `fin_ts` instance in backtrader, rebalancing to `pass_` dollar targets each bar. `run()` resets `FinStrat` decay state. Pass **`commission`** (broker rate) and optional **`slippage_pct`** (adverse percent via backtrader’s `set_slippage_perc`). For `neutralization="group"`, `group_column` defaults to `"Sector"` (or set `"Industry"` / `"SubIndustry"`). `broker_deltas` / `target_usd_universe` in `shunya.algorithm.targets` mirror how live orders diff targets vs positions.

4. **`FinTrade(fin_strat, ...)`** uses the Alpaca Trading API (`alpaca-py`): `run(tradecapital, fin_ts)` builds the panel (latest date by default), runs `pass_`, diffs targets vs open positions, and submits **market DAY** orders by **notional**. Set `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, or pass `api_key` / `secret_key`. `dry_run=True` still fetches positions but does not submit. Temporal `decay` state is **not** reset between `run` calls (unlike `FinBT.run`). For `neutralization="group"`, `group_column` defaults to `"Sector"` (or set `"Industry"` / `"SubIndustry"`). Optional controls include sector gross/net caps, turnover budget, ADV participation caps, and post-submit reconciliation policies. The return value is an `ExecutionReport` dataclass; use `ExecutionReport.as_dict()` if you need a JSON-serializable summary.

5. **Streaming tick-to-trade foundation** lives beside the batch panel flow:
   - `shunya.streaming` provides `MarketEvent`, `SymbolRingBuffer`, `StreamingState`, `MicroBarAggregator`, `SnapshotBuilder`, `SubscriptionManager`, `UniverseSelector`, and `AlpacaStreamClient`.
   - `StreamingRunner` + `StreamingContextBuilder` let you materialize streaming micro-bars into the same `(time, n_tickers)` alpha context shape used by `FinStrat`.
   - `OrderManager` is a stateful OMS helper for streaming target updates: it caches positions/open orders, suppresses duplicate submits while orders are working, and can route either through `submit_delta_orders` or explicit `OrderSpec` generation via `RiskPolicy`.

6. **`DecisionContext`** (`shunya.algorithm.decision`) pins **signal time** and **data provenance** (`yfinance_research` vs `alpaca_bars`) so live logic does not silently mix “Yahoo’s last bar” with “submit now.”  Pass `decision=DecisionContext(as_of=..., data_source=...)` into `FinTrade.run`, or set `as_of=` explicitly; otherwise the last date in the panel index is used.

7. **`MarketDataProvider`** (`shunya.data.providers`) abstracts history loading: default `YFinanceMarketDataProvider` in `finTs`, optional `AlpacaHistoricalMarketDataProvider` for broker-aligned panels and parity checks vs Yahoo.
   - Provider output contract is consistent: `DatetimeIndex` named `Date`, normalized to daily granularity.
   - `AlpacaHistoricalMarketDataProvider` is strict: if requested symbols are missing bars, it raises a `ValueError` listing those symbols.

8. **`cross_section`** — JIT-friendly helpers: `rank`, `zscore`, `scale`, `sign`, `winsorize`, `neutralize_market`, `neutralize_groups`. `rank(x)` is increasing in `x` (smallest → ~0, largest → ~1); use `rank(-x)` to flip.

## Trading-time axis (minute/hour/day)

- `finTs(..., trading_axis_mode="observed")` keeps legacy behavior (calendar derived from observed panel rows).
- `finTs(..., trading_axis_mode="canonical")` builds a canonical US-equities trading calendar for the selected `BarSpec` (weekend gaps removed from bar progression).
- `strict_trading_grid=True` enforces provider timestamps lie on the canonical bar grid and have no in-session holes.
- `FinStrat(..., temporal_mode="bar_step")` advances decay one step per bar.
- `FinStrat(..., temporal_mode="elapsed_trading_time")` advances decay by trading-time distance; `FinBT` and `FinTrade` pass execution timestamps so this mode works out of the box.

## Operator helpers

- `shunya.algorithm.logical`
  - `trade_when(condition, alpha, otherwise, exit_condition=...)`
  - `if_else`, `logical_and`, `logical_or`, `logical_not`
- `shunya.algorithm.time_series`
  - `tsdelta`, `tsdelay`, `tssum`, `tsmean`, `tsrank`, `tszscore`, `tsstddev`
  - `tsregression(y, x, window, lag, retval)` with `retval in {"error", "a", "b", "estimate"}`
  - `humpdecay`
- `shunya.algorithm.group_ops`
  - `group_rank`, `group_zscore`, `group_mean`, `group_neutralize`

```python
import jax.numpy as jnp
from shunya.algorithm import cross_section, group_ops, logical, time_series

signal = cross_section.zscore(jnp.array([1.0, 2.0, 3.0]))
gated = logical.trade_when(signal > 0, signal, 0.0)
```

## Quick start

```bash
pip install "shunya-py[dev]"   # library + pytest (for upstream / CI-style checks)

# From a clone (installs the local project; add --extra notebook for Jupyter notebooks):
uv sync --extra dev
uv run pytest
```

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
    neutralization="group",   # defaults to Sector in FinBT/FinTrade when applicable
    truncation=0.02,
)

bt = FinBT(fs, fts, cash=100_000.0, commission=0.0005, slippage_pct=0.0005).run()
results = bt.results(show=False)
print(results["metrics"])
```

### Using Alpaca historical bars in `finTs`

```python
from shunya import (
    AlpacaHistoricalMarketDataProvider,
    DecisionContext,
    FinTrade,
    FinStrat,
    finTs,
)

provider = AlpacaHistoricalMarketDataProvider(
    api_key="YOUR_KEY",
    secret_key="YOUR_SECRET",
    paper=True,
)
fts = finTs(
    "2024-01-01",
    "2024-03-01",
    ["AAPL", "MSFT"],
    market_data=provider,
    attach_yfinance_classifications=False,
)

# Use alpaca_bars provenance for tighter data/execution parity checks.
decision = DecisionContext(data_source="alpaca_bars")
```

### Streaming tick-to-trade foundation

The streaming path is intentionally separate from `finTs` / `FinTrade` batch orchestration.
It reuses `FinStrat` math and the broker execution adapters, but drives them from
streaming market events instead of a prebuilt panel.

```python
from shunya import FinStrat
from shunya.algorithm import OrderManager, StreamingRunner
from shunya.streaming import trade_event

runner = StreamingRunner(
    fin_strat=fs,
    order_manager=OrderManager(execution_adapter=adapter, min_order_notional=1.0),
    lookback=32,
    bar_interval="1s",
    max_concurrent_symbols=16,
)
runner.set_target_universe(["AAPL", "MSFT", "NVDA"])

runner.on_event(trade_event("AAPL", "2024-01-01 09:30:00.100", 190.25), capital=100_000.0)
runner.on_event(trade_event("MSFT", "2024-01-01 09:30:00.200", 402.10), capital=100_000.0)
decision = runner.on_event(
    trade_event("AAPL", "2024-01-01 09:30:01.100", 190.40),
    capital=100_000.0,
    dry_run=True,
)
```

The main streaming building blocks are:

- `AlpacaStreamClient`: lazy wrapper around Alpaca stock live data subscriptions
- `MarketEvent`: normalized trade/quote event shape
- `SymbolRingBuffer` / `StreamingState`: per-symbol FIFO event storage
- `MicroBarAggregator`: converts asynchronous events into short rolling bars
- `SnapshotBuilder`: builds rectangular `(time, n_tickers)` snapshots
- `StreamingContextBuilder`: turns streaming snapshots into regular `AlphaContext`
- `StreamingRunner`: computes targets from streaming snapshots with `FinStrat`
- `OrderManager`: stateful OMS helper for duplicate suppression and in-flight awareness

## Notebooks

- [`vwap_close_rank_backtest_yfinance.ipynb`](vwap_close_rank_backtest_yfinance.ipynb) — `finTs` (default `YFinanceMarketDataProvider`) → `FinStrat` (`rank(VWAP/Close)`) → `FinBT`.
- [`vwap_close_rank_backtest.ipynb`](vwap_close_rank_backtest.ipynb) — same alpha and `FinBT` flow with **`AlpacaHistoricalMarketDataProvider`** (requires `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`).

## Requirements

- Python **≥ 3.12**
- Main libraries: `jax`, `pandas`, `yfinance`, `backtrader`, `finta`, `matplotlib`, … (see `pyproject.toml`)

Install from [PyPI](https://pypi.org/project/shunya-py/) (import the **`shunya`** package):

```bash
pip install shunya-py
# optional: Jupyter kernel for the bundled notebooks
pip install "shunya-py[notebook]"
```

Install from a clone (e.g. with [uv](https://docs.astral.sh/uv/)):

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

`indicators` defines a lookahead column `Future_1d_Ret`. `FinStrat` context execution uses live OHLCV history and excludes lookahead fields by design.

## Risks: Yahoo vs Alpaca, margin, paper vs live

- **yfinance vs Alpaca:** Yahoo-adjusted history, time zones, and corporate-action handling can differ from Alpaca’s tape. Treat research PnL on Yahoo-only panels as indicative; for execution alignment prefer `AlpacaHistoricalMarketDataProvider` and `DecisionContext(data_source="alpaca_bars")`, or reconcile closes explicitly before trusting live notionals.
- **Shorting and margin:** Negative target notionals imply shorts. Alpaca requires margin, borrow availability, and `shortable` assets; orders can reject if the account is cash-only or the name is not shortable. The execution layer warns on non-shortable names but does not guarantee borrow.
- **Paper vs live checklist:** Confirm `paper=True`/`False` on `TradingClient`, that keys are scoped and never committed to git, use `dry_run=True` for rehearsal, set `require_market_open=True` when you must avoid after-hours submits, and read `ExecutionReport.warnings` (buying-power cap, Yahoo parity note, etc.) on every run.

## What Is Not There Yet

- The streaming layer is **Alpaca-only** today. There is no `KiteTicker` / Zerodha websocket client in `shunya.streaming`.
- `StreamingRunner` reuses `FinStrat` processing and broker adapters, but it does **not** inherit the full `FinTrade.run(...)` contract automatically. Decision-time guards, `DecisionContext`, reconciliation policy handling, sector/net/turnover/ADV controls, and post-submit remediation remain documented on the batch `FinTrade` path.
- There is no built-in daemon/service runner, persisted event log, or replay CLI for streaming sessions yet.
- The streaming docs currently live in this README only; `CONTRIBUTING.md` and the alpha examples are still centered on the historical panel / batch execution path.

## Development tests

```bash
uv sync --all-extras
uv run pytest
```

## Publishing (maintainers)

Build with `uv build` (wheel and sdist). Upload with [Twine](https://twine.readthedocs.io/) or [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) from CI. Do not commit API tokens or `.pypirc`.

## Roadmap status

- P0 completed: yfinance classifications, group defaults, order-status observation, sector gross cap.
- P1 completed: decision/session guards, panel QA diagnostics, richer backtest diagnostics.
- P2 completed: reconciliation loop + remediation hooks, net/turnover/ADV constraints, integration tests.
- P3 in progress: tick-to-trade foundation (`shunya.streaming`, `StreamingRunner`, `OrderManager`, open-order adapter snapshots).

## Documentation

- Main usage and behavior: [`README.md`](README.md)
- Contributor and architecture guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
