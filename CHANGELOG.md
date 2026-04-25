# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Tick-to-trade streaming foundation under `shunya.streaming`:
  - `MarketEvent`, `trade_event(...)`, `quote_event(...)`
  - `SymbolRingBuffer` and `StreamingState` for per-symbol FIFO event storage
  - `MicroBarAggregator`, `SnapshotBuilder`, and `StreamingSnapshot` for asynchronous event aggregation into rectangular alpha inputs
  - `SubscriptionManager` and `UniverseSelector` for bounded active-universe handling
  - `StreamingMetrics` for lightweight in-memory telemetry
  - `AlpacaStreamClient` plus Alpaca trade/quote normalizers
- New streaming orchestration / OMS types in `shunya.algorithm`:
  - `StreamingContextBuilder`
  - `StreamingRunner` and `StreamingDecision`
  - `OrderManager` and `ManagedOrderBatch`
- Broker-neutral open-order snapshot surface:
  - `OpenOrderView`
  - `ExecutionAdapter.list_open_orders()`
  - `AlpacaExecutionAdapter.list_open_orders()`
  - `KiteExecutionAdapter.list_open_orders()`
- Context-based alpha authoring API in `shunya.algorithm.alpha_context`:
  - `AlphaContext` with canonical series fields (`open`, `high`, `low`, `close`, `adj_volume`)
  - `AlphaSeries` wrapper for history tensors
  - namespaced operators via `ctx.ts.*` and `ctx.cs.*`
- New example alpha package under `examples/alphas`:
  - `sma_ratio_50`
  - `mean_reversion_5`
  - `breakout_20`
- New research notebook `examples/alpha_benchmark_oex.ipynb` to compare example alpha returns/correlations/metrics against `^OEX` and against each other.
- `BarIndexPolicy` and `default_bar_index_policy()`: market data indices default to **America/New_York** (session-stable intraday labels across US DST), with optional `timezone`, `naive`, and `daily_anchor` (`"timezone"` vs legacy `"utc"` for daily-like bars).
- `finTs(..., bar_index_policy=...)`, `finTs.bar_index_policy`, and `MarketDataProvider.download(..., bar_index_policy=...)`.
- `bounds_for_validation()` for timezone-consistent `start_date` / `end_date` windows.
- Trading-time axis primitives in `shunya.data.timeframes`:
  - `build_trading_calendar(...)` for canonical US-equities bar grids across minute/hour/day cadences.
  - `timestamp_is_on_trading_grid(...)` and `trading_time_distance(...)` for grid membership and trading-time deltas.
- **BRAIN-style pipeline parity (math + data hygiene):**
  - `finTs.align_universe(...)` — intersect trading calendars, drop or raise on incomplete tickers, dense `(Ticker, Date)` reindex; returns `PanelAlignReport`; sets `_aligned_calendar`.
  - `finTs.get_trading_calendar()` / `finTs.execution_lag_calendar_date(..., lag=)` — calendar-aware **Delay1**-style lag.
  - `PanelAlignReport` (exported from `shunya` / `shunya.data`).
  - `FinStrat`: `decay_mode` (`"ema"` or `"linear"`), `decay_window`, `signal_delay`, `nan_policy`, `panel_date_for_execution()`, linear decay on raw scores.
  - `FinBT`: OHLCV feeds reindexed to a shared calendar; `validate_finite_targets` (default on).
- `cross_section`: finite-safe `neutralize_market` / `neutralize_groups` / `winsorize`; `rank` sorts non-finite last.
- Notebook examples: `vwap_close_rank_backtest_yfinance.ipynb` and `vwap_close_rank_backtest.ipynb` call `align_universe` after `finTs(...)`.
- yfinance-based classification mapping for `Sector`, `Industry`, and `SubIndustry` with deterministic fallback labels.
- New `finTs` controls:
  - `classifications=...`
  - `attach_yfinance_classifications=True`
- Group defaults and validation improvements for neutralization paths in backtest/trading flows.
- Paper-safe execution status observation fields in `OrderAttempt` / `ExecutionReport`:
  - initial/final status
  - fill quantity and average fill price
  - status polling errors
- Optional sector gross cap enforcement in shared target helpers and integration in `FinBT`/`FinTrade`.
- Session-aware decision-time guardrails:
  - weekend and future-date checks
  - strict same-session option
  - staleness warnings
- Data QA diagnostics in `finTs`:
  - duplicate row detection
  - missing ticker/date coverage checks
  - stale panel checks
  - invalid OHLCV row checks
- New `finTs` controls:
  - `trading_axis_mode` (`"observed"` or `"canonical"`) for calendar/lag helpers.
  - `strict_trading_grid` for off-grid/holey timestamp validation when strict loading is enabled.
- Richer backtest analytics:
  - turnover history and summary metrics
  - concentration metrics
  - group exposure snapshots
- Reconciliation loop and remediation hooks in live/paper trading:
  - `warn_only`
  - `retry_once`
  - `cancel_and_retarget`
- Additional shared constraints:
  - group net caps
  - turnover budget enforcement
  - ADV participation caps
- New documentation:
  - `CONTRIBUTING.md` (contributor guide; previously `CONTRIBUTION.md`)
  - expanded `README.md` sections for controls, diagnostics, and roadmap status

### Changed

- `FinStrat` now exposes reusable seams for non-`finTs` runners:
  - `scores_from_context(ctx)`
  - `process_raw_scores(raw_scores, capital, ...)`
- `README.md` now documents the streaming foundation and explicitly calls out what is not implemented yet.
- **Breaking:** `FinStrat` now executes context-style alpha callables (`algorithm(ctx)`), replacing legacy panel-index authoring (`algorithm(panel)` / `IX_*` flow).
- `FinBT` / `FinTrade` now route signal execution through context-based alpha evaluation while preserving downstream sizing/neutralization controls.
- **Breaking:** `normalize_history_index` and bundled providers align Yahoo/Alpaca timestamps to `BarIndexPolicy` (default **America/New_York**), not forced UTC-naive. Use `BarIndexPolicy(timezone="UTC")` and `daily_anchor="utc"` to recover older daily-like alignment.
- `validate_core_ohlcv_coverage(..., bar_index_policy=...)` interprets coverage windows in the policy timezone for intraday and daily-like bars.
- **Breaking:** The installable Python package is published on PyPI as **`shunya-py`** (the name `shunya` was already registered by another project). Import the library as **`shunya`** (`from shunya import finTs`, etc.), not `src`.
- `FinStrat.__init__` includes `decay_mode`, `decay_window`, `signal_delay`, `nan_policy`; temporal smoothing requires `tickers` in `pass_` when EMA or multi-day linear decay is active.
- `FinStrat` adds `temporal_mode` (`"bar_step"` or `"elapsed_trading_time"`). Elapsed mode advances decay by trading-time gaps rather than one-step-per-observed-bar.
- `FinStrat.pass_` accepts optional `execution_date`; `FinBT`/`FinTrade` now pass execution timestamps for trading-time-aware decay.
- `FinStrat.panel_at` / `group_labels_at` respect `signal_delay` (execution date → lagged panel date on `get_trading_calendar()`).
- `FinBT._ohlcv_frames` uses `finTs._aligned_calendar` or the intersection of per-ticker indices.
- `FinTrade.run(...)` interface extended with additional risk, decision-time, and reconciliation knobs.
- `FinBT` / `FinTrade`: richer constraint controls, finite target validation (backtest), enhanced diagnostics output.
- Public exports updated in `shunya/__init__.py` and `shunya/algorithm/__init__.py` (`PanelAlignReport`, helpers, diagnostics types).

### Testing

- Added tests:
  - `tests/test_streaming_pipeline.py`
  - `tests/test_fints_classification.py`
  - `tests/test_data_qa.py`
  - `tests/test_execution_adapter.py`
  - `tests/test_constraints.py`
  - `tests/test_integration_rebalance.py`
  - `tests/test_panel_align.py`
  - `tests/test_brain_pipeline.py`
  - `tests/test_timeframes.py`
  - `tests/test_fints_validation.py`
  - `tests/test_providers.py`
- Expanded tests:
  - `tests/test_decision.py`
  - `tests/test_finbt.py`
  - `tests/test_fintrade.py`
  - `tests/test_finstrat.py`
  - `tests/test_targets.py`
  - `tests/test_execution_adapter.py`
  - `tests/test_kite_execution.py`
- Added trading-time coverage:
  - canonical calendar generation and weekend gap handling
  - strict trading-grid validation on off-grid timestamps
  - elapsed-trading-time decay weighting vs bar-step mode
  - intraday lag parity checks across minute/hour bars
- Added streaming coverage:
  - micro-bar snapshot construction
  - `FinStrat.process_raw_scores(...)` parity with `pass_()`
  - open-order-aware order suppression in `OrderManager`
  - adapter open-order snapshots for Alpaca and Kite
- Current status: full suite passing (`175 passed`).

### Not Yet Implemented

- No Kite/Zerodha websocket streaming client in `shunya.streaming`; the new live event feed wrapper is Alpaca-only.
- `StreamingRunner` does not yet replicate the full `FinTrade.run(...)` feature surface:
  - `DecisionContext`
  - decision/session guards
  - reconciliation/remediation loop
  - sector/net/turnover/ADV controls
- No built-in streaming daemon, persistence layer, or replay CLI yet.
