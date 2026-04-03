# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

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
  - `CONTRIBUTION.md`
  - expanded `README.md` sections for controls, diagnostics, and roadmap status

### Changed

- `FinTrade.run(...)` interface extended with additional risk, decision-time, and reconciliation knobs.
- `FinBT` interface extended with richer constraint controls and enhanced diagnostics output.
- Public exports updated in `src/__init__.py` and `src/algorithm/__init__.py` for newly added helpers and diagnostics types.

### Testing

- Added tests:
  - `tests/test_fints_classification.py`
  - `tests/test_data_qa.py`
  - `tests/test_execution_adapter.py`
  - `tests/test_constraints.py`
  - `tests/test_integration_rebalance.py`
- Expanded tests:
  - `tests/test_decision.py`
  - `tests/test_finbt.py`
  - `tests/test_fintrade.py`
  - `tests/test_finstrat.py`
  - `tests/test_targets.py`
- Current status: full suite passing (`41 passed`).
