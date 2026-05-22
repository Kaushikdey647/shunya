# Reliability & accuracy roadmap

Engineering backlog for **trustworthy backtests** and **broker-faithful execution** on the Alpaca-centric path (see `shunya/integration/alpaca_settings.py`, `shunya/algorithm/finbt.py`, `shunya/oms/`, `shunya/ems/`). Treat items as candidates to prioritize; link PRs when started.

---

## Principles

- **Parity**: Every assumption used in backtest (timing, prices, costs, constraints) should be traceable to a documented rule or a broker/API contract.
- **Evidence**: Prefer paper → small-live checks, reconciliation diffs, and golden tests over “feels right.”
- **Single source of truth**: Account, positions, and open orders should reconcile to Alpaca (or be explicitly marked stale with reason).

---

## Backtest fidelity (`FinBT`, `FinStrat`, `finTs`)

- [ ] **Document the simulation contract** — bar timestamps (open vs close), rebalance moment vs bar boundary, and how `FinStrat.pass_` execution_date maps to fills (see `docs/concepts/pipeline-alpha-to-execution.md`, extend if gaps).
- [ ] **Align default costs with Alpaca** — commission schedule, regulatory fees, and whether backtest uses the same assumptions as paper/live (configurable, versioned defaults).
- [ ] **Slippage model** — explicit model (fixed bps, spread fraction, volume participation) and sensitivity runs; avoid implicit “perfect mid” unless documented.
- [ ] **Price basis** — define and test: last vs bid/ask vs bar close for notionals and fills; match what live `AlpacaExecutionAdapter` / desk uses for `prices` inputs.
- [ ] **Corporate actions** — document current behavior (none vs simplistic); if excluded, state impact on long-held backtests.
- [ ] **Intraday vs daily** — if `bar_spec_is_intraday` paths differ, add tests for session boundaries and `intraday_session_isolated_lag` behavior (`finbt.py`).
- [ ] **Constraint semantics** — sector/group caps, turnover budget, ADV participation: unit tests for edge cases (empty universe, single name, all NaN scores).
- [ ] **Determinism** — seed or fix any stochastic paths; record library versions in backtest job metadata for reproducibility.
- [ ] **Benchmark path** — ensure benchmark series uses same calendar and return definition as portfolio equity (`api/runner.py` benchmark block).

---

## Live & paper execution accuracy (`live/desk`, OMS, EMS, `alpaca-py`)

- [ ] **Order idempotency** — client order IDs: uniqueness, retries, and worker restarts without duplicate submissions.
- [ ] **Partial fills & cancels** — OMS state machine tests against Alpaca statuses; ensure `AlpacaOMSTradeStream` covers all relevant event types you rely on.
- [ ] **Fractional vs whole shares** — explicit policy per strategy/desk; reject or round with documented rules before submit.
- [ ] **Buying power & PDT** — surface Alpaca reject reasons to logs/metrics; optional pre-flight using account config APIs where available.
- [ ] **Extended hours** — if supported, explicit flag on orders and in risk checks; if not supported, block with clear errors.
- [ ] **Limit price / micro-price** — audit `ems/micro_price.py` and gateway rounding vs Alpaca tick constraints; add regression tests.
- [ ] **Warmup & race windows** — document `stream_warmup` and any sleep between subscribe and submit; reduce or replace with readiness signals where possible (`live/desk.py`).
- [ ] **Clock & timezone** — all scheduled cycles use market timezone explicitly; DST edge tests.

---

## Reconciliation & observability

- [ ] **Startup reconciliation** — on process start: pull positions + open orders from Alpaca; diff vs local OMS ledger; repair or alert (pattern similar to Nautilus “reconciliation_lookback” idea, implemented natively).
- [ ] **Periodic reconcile job** — optional cron/worker task: DB vs Alpaca positions for paper/live accounts.
- [ ] **Structured execution audit log** — intent → parent → child → fill chain with correlation_id (already partially there; ensure full trace in one queryable shape).
- [ ] **Metrics** — submit latency, fill rate, cancel/reject counts, slippage vs arrival mid (where L1 exists).
- [ ] **Alerting hooks** — stuck parent, unexpected position drift, stream disconnect > N seconds.

---

## Market data accuracy (routing, L1, bars)

- [ ] **Provenance everywhere** — ensure OHLCV and snapshots always carry `provenance` / upstream id for debugging mismatches (`market_router`, instrument APIs).
- [ ] **Backtest data vs live data** — document which provider feeds FinTs for jobs vs what Alpaca paper uses for execution prices; flag “research bar ≠ execution bar” scenarios.
- [ ] **L1 hub** — monitor `SymbolLimitExceeded` and hub health; document multi-tab limits (`SHUNYA_ALPACA_L1_MAX_SYMBOLS`).
- [ ] **Stale quotes** — optional max age for L1-driven UI or execution guardrails.

---

## Testing & validation

- [ ] **Golden backtest** — small fixed panel + alpha → committed expected metrics hash or narrow bounds (CI-friendly).
- [ ] **Paper soak tests** — scripted `shunya-paper` cycles against paper API in CI nightly or weekly (gated by secrets).
- [ ] **Mock Alpaca** — expand HTTP/WS mocks for order lifecycle edges (partial fill, replace, reject).
- [ ] **Property or fuzz tests** — target weights sum, gross/net bounds after `FinStrat` + caps.

---

## API & worker reliability

- [ ] **Backtest job idempotency** — duplicate job enqueue, worker retry: safe outcomes, no double broker calls if any future live hook exists in worker.
- [ ] **Timeouts** — worker kill hung backtests; API returns consistent job state.
- [ ] **Serialization limits** — tune `max_*_points` and document truncation impact on metrics (`api/runner.py`).

---

## Documentation & runbooks

- [ ] **“Truth table” doc** — one page: data source × environment (backtest / paper / live) × symbol type.
- [ ] **Incident runbook** — stream down, mass reject, reconcile drift: steps and rollback.
- [ ] **Config reference** — env vars affecting execution and backtest (`SHUNYA_ALPACA_*`, TLS, feeds) in single table.

---

## Research ↔ execution bridge (optional but high leverage)

- [ ] **Declared handoff format** — versioned JSON/schema for “targets as of T” from research lane consumed by desk/runner (weights, notionals, valid_until).
- [ ] **Pre-trade checks** — same risk rules callable from backtest post-pass and live `PortfolioRiskEngine` where possible (shared module, two callers).

---

## Future / non-blocking (revisit if venue or engine strategy changes)

- [ ] **Re-evaluate NautilusTrader** if/when a **maintained first-party Alpaca adapter** exists — then reassess backtest/live parity vs cost of owning `alpaca-py` stack.
- [ ] **Interactive Brokers** or other venues — only if product expands; Nautilus adapter coverage differs by venue.
- [ ] **Parquet / Arrow catalog** — optional storage format for bar replay without committing to a full engine migration.

---

## How to use this file

- Pick a **small vertical slice** (e.g. “reconciliation + one golden backtest”) per milestone.
- When an item ships, move it to `CHANGELOG.md` / `docs/` per repo conventions and strike it here or remove it.
