# ADR 001: Shunya layering (library, ports, adapters, API)

## Context

The quant library (`shunya`), HTTP API (`api`), and UI must stay evolvable: swap simulation engines (Backtrader today) and optional NumPy-only cross-section paths without rewriting callers.

## Decision

1. **Ports** live under `shunya/ports/` as small `Protocol` modules. Import **submodules** directly (e.g. `from shunya.ports.backtest_engine import BacktestEngine`). Keep `shunya/ports/__init__.py` empty to avoid import cycles with `shunya.algorithm`.
2. **Adapters** live under `shunya/adapters/` (e.g. `BacktraderBacktestEngine`, `JaxCrossSectionOps`, `NumpyCrossSectionOps`). Keep `shunya/adapters/__init__.py` empty for the same reason.
3. **API** orchestration (`api/runner.py`) depends on ports + default adapters; tests may monkeypatch `api.runner._default_backtest_engine`.
4. **Errors**: domain exceptions and stable string codes live in `shunya/errors.py`; FastAPI maps them via `api/exception_handlers.py`.

## Consequences

- New engines or array backends add a class + wire-up in runner / `FinStrat` without changing serializers.
- Do not re-export heavy symbols from `shunya.adapters` package `__init__.py`.
