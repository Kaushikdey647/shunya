# shunya

![Shunya](docs/banner.png)

[![PyPI](https://img.shields.io/pypi/v/shunya-py.svg)](https://pypi.org/project/shunya-py/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/package_manager-uv-6f42c1.svg)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![Data](https://img.shields.io/badge/data-yfinance-informational.svg)](https://pypi.org/project/yfinance/)
[![Broker](https://img.shields.io/badge/broker-alpaca--py-orange.svg)](https://github.com/alpacahq/alpaca-py)
[![Web UI](https://img.shields.io/badge/UI-shunya--ui-646CFF?logo=react&logoColor=white)](https://github.com/Kaushikdey647/shunya-ui)

**Shunya** is a Python stack for **systematic equity research**: multi-ticker **OHLCV panels** (`finTs`), **JAX** alpha pipelines (**FinStrat** / `cross_section`), **backtrader** execution (**FinBT**), a decoupled **portfolio** layer (`PortfolioConstructionService` with `TargetBlendConfig` / `AlphaBlendConfig`, plus legacy `PortfolioManager` / `AlphaBlendPortfolioManager` facades, `RollingSharpeTracker`), optional **pre-trade risk** (`PortfolioRiskEngine`, `RiskVetConfig` / `RiskVetResult`, optional **`[risk]`** extra for CVX-backed checks), an institutional **OMS** (`shunya.oms` — parent FSM, share reconciliation, Alpaca trade stream bridge, optional SQLAlchemy persistence + **Alembic** migrations under `alembic/`) and **EMS** (`shunya.ems` — broker gateway, TWAP/VWAP slices, micro-price limits, async parent runner), optional **Alpaca** execution primitives (`AlpacaExecutionAdapter`, `OrderManager`), optional **TimescaleDB** for durable bars and fundamentals, and a repo-local **FastAPI** service for **alphas, async backtests, instruments, market dashboards, and data coverage APIs**. A separate **React** app (**[shunya-ui](https://github.com/Kaushikdey647/shunya-ui)**) provides Alpha Studio (Monaco + lint/assist), backtest management, a **Trade** desk (portfolios, live, execution, risk) with mock client state until OMS/EMS HTTP APIs land, and charts against this API.

Historical data is provider-driven: **`yfinance`** by default, optional **Alpaca** bars, **Tiingo** EOD for ingest, and **Timescale**-backed reads when `DATABASE_URL` is configured. Technicals attach via **`finta`**.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for architecture, extension patterns, and coding guidelines.

**Navigate:** [Layout](#layout) · [Core ideas](#core-ideas) · [Quick start](#quick-start) · [Service and UI setup](#service-and-ui-setup) · [Local TimescaleDB](#local-timescaledb-optional) · [HTTP / dashboard API](#http-api-and-dashboard-api) · [Portfolio management](#portfolio-management) · [OMS and EMS](#oms-and-ems-institutional-execution) · [Development tests](#development-tests) · [Documentation](#documentation)

## Layout

| Package | Role |
|--------|------|
| `shunya.data` | `finTs` — download OHLCV, build MultiIndex `(Ticker, Date)` frame, attach engineered columns; optional **TimescaleDB** persistence and read providers (see [Local TimescaleDB](#local-timescaledb-optional)) |
| `shunya.algorithm` | `FinStrat`, `FinBT`, `PortfolioConstructionService`, `PortfolioManager` / `AlphaBlendPortfolioManager` / `StrategyRegistry`, `RollingSharpeTracker`, `PortfolioRiskEngine` / `RiskVetConfig` (optional **`[risk]`**), `AlpacaExecutionAdapter` / `OrderManager`, `cross_section`, … |
| `shunya.oms` | Institutional OMS: parent-order FSM, in-memory ledger, share reconciliation vs USD targets, Alpaca stream + REST helpers, optional **Postgres** persistence via `shunya/oms/db` and repo-root **Alembic** (`alembic/versions/`) |
| `shunya.ems` | EMS: `BrokerGateway` / `AlpacaBrokerGateway`, TWAP/VWAP schedulers, micro-price limit helpers, `EMSParentRunner` for child lifecycle |
| `shunya.utils` | `indicators` — column namespaces (`COL`, `IX`, `IX_LIVE`), strategy feature lists, helpers |
| `api` | **FastAPI** service (alphas, backtest jobs, worker queue, data dashboard, instruments, market routes, optional **Ollama**-backed alpha assist / backtest review); requires `--extra api` (+ `--extra timescale` for Postgres). Consumed by **[shunya-ui](https://github.com/Kaushikdey647/shunya-ui)**. See [HTTP API and dashboard](#http-api-and-dashboard-api). |

Common imports from `shunya` (illustrative):

```python
from shunya import (
    AlphaBlendPortfolioManager,
    DecisionContext,
    FinBT,
    FinStrat,
    PortfolioConstructionService,
    PortfolioManager,
    cross_section,
    finTs,
    indicators,
)
```

The canonical set of symbols re-exported at the package root is `__all__` in [`shunya/__init__.py`](shunya/__init__.py) (for example OMS/EMS types such as `InstitutionalOMS`, `EMSParentRunner`, `PortfolioRiskEngine`, `PanelQADiagnostics`, `YFinanceMarketDataProvider`, `TimescaleMarketDataProvider`, `TimescaleFundamentalDataProvider`, `apply_migrations`, `get_database_url`, `PortfolioConstructionService`, `PortfolioManager`, `AlphaBlendPortfolioManager`, `StrategyRegistry`, `OrderManager`, target helpers, `logical`, `time_series`, and timestamp helpers). [`shunya.data`](shunya/data/__init__.py) also exports `TiingoMarketDataProvider` for Tiingo-backed OHLCV loading and CLI ingest.

## Core ideas

1. **`finTs`** loads one or many tickers and produces a dataframe whose live columns include raw **Open / High / Low / Close / Volume** first, then indicators (`VWAP`, `SMA_50`, `RSI_14`, …). It also attaches best-effort yfinance classification columns: **`Sector`**, **`Industry`**, **`SubIndustry`** (with deterministic `Unknown*` fallbacks). See `indicators.STRATEGY_FEATURES_LIVE` for the full default ordering.

2. **`FinStrat(fin_ts, algorithm, ...)`** binds the panel to a context callable `algorithm(ctx) -> (n_stocks,)`, where `ctx` exposes:
   - base series: `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.adj_volume`
   - time-series operators: `ctx.ts.*` (for example, `ctx.ts.mean(ctx.close, 50)`)
   - cross-sectional operators: `ctx.cs.*` (for example, `ctx.cs.rank(signal)`)

   Optional BRAIN-like knobs:
   - `decay` (per-name EMA on raw scores — pass `tickers=` into `pass_`)
   - `truncation` (cross-sectional winsorize)
   - `neutralization`: `"market"`, `"none"`, `"sector"` (demean within `Sector`), `"industry"` (demean within `Industry`), or `"group"` (demean within caller-supplied `group_ids` / custom `group_column` on `FinBT`)
   - `max_single_weight`

3. **`FinBT(fin_strat, fin_ts, ...)`** runs the same `FinStrat` on the same `fin_ts` instance in backtrader, rebalancing to `pass_` dollar targets each bar. `run()` resets `FinStrat` decay state. Pass **`commission`** (broker rate) and optional **`slippage_pct`** (adverse percent via backtrader’s `set_slippage_perc`). For `neutralization="group"`, pass **`group_column`** (defaults to `"Sector"` when omitted). `neutralization="sector"` / `"industry"` require the corresponding column on `fin_ts.df`. `broker_deltas` / `target_usd_universe` in `shunya.algorithm.targets` mirror how live orders diff targets vs positions.

4. **`PortfolioConstructionService`** (`shunya.algorithm.portfolio_manager`): single entry point wrapping **`TargetBlendConfig`** (same semantics as `PortfolioManager`) or **`AlphaBlendConfig`** (same as `AlphaBlendPortfolioManager`). Use **`construct(...)`** for a [`PortfolioConstructionResult`](shunya/algorithm/portfolio_manager.py) (`targets`, `requested_capital`, `active_capital`, correlation flags, ticker list) for risk / OMS logging; **`net_targets`** remains a thin wrapper over `construct(...).targets`. **`PortfolioManager`** / **`AlphaBlendPortfolioManager`** are legacy facades over the same engines. Rolling Sharpe uses **caller-supplied** simple returns, or an optional **`StrategyReturnFeed`** when `record_returns_from_feed_on_construct=True`. Portfolio math is decoupled from market-data transports and from order routing.

5. **`AlpacaExecutionAdapter` / `OrderManager`** — low-level Alpaca helpers for translating signed USD deltas into orders and for caching open-order state across **your** rebalance loop. There is no bundled tick-to-trade runner; wire these behind your own scheduler or service.

6. **`DecisionContext`** (`shunya.algorithm.decision`) pins **signal time** and **data provenance** (`yfinance_research` vs `alpaca_bars`) so research and live workflows do not silently mix data sources. Use it wherever you resolve an `as_of` timestamp against a panel.

7. **`MarketDataProvider`** (`shunya.data.providers`) abstracts history loading: default `YFinanceMarketDataProvider` in `finTs`, optional `AlpacaHistoricalMarketDataProvider` for broker-aligned panels and parity checks vs Yahoo, **`TiingoMarketDataProvider`** for daily end-of-day OHLCV from [Tiingo](https://www.tiingo.com/) (unadjusted OHLC + volume; use `SHUNYA_TIINGO_API_KEY` or `TIINGO_API_KEY`), and optional **`TimescaleMarketDataProvider`** for panels read from a local Postgres/Timescale store after ingest (same OHLCV contract as Yahoo).
   - Provider output contract is consistent: `DatetimeIndex` named `Date`, normalized to daily granularity.
   - `AlpacaHistoricalMarketDataProvider` is strict: if requested symbols are missing bars, it raises a `ValueError` listing those symbols.
   - `TiingoMarketDataProvider` is **daily-only**; Yahoo-style tickers are mapped to Tiingo symbology for API calls only (e.g. `BRK.B` → `BRK-B`) while `symbols.ticker` in the database stays unchanged.

8. **`cross_section`** — JIT-friendly helpers: `rank`, `zscore`, `scale`, `sign`, `winsorize`, `neutralize_market`, `neutralize_groups`. `rank(x)` is increasing in `x` (smallest → ~0, largest → ~1); use `rank(-x)` to flip.

## Trading-time axis (minute/hour/day)

- `finTs(..., trading_axis_mode="observed")` keeps legacy behavior (calendar derived from observed panel rows).
- `finTs(..., trading_axis_mode="canonical")` builds a canonical US-equities trading calendar for the selected `BarSpec` (weekend gaps removed from bar progression).
- `strict_trading_grid=True` enforces provider timestamps lie on the canonical bar grid and have no in-session holes.
- `FinStrat(..., temporal_mode="bar_step")` advances decay one step per bar.
- `FinStrat(..., temporal_mode="elapsed_trading_time")` advances decay by trading-time distance; `FinBT` passes execution timestamps so this mode works out of the box.

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
pip install "shunya-py[timescale]"   # optional: Postgres client for local Timescale ingest + read providers

# From a clone (installs the local project; add --extra notebook for Jupyter notebooks):
uv sync --extra dev --extra timescale
# optional: HTTP API + worker (FastAPI) — see [Service and UI setup](#service-and-ui-setup)
# uv sync --extra dev --extra timescale --extra api
uv run pytest
```

## Service and UI setup

Run the **FastAPI service** from this repo, then the **[shunya-ui](https://github.com/Kaushikdey647/shunya-ui)** dev server. Order matters: start the API before the UI so health checks and proxied calls succeed.

### 1) API service (FastAPI + worker)

**Option A — Docker Compose** (TimescaleDB + API on **http://127.0.0.1:8000**)

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
docker compose up -d
# First time (or after new migrations): apply schema inside the running API container
docker compose exec api uv run shunya-timescale migrate
curl -sSf http://127.0.0.1:8000/healthz   # expect HTTP 200
```

See [`docker-compose.yml`](docker-compose.yml): the `api` service runs `uvicorn api.main:app` with `DATABASE_URL` pointing at the bundled `timescaledb` service.

**Option B — Local `uv`** (typical development)

```bash
git clone https://github.com/Kaushikdey647/shunya.git
cd shunya
uv sync --extra dev --extra api --extra timescale
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya   # omit only if you accept limited routes
shunya-timescale migrate
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
# equivalent: uv run python -m api
```

- **Without `DATABASE_URL`:** many **yfinance**-backed routes still work; **alphas, backtest job queue, `/data/dashboard`**, and similar features expect Postgres — use Docker Compose or point `DATABASE_URL` at a local Timescale instance ([Local TimescaleDB](#local-timescaledb-optional)).
- **Smoke test:** `curl -sSf http://127.0.0.1:8000/healthz` (fast liveness). Interactive docs: **http://127.0.0.1:8000/docs**.
- **Browser from another origin** (e.g. UI on port 5173): set **`SHUNYA_CORS_ORIGINS`** to the exact UI origin(s), comma-separated, no path or trailing slash — example: `SHUNYA_CORS_ORIGINS=http://localhost:5173`.
- **Alpha Studio AI** (lint/assist/backtest review): optional **`SHUNYA_API_OLLAMA_HOST`** (e.g. `http://127.0.0.1:11434`) and **`SHUNYA_API_OLLAMA_MODEL`** on the API process ([HTTP API and dashboard](#http-api-and-dashboard-api)).

More env vars, Railway, and route-level detail: [`api/README.md`](api/README.md).

### 2) Web UI (shunya-ui)

```bash
git clone https://github.com/Kaushikdey647/shunya-ui.git
cd shunya-ui
npm install
npm run dev
```

Open the URL Vite prints (default **http://localhost:5173**). In dev, **`vite.config.ts` proxies `/api` → `http://127.0.0.1:8000`**, so the API should listen on **8000** unless you change the proxy.

**Production UI:** set **`VITE_API_BASE`** at **build** time to your public API origin if the UI is not served behind the same host as `/api`. See the **[shunya-ui README](https://github.com/Kaushikdey647/shunya-ui/blob/main/README.md)** for `npm run build`, preview, and hosting notes.

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
    neutralization="sector",  # or "industry", "market", "none", or "group" + group_column
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

### Local TimescaleDB (optional)

Use a local **TimescaleDB** (Postgres + Timescale extension) as the durable store for OHLCV bars, fundamentals (EAV), and yfinance-style classifications. External APIs remain **loaders**; after ingest, research can point `finTs` at **`TimescaleMarketDataProvider`** / **`TimescaleFundamentalDataProvider`** so panels match the same contracts as Yahoo-backed paths (technicals are still computed in memory from OHLCV).

**Install:** `pip install "shunya-py[timescale]"` (adds `psycopg` with binary wheels; the core PyPI wheel does not depend on it).

**Connection:** set **`DATABASE_URL`** or **`SHUNYA_DATABASE_URL`** (example: `postgresql://postgres:postgres@localhost:5432/shunya`). Do not commit real passwords.

**Bootstrap (repo root):**

```bash
docker compose up -d
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
shunya-timescale migrate
shunya-timescale ingest-ohlcv --symbols "AAPL MSFT" --start 2020-01-01 --end 2024-01-01
# optional: shunya-timescale ingest-fundamentals … , ingest-classifications …
```

**Bootstrap S&P 100 (`SP100`) for local backtests:** from a **clone** of this repo (the fundamentals step imports `examples/yfinance_fundamental_provider`), with `DATABASE_URL` set and `uv sync --extra timescale`, run migrations plus membership sync, roughly five years of daily OHLCV for SP100 constituents and benchmark **`^OEX`**, fundamentals over a longer window (so quarterly fields forward-fill at attach time), and yfinance classifications:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
uv run python scripts/bootstrap_sp100_timescale.py
```

HTTP index backtests use `index_code: "SP100"` and require **`^OEX`** OHLCV in the backtest window alongside constituent bars (see [`api/backtest_resolve.py`](api/backtest_resolve.py)). Use `--dry-run` to print windows and (if the DB is reachable) current membership counts without writing; `--strict` fails the script if any constituent lacks bars in the OHLCV window.

**Tiingo EOD ingest:** set **`SHUNYA_TIINGO_API_KEY`** or **`TIINGO_API_KEY`**, then load daily bars into `ohlcv_bars` (same upsert path as Yahoo/Alpha Vantage). `--end` is **exclusive**; use `--tiingo-delay-seconds` to pace requests under account quotas; `--db-limit` / `--db-offset` chunk `--symbols-from-db` runs.

```bash
export TIINGO_API_KEY=...   # or SHUNYA_TIINGO_API_KEY
shunya-timescale ingest-ohlcv --symbols-from-db --start 2010-01-01 --end 2026-01-01 \
  --provider tiingo --source tiingo --tiingo-delay-seconds 0
```

Equivalent: `python -m shunya.data.timescale.cli …`. Override the DSN per run with `--database-url`.

### HTTP API and dashboard (`api`)

The **Python package** is imported as `api` (directory `api/` at **repo root**; it is not shipped in the PyPI wheel, which only bundles `shunya`). From a **clone**, install API dependencies with **`uv sync --extra api`** (add **`--extra timescale`** when routes need Postgres). The **`api`** optional extra on PyPI exists so dependency versions align when developing from source; use **`pip install "shunya-py[api,timescale]"`** only when you need those libraries alongside a checkout.

**Run the server** (install, `DATABASE_URL`, migrate, `uvicorn`) is covered in **[Service and UI setup](#service-and-ui-setup)** above; this subsection documents behavior, hosting, and integrations.

Bind address and port default to `127.0.0.1` / `8000`; override with **`SHUNYA_API_HOST`** and **`SHUNYA_API_PORT`**. For **Railway** and similar hosts, bind **`0.0.0.0`** and **`PORT`** (e.g. `uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT`). Use **`GET /healthz`** for load-balancer liveness (instant **200**); **`GET /health`** runs Postgres + Yahoo checks and is not suitable as a deploy probe. Example Railway settings: [`railway.toml`](railway.toml).

**Browser CORS:** If the UI is served from another origin (e.g. **shunya-ui** on Vercel), set **`SHUNYA_CORS_ORIGINS`** to a comma-separated **allowlist** of exact origins—scheme + host (+ optional port), **no path or trailing slash**. Example: `SHUNYA_CORS_ORIGINS=https://shunya-ui.vercel.app`. Add preview URLs as separate entries if needed. Register this on the **API** service (Railway variables); restart/redeploy so `create_app()` picks it up. Leave unset for same-origin or non-browser clients only.

**Docker Compose:** `docker compose up` starts TimescaleDB plus the API (`uvicorn api.main:app` on port **8000**); see [`docker-compose.yml`](docker-compose.yml).

**Backtest HTTP API (repo clone):** migrations include `api_alphas` / `api_backtest_jobs`, `equity_indexes` / `symbol_index_membership`. The API supports **`POST /backtests` with `index_code`** for Timescale-only index universes and **raw index** benchmark tickers; backtests use a **fixed daily window** `2020-01-01`–`2026-01-01` (exclusive end) with optional **tune-only** results hiding **2025-01-01** onward unless `include_test_period_in_results` is true (see [`api/README.md`](api/README.md)).

Additional **market overview** routes (yfinance-backed, used by the **shunya-ui** home dashboard):

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/market/snapshot` | Batched daily closes for requested symbols (last price, 1d % change, volume, sparkline series). |
| `GET` | `/market/movers` | Predefined Yahoo screeners: `kind=gainers\|losers\|active`. |
| `GET` | `/market/headlines` | Broad financial headlines via Yahoo Search. |

Service implementations live under `api/services/` (`market_snapshot.py`, `market_movers.py`, `market_headlines.py`); shared symbol validation is `api/services/market_symbols.py`.

**Alpha Studio helpers (optional Ollama):** `POST /alphas/lint-body`, `POST /alphas/assist-body`, and `POST /alphas/assist-backtest-review` power the **[shunya-ui](https://github.com/Kaushikdey647/shunya-ui)** Monaco workspace. Set **`SHUNYA_API_OLLAMA_HOST`** (and optionally **`SHUNYA_API_OLLAMA_MODEL`**, **`SHUNYA_API_OLLAMA_TIMEOUT_SECONDS`**) on the API process; when the host is unset, assist and review return empty or stub payloads (see `api/alpha_assist.py`).

**Read in code:**

```python
import os
from shunya import TimescaleMarketDataProvider, finTs

os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/shunya"

fts = finTs(
    "2020-01-01",
    "2024-01-01",
    ["AAPL", "MSFT"],
    market_data=TimescaleMarketDataProvider(),
)
```

Full workflow (test markers, testcontainers, fundamentals CLI quirks) is in [`docs/data_timescale.md`](docs/data_timescale.md).

### Portfolio management

**Canonical service:** [`PortfolioConstructionService`](shunya/algorithm/portfolio_manager.py) with either [`TargetBlendConfig`](shunya/algorithm/portfolio_manager.py) or [`AlphaBlendConfig`](shunya/algorithm/portfolio_manager.py). [`construct`](shunya/algorithm/portfolio_manager.py) returns [`PortfolioConstructionResult`](shunya/algorithm/portfolio_manager.py) (USD `targets`, `requested_capital`, `active_capital`, optional correlation diagnostics, ticker list). [`PortfolioManager`](shunya/algorithm/portfolio_manager.py) and [`AlphaBlendPortfolioManager`](shunya/algorithm/portfolio_manager.py) remain thin facades for existing call sites.

Two combination modes share the same **panel + `finTs`** contract; neither talks to brokers.

**Target blending (late aggregation)** — [`TargetBlendConfig`](shunya/algorithm/portfolio_manager.py) / [`PortfolioManager`](shunya/algorithm/portfolio_manager.py): each sub-advisor is a full `FinStrat` with its own neutralization / truncation / scaling inside `pass_`. You assign **capital weights** `w_i` summing to 1; each runs `pass_(…, capital=w_i * C)` and USD targets are **summed** per symbol. Use this when sub-strategies must keep independent risk post-processing.

**Alpha blending (early aggregation)** — [`AlphaBlendConfig`](shunya/algorithm/portfolio_manager.py) / [`AlphaBlendPortfolioManager`](shunya/algorithm/portfolio_manager.py): register sub-advisors in a [`StrategyRegistry`](shunya/algorithm/portfolio_manager.py) of [`StrategySpec`](shunya/algorithm/portfolio_manager.py) entries (`strategy_id`, `sub_strat`, `conviction_z`, optional turnover metadata). The PM builds **one** [`AlphaContext`](shunya/algorithm/alpha_context.py) from the reference sub-strat, runs **`scores_from_context`** on each sub (raw cross-section only), blends with normalized convictions `z_i`, then runs a single **master** [`FinStrat.process_raw_scores`](shunya/algorithm/finstrat.py) (decay, winsorize, neutralize, gross scale). **Sub-strat `neutralization` does not apply on this path**—only the master’s pipeline does. **Decay is master-only** in v1 (set sub-strat decay to 0 if you rely purely on the master).

**Ticker universe policy:** default [`TickerUniversePolicy.STRICT_INTERSECTION`](shunya/algorithm/portfolio_manager.py) requires identical `tickers_at(execution_date)` across all sub-strategies (safe fixed-length JAX stacks). Combining a Nasdaq-only book with an S&P-only book in one service instance needs a future **superset-and-mask** path (not implemented yet).

**Netting / internal crossing:** opposite raw contributions on the same ticker net before the master scales, reducing unnecessary turnover versus independent full books.

**Correlation dampening (optional):** set `correlation_max_pairwise_threshold` on alpha blend; when rolling **path** correlation of stacked raw vectors between any pair exceeds the threshold, **gross exposure** is cut by passing **`active_capital = correlation_penalty * capital`** into the master’s `process_raw_scores` (conviction weights `z_i` are unchanged).

**Returns and Sharpe:** simple daily returns are **caller-supplied** (`record_strategy_return` / FinBT PnL, live marks, etc.). Optionally pass a [`StrategyReturnFeed`](shunya/algorithm/portfolio_manager.py) and set `record_returns_from_feed_on_construct=True` on the service to ingest returns on each `construct`. [`mark_to_market_strategy_pnl_usd`](shunya/algorithm/portfolio_manager.py) is a pure helper for per-strategy PnL from a [`VirtualLedger`](shunya/algorithm/portfolio_manager.py) and two price snapshots (tests / future OMS wiring).

**Virtual ledger:** [`VirtualLedger`](shunya/algorithm/portfolio_manager.py) holds per-strategy theoretical USD; use [`allocate_proportional_by_request`](shunya/algorithm/portfolio_manager.py) to split a broker fill across sub-advisors (`filled * req_i / sum(req)`), then `apply_delta`. Attribution to live fills is still your service’s responsibility.

**Dynamic weights:** [`inverse_vol_weights`](shunya/algorithm/portfolio_manager.py) maps vol estimates to weights; [`AlphaBlendPortfolioManager.convictions_from_inverse_vol`](shunya/algorithm/portfolio_manager.py) uses rolling return vol from the embedded `RollingSharpeTracker` after you `record_strategy_return`. Rebuild a `StrategyRegistry` with updated convictions if you change the book.

```python
from shunya import (
    AlphaBlendConfig,
    AlphaBlendPortfolioManager,
    FinStrat,
    PortfolioConstructionService,
    PortfolioManager,
    StrategyRegistry,
    StrategySpec,
    TargetBlendConfig,
    finTs,
)

# Target blend: independent pass_ per strat, then sum USD targets
pm = PortfolioManager([("momentum", fs_a, 0.6), ("value", fs_b, 0.4)], sharpe_window=126)
targets_tb = pm.net_targets(capital=250_000.0, execution_date="2024-06-03")

svc_tb = PortfolioConstructionService(
    TargetBlendConfig((("momentum", fs_a, 0.6), ("value", fs_b, 0.4))),
    sharpe_window=126,
)
res_tb = svc_tb.construct(capital=250_000.0, execution_date="2024-06-03")
assert res_tb.targets == svc_tb.net_targets(capital=250_000.0, execution_date="2024-06-03")

# Alpha blend: raw blend + master process_raw_scores
reg = StrategyRegistry.from_specs(
    [
        StrategySpec("momentum", fs_a, conviction_z=1.0),
        StrategySpec("value", fs_b, conviction_z=1.0),
    ]
)
ab = AlphaBlendPortfolioManager(registry=reg, master=master_fs, sharpe_window=126)
targets_ab = ab.net_targets(capital=250_000.0, execution_date="2024-06-03")
res_ab = ab.construct(capital=250_000.0, execution_date="2024-06-03")
ab.record_strategy_return("momentum", 0.0012)
ab.record_portfolio_return(0.0009)
```

## OMS and EMS (institutional execution)

These modules are **library building blocks** for a split OMS/EMS architecture (parent intent vs child broker orders). They do not replace your own scheduling, auth, or hosting.

- **`shunya.oms`** — Convert risk-vetted **USD** targets into share deltas (`required_delta_shares`, `usd_targets_to_share_targets`), track parents with `ParentOrder` + `InMemoryLedger`, optional **SQLAlchemy** repositories under [`shunya/oms/db`](shunya/oms/db/schema.py), and Alpaca-oriented adapters (`AlpacaOMSTradeStream`, REST snapshot helpers). Apply **OMS schema** migrations from a clone with `alembic upgrade head` (see `alembic.ini`); use **`--extra dev`** for Alembic + `psycopg` in local workflows.
- **`shunya.ems`** — Route parent intents through `BrokerGateway` (Alpaca implementation included), schedule **TWAP/VWAP** child slices, derive limit prices from microstructure hints (`limit_price_for_child`), and run **`EMSParentRunner`** for submit / timeout / cancel / escalation loops.

Wire **`PortfolioRiskEngine`** / `RiskVetResult` into OMS ingest via `shunya.oms.risk_bridge` helpers when you want the same vet outputs in Python that the UI may eventually mirror over HTTP.

## Notebooks

- [`vwap_close_rank_backtest_yfinance.ipynb`](vwap_close_rank_backtest_yfinance.ipynb) — `finTs` (default `YFinanceMarketDataProvider`) → `FinStrat` (`rank(VWAP/Close)`) → `FinBT`.
- [`vwap_close_rank_backtest.ipynb`](vwap_close_rank_backtest.ipynb) — same alpha and `FinBT` flow with **`AlpacaHistoricalMarketDataProvider`** (requires `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`).

## Requirements

- Python **≥ 3.12**
- Main libraries: `jax`, `pandas`, `yfinance`, `tiingo`, `backtrader`, `finta`, `matplotlib`, … (see `pyproject.toml`)
- Optional **`[risk]`** extra (`cvxpy`, `osqp`, `scikit-learn`) for full **`PortfolioRiskEngine`** covariance / optimization paths — `uv sync --extra risk` or `pip install "shunya-py[risk]"`.

Install from [PyPI](https://pypi.org/project/shunya-py/) (import the **`shunya`** package):

```bash
pip install shunya-py
# optional: Jupyter kernel for the bundled notebooks
pip install "shunya-py[notebook]"
# optional: local TimescaleDB ingest + DB-backed market/fundamental providers
pip install "shunya-py[timescale]"
# optional: FastAPI/uvicorn deps (use with a repo clone; the `api` module is not in the wheel)
pip install "shunya-py[api]"
```

Install from a clone (e.g. with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
# optional: FastAPI HTTP API — add --extra api (+ --extra timescale for DB-backed routes)
```

## Classification and sector controls

- `finTs` attaches `Sector`, `Industry`, `SubIndustry` with deterministic fallbacks:
  - `UnknownSector`, `UnknownIndustry`, `UnknownSubIndustry`
- For sector- or industry-neutral alphas:
  - set `FinStrat(..., neutralization="sector")` or `neutralization="industry"` (requires `Sector` / `Industry` on the panel).
- For custom group columns (e.g. `SubIndustry`):
  - set `FinStrat(..., neutralization="group")` and pass `group_column=` on `FinBT` (defaults to `Sector` when omitted).
- For pre-trade concentration controls (apply in your own pipeline after `PortfolioManager.net_targets` or inside `FinBT`):
  - `FinBT(..., sector_gross_cap_fraction=0.30, sector_cap_mode="rescale")`
- For net/turnover/participation controls:
  - `group_net_cap_fraction=...` (fraction of portfolio gross)
  - `turnover_budget_fraction=...` (fraction of target gross turnover)
  - `adv_participation_fraction=...` (caps order deltas by ADV participation)

## Decision-time guards and reconciliation

- Use `DecisionContext` / `validate_panel_timestamp` when resolving `as_of` against a research panel.
- When you build your own submit loop on top of `AlpacaExecutionAdapter`, you can mirror backtest-style controls with helpers in `shunya.algorithm.targets` (sector gross/net caps, turnover budget, ADV participation) before calling `submit_delta_orders` or `submit_orders`.
- Patterns that used to live on the removed `FinTrade.run(...)` surface (for example `reconciliation_policy in {"warn_only", "retry_once", "cancel_and_retarget"}`) should be implemented in your service layer alongside `ExecutionReport`-style logging if you need them.

## Note on live data

`indicators` defines a lookahead column `Future_1d_Ret`. `FinStrat` context execution uses live OHLCV history and excludes lookahead fields by design.

## Risks: Yahoo vs Alpaca, margin, paper vs live

- **yfinance vs Alpaca vs Tiingo:** Yahoo-adjusted history, time zones, and corporate-action handling can differ from Alpaca’s tape and from Tiingo’s composite EOD feed. Treat research PnL on Yahoo-only panels as indicative; for execution alignment prefer `AlpacaHistoricalMarketDataProvider` and `DecisionContext(data_source="alpaca_bars")`, or reconcile closes explicitly before trusting live notionals.
- **Shorting and margin:** Negative target notionals imply shorts. Alpaca requires margin, borrow availability, and `shortable` assets; orders can reject if the account is cash-only or the name is not shortable. The execution layer warns on non-shortable names but does not guarantee borrow.
- **Paper vs live checklist:** Confirm `paper=True`/`False` on `TradingClient`, that keys are scoped and never committed to git, use `dry_run=True` on adapters for rehearsal, and surface warnings from your own submit loop (for example buying-power caps or Yahoo parity notes when you log `ExecutionReport`-style summaries).

## What Is Not There Yet

- No first-party market-data **websocket** client in this repository; ingest or poll prices in your own process and refresh `finTs` / marks on whatever schedule you choose.
- No bundled live **orchestrator** daemon: `PortfolioManager`, `AlpacaExecutionAdapter`, `InstitutionalOMS`, and `EMSParentRunner` are libraries for you to schedule (cron, worker, or a service you own). Persisted OMS rows require you to configure `DATABASE_URL` and run Alembic; there is no multi-tenant hosted OMS product in this repo.

## Development tests

```bash
uv sync --extra dev --extra timescale --extra api   # or: uv sync --all-extras
uv run pytest                      # default: unit tests only (@pytest.mark.timescale skipped unless env set)
uv run pytest tests/test_api/ -m "not timescale"   # FastAPI package tests only (no DB container)
export DATABASE_URL=postgresql://... && uv run pytest -m timescale   # against your DB
# or: SHUNYA_RUN_TIMESCALE_CONTAINER=1 uv run pytest -m timescale     # ephemeral Timescale via Docker
```

Details: [`docs/data_timescale.md`](docs/data_timescale.md).

## Publishing (maintainers)

Build with `uv build` (wheel and sdist). Upload with [Twine](https://twine.readthedocs.io/) or [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) from CI. Do not commit API tokens or `.pypirc`.

## Roadmap status

- P0 completed: yfinance classifications, group defaults, order-status observation, sector gross cap.
- P1 completed: decision/session guards, panel QA diagnostics, richer backtest diagnostics.
- P2 completed: reconciliation loop + remediation hooks, net/turnover/ADV constraints, integration tests.
- P3 superseded: tick-to-trade streaming (`shunya.streaming`, `StreamingRunner`) and the `FinTrade` orchestrator were removed in favor of a decoupled `PortfolioManager` plus explicit adapter usage.
- P4 in progress: **OMS/EMS** Python modules (`shunya.oms`, `shunya.ems`) and **`PortfolioRiskEngine`** for pre-trade checks; HTTP surface in `api/` for live trade desk TBD — **[shunya-ui](https://github.com/Kaushikdey647/shunya-ui)** currently uses client-side mock state for Trade routes.

## Documentation

- Main usage and behavior: [`README.md`](README.md)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)
- **Web UI (Alpha Studio, dashboards, Trade mock):** **[shunya-ui](https://github.com/Kaushikdey647/shunya-ui)** — [`README.md`](https://github.com/Kaushikdey647/shunya-ui/blob/main/README.md)
- Contributor and architecture guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Local Timescale market store (compose, migrate, ingest, `finTs`): [`docs/data_timescale.md`](docs/data_timescale.md)
- **Backtest + instrument HTTP API** (alphas, jobs, data dashboard, instruments, **market** overview): [`api/README.md`](api/README.md)
