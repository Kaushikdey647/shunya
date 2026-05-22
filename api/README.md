# Shunya HTTP API (`api/` package)

**`api`** is distributed inside the **`shunya-py`** wheel (same install as the **`shunya`** library). The **`[api]`** optional extra adds **FastAPI**, **uvicorn**, and related dependencies needed to **run** the HTTP process (`pip install "shunya-py[api,timescale]"` is typical). Clone-only workflows use **`uv sync --extra api --extra timescale`** from the repository root.

- **`/alphas`** — CRUD on alpha definitions: optional **inline** `source_code` (Python saved in `api_alphas`, executed in the worker) or module **`import_ref`** (allow-list: `examples.alphas.<module>:alpha` when not using inline). At least one must be set. If `source_code` is non-empty, it takes precedence at backtest time. Run migration **`003_alpha_source_code.sql`** (via `shunya-timescale migrate`) for the `source_code` column. Stored `finstrat_config` JSON.
- **`/indices`** — List equity indexes from Timescale (`equity_indexes` + membership counts) with **raw index tickers** (e.g. `^GSPC`, `^BFX`) as `benchmark_ticker` for comparison series.
- **`/universes`** — CRUD on **`api_universes`**, membership under **`api_universe_members`** (equity-only when `symbol_classifications.quote_type` is present), **`GET /universes/{id}/tickers`** (flat member list), **`GET /universes/{id}/summary`** (sector/industry breakdown excluding unknown/other buckets; fundamentals aggregates from latest **`fundamentals_daily`** per member), **`GET /universes/{id}/return-analytics`** (aligned daily closes from **`ohlcv_bars`**: simple and log return correlations, cross-sectional volatility, PCA on standardized returns, HHI/CR5/CR10 from latest market caps). Migration **`014_api_universes.sql`** adds tables and optional **`api_alphas.default_universe_id`**.
- **`/backtests`** — Enqueue async jobs (`POST`), list/get status, fetch JSON results when `succeeded` (same shape as `FinBT.results(show=False)` plus optional `benchmark`). **`POST` body** always uses the fixed simulation window **`[2020-01-01, 2026-01-01)`** (end exclusive) and **daily** bars; client-supplied `fin_ts.start_date`, `end_date`, and `bar_spec` are overwritten. **`include_test_period_in_results`** (default `false`): when `false`, stored metrics and time series exclude the **test** slice from **2025-01-01** onward (tune-only view). **`POST` with `index_code`**: resolves constituents from `symbol_index_membership`, sets `benchmark_ticker` to the catalog **raw index** symbol, forces **`market_data_provider=timescale`** and **no Yahoo** for the run. **`POST` with `universe_id`** (mutually exclusive with `index_code`): resolves members from **`api_universe_members`**, requires client **`benchmark_ticker`**, Timescale-only panel, optional **`omit_universe_members_missing_ohlcv`** (mirror of the index omit flag). By default OHLCV must exist for **every** constituent plus the benchmark; set **`omit_index_members_missing_ohlcv`: true** for index jobs (or the universe omit flag above) to drop members with no bars in the window (benchmark still required — ingest e.g. `^NDX` / `^GSPC` if missing).
- **`POST /data`** — Panel NaN counts per ticker/column and per-ticker **`return_pct`**, **`log_return_pct`** (`100 * ln(c_last / c_first)` when endpoint closes are strictly positive), annualized vol, Sharpe, and Sortino from `finTs` (Timescale when `DATABASE_URL` is set and `market_data_provider` is `auto` or `timescale`).
- **`GET /data/dashboard`** — Database-wide analytics for a stored `interval` / `source`: reference window `[MIN(ts), MAX(ts)]` over `ohlcv_bars`, per-ticker completeness vs that window (coverage time buckets), **`return_pct`** / **`log_return_pct`** / vol / Sharpe / Sortino from stored closes, and completeness histogram bins. Bucket granularity defaults to **`auto`** (chooses day, week, or month so the column count stays within `max_buckets`, default **200**); adjacent periods may be merged (logical OR). Optional **`SHUNYA_DASHBOARD_MAX_TICKERS`** caps symbols (alphabetical order).
- **`/instruments/...`** — Search and OHLCV: **`GET .../ohlcv`** accepts **`route`** (`auto`, `best_effort`, `timescale`, or explicit upstream such as `yfinance`, `alpaca_sip`); responses expose **`provenance`** (`read_path`, `upstream_source_id`, `route_rule_id`, …). Prefers Timescale when the DB is reachable, manifest-fresh, and coverage validates; otherwise live fetch per route. Optional write-back uses the **upstream** that satisfied the fetch. Env: **`SHUNYA_ALPACA_BAR_FEED`** (`sip`|`iex`|`delayed_sip`), **`SHUNYA_MARKET_DATA_DEMO_RELAXED`** (allows yfinance intraday in dev). See **`docs/market_data_routing.md`**. **WebSocket** **`/instruments/{symbol}/stream/alpaca-l1`** — when **`SHUNYA_API_ALPACA_ENABLED`** is set and Alpaca keys are present, streams stock/ETF **IEX** **quotes** and **trades** as JSON (`hello`, `quote`, `trade`, …); multiplexed via **`api.services.alpaca_l1_feed_hub`** (one Alpaca market-data WebSocket per key per API process; **`SHUNYA_ALPACA_L1_MAX_SYMBOLS`** caps distinct symbols, default **30**). See `api/routers/instrument_l1_stream.py`. Legacy **`.../stream/alpaca-bars`** returns **`deprecated_stream`**; see `api/routers/instrument_stream.py`.
- **`/market/...`** — Dashboard-oriented market data: **`POST /market/snapshot`** (batched daily OHLCV-derived quotes for macro strip / watchlists), **`GET /market/movers`** (`kind=gainers|losers|active`, Yahoo predefined screeners), **`GET /market/headlines`** (general financial headlines via Yahoo Search). Implemented in `api/services/market_*.py`.
- **`POST /trade/paper/cycle`** — One **paper** cycle: `PortfolioRiskEngine` → `InstitutionalOMS` → `EMSParentRunner` + Alpaca trade stream. Requires **`SHUNYA_API_ALPACA_ENABLED=1`**, Alpaca keys in the environment (`APCA_*` or `SHUNYA_ALPACA_*`), and **`SHUNYA_API_TRADE_DESK_TOKEN`**; send header **`X-Shunya-Trade-Desk-Token`** with that same value. Body: `capital`, `execution_date` (YYYY-MM-DD), either **`use_demo_pcs: true`** (built-in SPY/QQQ stub book) or **`targets_usd` + `universe` + `prices`** for a fixed-target run. Optional **`universe_resolution_note`** is echoed into response **`messages`** for desk audit. See [`api/routers/trade_desk.py`](routers/trade_desk.py).
- **`GET /settings/app`** — Read-only **environment** flags (no secrets) plus **effective runtime tunables** (env defaults merged with the optional DB overlay in `api_runtime_config`) and per-field **`sources`** (`database` vs `environment`). Does not require the trade-desk token.
- **`PATCH /settings/app`** — Merge **non-secret operator tunables** into `api_runtime_config` (same keys as `SHUNYA_API_*` caps such as worker poll interval, serialization caps, Ollama model/timeout, cache TTL). Requires **`DATABASE_URL`** (or equivalent) and migration **`013_api_runtime_config.sql`**. **Auth:** same **`X-Shunya-Trade-Desk-Token`** as **`POST /trade/paper/cycle`** when **`SHUNYA_API_TRADE_DESK_TOKEN`** is set; if that token is unset, returns **503** (writes disabled). Secrets (DB URL, Alpaca keys, `SHUNYA_API_OLLAMA_HOST`, CORS, trade-desk token) are **never** stored in the overlay.

## Install

From the repo root:

```bash
uv sync --extra api --extra timescale
```

### Web UI (optional)

The **`ui/`** React app uses **Node.js 20+** and **npm**, separate from the Python venv. From the repository root (with the API running on port **8000** for dev):

```bash
cd ui
npm ci
npm run dev
```

Vite prints a local URL (default **http://localhost:5173**); dev mode proxies **`/api`** to **`http://127.0.0.1:8000`** (HTTP and **WebSocket** upgrades, e.g. instrument Alpaca live bars). See [`ui/README.md`](../ui/README.md) and the docs **[Quickstart](https://kaushikdey647.github.io/shunya/quickstart/)** / **[Local development: API, worker, and UI](https://kaushikdey647.github.io/shunya/how-to/local-dev-api-ui/)**.

Set `DATABASE_URL` (or `SHUNYA_DATABASE_URL`) to your Postgres URL, apply migrations (including **`013_api_runtime_config.sql`** for `GET`/`PATCH /settings/app`), optionally seed data and alphas (see **[Bootstrap scripts (API + UI + DB)](https://kaushikdey647.github.io/shunya/how-to/bootstrap-scripts/)** or [`scripts/README.md`](../scripts/README.md)), then start the app:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
shunya-timescale migrate
shunya-timescale sync-index-memberships
uv run python scripts/bootstrap_example_alphas.py
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

For **SP100** or **full index-union** OHLCV and fundamentals ingest, use **`scripts/bootstrap_sp100_timescale.py`** or **`scripts/bootstrap_ts_data.py`** (ordering and flags in the bootstrap guide above).

### TimescaleDB (local, optional)

For a **quick local Timescale / Postgres** stack (Docker-based), you can use Timescale’s installer:

```bash
curl -sL https://tsdb.co/start-local | sh
```

Follow the script’s output for host, port, user, password, and database name, then point `DATABASE_URL` at that instance, for example:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@localhost:PORT/DATABASE'
```

If the installer creates only the default `postgres` database, either use that URL or create an app database (e.g. `createdb shunya`) and set `DATABASE_URL` accordingly. After `DATABASE_URL` is set, run **`shunya-timescale migrate`** once before starting the API.

Without Timescale (or with the DB down), instrument OHLCV and related paths still work via **yfinance**; see **Environment** below for TLS (`SHUNYA_TLS_VERIFY` and optional `curl-cffi`).

Alternatively, use **Docker Compose** in this repo: **`docker compose up --build`** builds **`timescaledb`**, runs a one-shot **`bootstrap`** (migrate + optional Yahoo ingest when the DB is empty), then **`api`** and **`ui`** (see the root **`docker-compose.yml`** and **`docs/quickstart.md`** / **`docs/data_timescale.md`**).

## Environment

At process start the API loads a **repo-root `.env`** (if present) via `python-dotenv`, then **pydantic-settings** with prefix **`SHUNYA_API_`** (see [`api/settings.py`](settings.py)). Changing `.env` still requires an **API restart**; values that must update live without restart belong in the **`api_runtime_config`** overlay (via **`PATCH /settings/app`**), not in `.env` edits from clients.

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` / `SHUNYA_DATABASE_URL` | Postgres for API tables + Timescale OHLCV reads |
| `SHUNYA_CORS_ORIGINS` | Optional comma-separated browser origins for cross-origin frontends (e.g. `https://app.vercel.app`). Omit if the UI is same-origin or only server-side clients call the API. |
| `SHUNYA_RUN_TIMESCALE_CONTAINER` | Set to `1` to run Timescale-backed tests via Docker testcontainers when `DATABASE_URL` is unset, and for an **isolated** DB for `test_alphas_crud_and_backtest_job` (ignores shared `DATABASE_URL` for that fixture). |
| `SHUNYA_API_INTEGRATION_DATABASE_URL` | Optional **dedicated** Postgres URL for queue-based API integration tests (`test_alphas_crud_and_backtest_job`). Use when you cannot run Docker testcontainers but must not share the job queue with another process (e.g. a running `uvicorn` on the same `DATABASE_URL`). |
| `SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS` | Set to `1` only with `DATABASE_URL` / `SHUNYA_DATABASE_URL` when **no** other API worker process uses that database; otherwise job-queue tests can flake or fail. Prefer testcontainers or `SHUNYA_API_INTEGRATION_DATABASE_URL`. |
| `SHUNYA_DASHBOARD_MAX_TICKERS` | Optional cap for `GET /data/dashboard` symbol list (positive integer); omit for no cap |
| `SHUNYA_API_ALPACA_ENABLED` | When `true` / `1`, the API builds shared Alpaca clients at startup (requires `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` or `SHUNYA_ALPACA_*` aliases). |
| `SHUNYA_API_TRADE_DESK_TOKEN` | Shared secret for `POST /trade/paper/cycle` and **`PATCH /settings/app`**; required header `X-Shunya-Trade-Desk-Token` must match exactly. |
| `SHUNYA_API_OLLAMA_HOST` | Base URL for Ollama (alpha assist); not stored in DB overlay. |
| `SHUNYA_API_OLLAMA_MODEL` | Default Ollama model id; may be overridden by **`api_runtime_config`** when set via **`PATCH /settings/app`**. |
| `SHUNYA_API_OLLAMA_TIMEOUT_SECONDS` | Default HTTP timeout for Ollama; may be overridden by the DB overlay. |
| `SHUNYA_API_DATABASE_URL` | Optional override (via `pydantic-settings`) |
| `SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS` | Worker poll interval (default `1.0`) |
| `SHUNYA_API_INDEX_OHLCV_BACKFILL_BATCH_SIZE` | Tickers per Yahoo batch when the worker backfills OHLCV after a recoverable index backtest data error (default `40`). |
| `SHUNYA_TLS_VERIFY` | **Unset** or `1` / `true` / `yes` / `on`: verify TLS certificates for **yfinance** and **Alpaca** clients built via `shunya.integration.alpaca_settings`. `0` / `false` / `no` / `off`: disable verification (yfinance uses `curl_cffi` with `verify=False` when installed; Alpaca REST `requests` session and trading WebSocket use insecure contexts)—**dev / broken corporate inspection only**. |

## High-level design (HLD)

**Diagram rendering:** Mermaid blocks below render on GitHub and in editors with Mermaid preview. The published docs site enables **Mermaid** via [`mkdocs.yml`](../mkdocs.yml) (`mermaid2` + `pymdownx.superfences`); this file is **`api/README.md`** (not under `docs/` nav by default—browse it in the repo or on GitHub).

### System context

One **`uvicorn`** OS process runs **FastAPI** and an **asyncio** background task that polls Postgres for queued backtest jobs ([`api/main.py`](main.py) lifespan → `backtest_worker_loop`). HTTP handlers and the worker share the same **`DATABASE_URL`** and Python interpreter.

```mermaid
flowchart TB
  subgraph clients [Clients]
    UI[Web_UI]
    CLI[API_clients]
  end
  subgraph uvicorn [Single_uvicorn_process]
    HTTP[FastAPI_routers]
    WL[backtest_worker_async_task]
  end
  PG[(Postgres_Timescale)]
  YF[Yahoo_yfinance]
  ALP[Alpaca_optional]
  OLL[Ollama_optional]

  UI --> HTTP
  CLI --> HTTP
  HTTP --> PG
  HTTP --> YF
  HTTP --> ALP
  HTTP --> OLL
  WL --> PG
  WL --> YF
```

### Logical components

Routers validate requests and delegate to **repositories** (SQL) or **services** (Timescale analytics, market helpers). **`api/db.py`** opens short-lived **psycopg** connections. Backtest execution flows through **`api/worker.py`** → **`api/worker_job.py`** → **`api/runner.py`**, which builds **`finTs`** via **`api/fin_ts_factory.py`**: **`resolve_market_data_provider`** (in **`shunya.data.market_data.fints_bridge`**) aligns eligibility with **`resolve_market_route`**, then attaches **`TimescaleMarketDataProvider`**, **`AlpacaHistoricalMarketDataProvider`**, or implicit Yahoo when appropriate. Simulation uses **`shunya`** (**FinStrat**, **FinBT** / `BacktraderBacktestEngine`).

```mermaid
flowchart TB
  R[api_routers] --> Rep[api_repositories]
  R --> Svc[api_services]
  Rep --> DBL[api_db]
  Svc --> DBL
  DBL --> PG[(Postgres)]
  WL[api_worker_loop] --> Rep
  WL --> WJ[worker_job]
  WJ --> Run[runner]
  Run --> FTS[fin_ts_factory]
  FTS --> TS[TimescaleMarketDataProvider]
  TS --> PG
  Run --> SH[shunya_FinStrat_FinBT]
  FTS --> FinTs[finTs_yfinance_or_auto]
```

### Market data provider selection

`FinTsRequest.market_data_provider` controls how **`build_fin_ts`** resolves OHLCV: **`timescale`**, **`yfinance`**, **`alpaca`**, **`best_effort`**, or **`auto`**. Selection runs through **`shunya.data.market_data.fints_bridge.resolve_market_data_provider`** (same **`resolve_market_route`** rules as HTTP where applicable). See [`api/fin_ts_factory.py`](fin_ts_factory.py), [`shunya/data/market_data/fints_bridge.py`](../shunya/data/market_data/fints_bridge.py), and [`shunya/data/timescale/market_provider.py`](../shunya/data/timescale/market_provider.py).

```mermaid
flowchart LR
  M[market_data_provider]
  M --> T[timescale]
  M --> Y[yfinance]
  M --> A[auto]
  T --> TS[TimescaleMarketDataProvider]
  A --> D{DSN_and_psycopg}
  D -->|ok| TS
  D -->|fail| FinTs[finTs_no_provider_Yahoo]
  Y --> FinTs
```

### Component architecture

Routers are grouped by URL prefix; **trade desk** and **alpha assist** touch optional external brokers and Ollama. The in-process **worker** only runs **backtest** jobs; it does not serve HTTP.

```mermaid
flowchart TB
  subgraph http [HTTP_routers]
    alphas[alphas]
    backtests[backtests]
    data[data]
    market[market]
    instruments[instruments]
    universes[universes]
    indices[indices]
    app_settings[app_settings]
    trade_desk[trade_desk]
  end
  subgraph orch [Job_orchestration]
    worker_py[worker.py]
    worker_job_py[worker_job.py]
    runner_py[runner.py]
  end
  subgraph persistence [Persistence]
    repos[repositories]
    dbmod[db.py]
  end
  subgraph svc [Services]
    services[api_services]
  end
  subgraph md [Market_data_in_finTs]
    fts[fin_ts_factory]
    tsmd[TimescaleMarketDataProvider]
  end

  http --> repos
  http --> services
  services --> dbmod
  repos --> dbmod
  dbmod --> PG[(Postgres)]
  worker_py --> repos
  worker_py --> worker_job_py
  worker_job_py --> runner_py
  runner_py --> fts
  fts --> tsmd
  tsmd --> PG
  runner_py --> ExtYF[Yahoo_yfinance]
  trade_desk --> Alpaca[Alpaca_HTTP]
  alphas --> Ollama[Ollama_HTTP]
```

## Low-level design (LLD)

### Layers and modules

| Layer | Path | Role |
|-------|------|------|
| **Routers** | [`api/routers/`](routers/) | HTTP boundary; query/body models from [`api/schemas/`](schemas/). |
| **Repositories** | [`api/repositories/`](repositories/) | Parameterized SQL: alphas, `api_backtest_jobs` queue, universes, runtime config overlay. |
| **Services** | [`api/services/`](services/) | Instrument OHLCV orchestration, market snapshot/movers/headlines, universe return analytics, yfinance backfill ([`api/services/ohlcv_yfinance_backfill.py`](services/ohlcv_yfinance_backfill.py)), dashboard-oriented Timescale queries used by routers. |
| **Data / dashboard** | [`api/data_service.py`](data_service.py), [`api/db_dashboard.py`](db_dashboard.py) | `POST /data` summaries vs `GET /data/dashboard` DB-wide coverage. |
| **Orchestration** | [`api/runner.py`](runner.py), [`api/worker_job.py`](worker_job.py), [`api/worker.py`](worker.py) | Serialize FinBT results; claim/execute jobs; async poll loop. |
| **Resolution** | [`api/resolver.py`](resolver.py), [`api/index_universe.py`](index_universe.py), [`api/universe_resolve.py`](universe_resolve.py), [`api/backtest_resolve.py`](backtest_resolve.py) | Alpha import vs inline code; index/universe membership for job payloads. |

Deeper package outline: [HTTP API package — Reference](../docs/reference/http-api-package.md).

### Settings resolution

Effective tunables merge **environment** (`SHUNYA_API_*` / `.env` loaded at startup) with the optional **`api_runtime_config`** JSON row (**`PATCH /settings/app`**). Secrets never go in the overlay; changing `.env` requires process restart. See the **Environment** table above and [`api/tunable_config.py`](tunable_config.py).

### Sequence — async backtest job

```mermaid
sequenceDiagram
  participant C as Client
  participant API as FastAPI_routers_backtests
  participant JR as repositories_backtests
  participant PG as Postgres
  participant WL as worker_loop
  participant WJ as worker_job
  participant RN as runner
  participant FTS as fin_ts_factory

  C->>API: POST /backtests
  API->>JR: insert queued job
  JR->>PG: INSERT api_backtest_jobs
  loop Poll_until_stop
    WL->>JR: claim_next_queued_job
    JR->>PG: UPDATE SKIP_LOCKED
  end
  WL->>WJ: execute_claimed_backtest_job
  WJ->>RN: run_backtest_from_payload
  RN->>FTS: build_fin_ts
  FTS->>PG: Timescale OHLCV optional
  RN-->>WJ: serialized plus summary
  WJ->>JR: mark_job_succeeded_or_failed
  JR->>PG: UPDATE api_backtest_jobs
```

Optional retry: for Timescale-backed jobs and recoverable `finTs` strictness errors, [`worker_job`](worker_job.py) may **yfinance-backfill** OHLCV then retry once.

### Sequence — dashboard read path

```mermaid
sequenceDiagram
  participant C as Client
  participant R as routers_data
  participant DD as db_dashboard
  participant PG as Postgres

  C->>R: GET /data/dashboard
  R->>DD: compute_data_dashboard
  DD->>PG: SQL on ohlcv_bars symbols
  PG-->>DD: aggregates
  DD-->>R: DataDashboardResponse
  R-->>C: JSON
```

### Data model

Schema is defined by SQL migrations under [`shunya/data/timescale/migrations/`](../shunya/data/timescale/migrations/). The diagram below shows **primary keys and foreign keys** only; hypertables include **`ohlcv_bars`** (time `ts`) and **`fundamentals_daily`** (`as_of_ts`). Full column lists live in **`001_init.sql`** through **`015_*.sql`**.

**API tables:** `api_alphas` (optional `default_universe_id` → `api_universes`), `api_backtest_jobs` (`alpha_id`, `request_payload`, `result_payload`, `status`, `execution_log`, `error_code`, …), `api_universe_members` (composite PK `universe_id` + `symbol_id`), `api_runtime_config` (singleton `id = 1`).

**Market / research tables:** `symbols`, `ohlcv_bars`, `equity_indexes`, `symbol_index_membership`, `symbol_classifications`, wide fundamentals (`fundamentals_daily`, `fundamentals_quarterly`, `fundamentals_annual`), events (`corporate_actions`, `insider_transactions`, `earnings_dates`), legacy EAV `fundamentals_field_values`, cache `ohlcv_symbol_interval_refresh`, `instrument_yfinance_documents`, `ingestion_runs`.

```mermaid
erDiagram
  api_alphas ||--o{ api_backtest_jobs : alpha_id
  api_universes ||--o{ api_universe_members : universe_id
  api_alphas }o--o| api_universes : default_universe_id
  symbols ||--o{ api_universe_members : symbol_id
  symbols ||--o{ ohlcv_bars : symbol_id
  symbols ||--o{ symbol_classifications : symbol_id
  symbols ||--o{ fundamentals_field_values : symbol_id
  symbols ||--o{ fundamentals_daily : symbol_id
  symbols ||--o{ fundamentals_quarterly : symbol_id
  symbols ||--o{ fundamentals_annual : symbol_id
  symbols ||--o{ corporate_actions : symbol_id
  symbols ||--o{ insider_transactions : symbol_id
  symbols ||--o{ earnings_dates : symbol_id
  symbols ||--o{ ohlcv_symbol_interval_refresh : symbol_id
  symbols ||--o{ instrument_yfinance_documents : symbol_id
  equity_indexes ||--o{ symbol_index_membership : index_code
  symbols ||--o{ symbol_index_membership : symbol_id

  api_alphas {
    uuid id PK
    text name UK
    text import_ref
    text source_code
    jsonb finstrat_config
    uuid default_universe_id FK
  }
  api_backtest_jobs {
    uuid id PK
    uuid alpha_id FK
    text status
    jsonb request_payload
    jsonb result_payload
    jsonb execution_log
    text error_code
  }
  api_universes {
    uuid id PK
    text name UK
  }
  api_universe_members {
    uuid universe_id PK_FK
    bigint symbol_id PK_FK
  }
  api_runtime_config {
    int id PK
    jsonb payload
  }
  symbols {
    bigint id PK
    text ticker UK
  }
  ohlcv_bars {
    bigint symbol_id FK
    timestamptz ts
    text interval
    text source
  }
  equity_indexes {
    text code PK
    text display_name UK
  }
  symbol_index_membership {
    bigint symbol_id PK_FK
    text index_code PK_FK
  }
```

## Optional: prune stored OHLCV to the HTTP backtest window

HTTP backtests use daily bars in **`[2020-01-01, 2026-01-01)`** (end exclusive). To drop older or newer `ohlcv_bars` rows so stored data matches that policy (adjust if your `ts` semantics differ), run SQL once as an operator:

```sql
DELETE FROM ohlcv_bars
WHERE ts < TIMESTAMPTZ '2020-01-01'
   OR ts >= TIMESTAMPTZ '2026-01-01';
```

Re-bootstrap with `scripts/bootstrap_ts_data.py` (defaults use the same window) to refill the canonical range. See [Bootstrap scripts (API + UI + DB)](https://kaushikdey647.github.io/shunya/how-to/bootstrap-scripts/) for ingest choices and `scripts/bootstrap_sp100_timescale.py`.

## Tests

- Unit: `pytest tests/test_api/ -m "not timescale"`
- DB + HTTP integration: `pytest tests/test_api/ -m timescale` with `DATABASE_URL` set, or `SHUNYA_RUN_TIMESCALE_CONTAINER=1` and Docker (skips if Docker is unavailable).
- **`test_alphas_crud_and_backtest_job`** enqueues `api_backtest_jobs` rows; any other process using the same database URL and running the API worker can **claim the same job** (`FOR UPDATE SKIP LOCKED`). That produces failures whose traceback points at `api/worker.py` even when pytest patches the in-process worker. Use **`SHUNYA_RUN_TIMESCALE_CONTAINER=1`** (isolated Timescale container for that test, even if `DATABASE_URL` is set) or **`SHUNYA_API_INTEGRATION_DATABASE_URL`** pointing at a database **no live API** is connected to. Only if you are sure nothing else is polling the queue: **`SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS=1`** with `DATABASE_URL`.
