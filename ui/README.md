# Shunya UI (`ui/`)

![Shunya UI](docs/banner.png)

[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)

Web application for **[Shunya](https://github.com/Kaushikdey647/shunya)** — the Python library and FastAPI service for multi-ticker panels, JAX-style alphas, backtrader backtests, and (in the library) OMS/EMS building blocks. This directory is the **web front-end** in the shunya monorepo; it talks to the backend over JSON HTTP APIs (no backtest execution in the browser). **Trade → Account** and **Settings → Alpaca** call broker proxy routes when the API has Alpaca enabled and you paste the same `SHUNYA_API_TRADE_DESK_TOKEN` as the `X-Shunya-Trade-Desk-Token` header. Other **Trade** pages still use **client-side mock state** (`localStorage` key `shunya_trade_desk_v1`) until additional live endpoints exist.

**Backend / library:** [github.com/Kaushikdey647/shunya](https://github.com/Kaushikdey647/shunya) · API reference: [api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md) · Release notes: [CHANGELOG.md](CHANGELOG.md)

**Setup (end-to-end):** [Shunya README — Service and UI setup](https://github.com/Kaushikdey647/shunya/blob/main/README.md#service-and-ui-setup) (API first, then this UI).

---

## Features

| Area | What you get |
|------|----------------|
| **Home** | Macro strip, movers, headlines, recent backtests, browser watchlist, health |
| **Alpha Studio** | Monaco editor, lint + AI assist (optional Ollama), inline DSL hints, backtest enqueue, **results** below the editor (metrics strip, AI review of numbers, Recharts / lightweight-charts tearsheets) |
| **Backtests** | Create jobs, list, detail pages with charts |
| **Data** | Coverage / dashboard views backed by API |
| **Instruments** | Search and OHLCV-style charts |
| **Trade** | **Portfolios** — blend configs and weights; **Live** — cockpit-style strip; **Account** — Alpaca equity snapshot, portfolio history, activities (needs API Alpaca + trade desk token); **Execution** / **Risk** — mock hub + tracer + limits in **`localStorage`** until more live API routes exist; **Settings → Alpaca** — broker account configuration (same token) |

Visual design: **Bloomberg-inspired** dark terminal chrome — amber primary (`yellow` scale anchored ~`#FCB000`), warm neutral `dark` scale, IBM Plex Mono for monospace ([`src/mantine/theme.ts`](src/mantine/theme.ts)).

Legacy paths `/alphas/*` redirect to `/studio/*`.

---

## Requirements

- **Node.js** 20+ (recommended)
- **npm**
- **[shunya](https://github.com/Kaushikdey647/shunya)** API running locally (or a deployed URL you set at build time)

---

## Setup (API service + this UI)

Run steps in order: **backend first**, then the UI.

### Step 0 — Check the API is up

From the machine where the API runs:

```bash
curl -sSf http://127.0.0.1:8000/healthz
```

Expect **HTTP 200**. If this fails, fix the [shunya](https://github.com/Kaushikdey647/shunya) service before starting Vite (see **[Service and UI setup](https://github.com/Kaushikdey647/shunya/blob/main/README.md#service-and-ui-setup)** for Docker Compose vs `uv`, `DATABASE_URL`, and `shunya-timescale migrate`).

### Step 1 — Start the API (shunya)

Full instructions (Docker, `uv`, Postgres, migrations, Ollama, CORS): **[shunya README → Service and UI setup](https://github.com/Kaushikdey647/shunya/blob/main/README.md#service-and-ui-setup)** and **[api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)**.

Minimal local dev:

```bash
git clone https://github.com/Kaushikdey647/shunya.git && cd shunya
uv sync --extra dev --extra api --extra timescale
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
shunya-timescale migrate
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**Docker alternative:** in the **shunya** repo, `docker compose up -d` then `docker compose exec api uv run shunya-timescale migrate` (first run or after new migrations). Details: [Service and UI setup](https://github.com/Kaushikdey647/shunya/blob/main/README.md#service-and-ui-setup).

**CORS:** If you open the UI at `http://localhost:5173` and the API at `http://127.0.0.1:8000`, the Vite dev proxy avoids CORS. If you **disable the proxy** or use **different hosts/ports**, set on the API:

`SHUNYA_CORS_ORIGINS=http://localhost:5173` (exact origin, no trailing slash; comma-separate multiple).

**Alpha Studio AI:** set `SHUNYA_API_OLLAMA_HOST` (e.g. `http://127.0.0.1:11434`) and optionally `SHUNYA_API_OLLAMA_MODEL` on the API process.

### Step 2 — Install and run this UI

From the **shunya** repository root (after cloning [shunya](https://github.com/Kaushikdey647/shunya)):

```bash
cd ui
npm ci
npm run dev
```

Open the URL Vite prints (default **http://localhost:5173**).

| Topic | Detail |
|--------|--------|
| **Env files** | Vite loads **`.env`**, **`.env.local`**, **`.env.[mode]`**, etc. from this directory at **dev/build** time. Only **`VITE_*`** variables are exposed to client code (`import.meta.env`). See [`.env.example`](.env.example). |
| **Proxy** | [`vite.config.ts`](vite.config.ts) forwards browser requests from `/api` to **`http://127.0.0.1:8000`** and strips the `/api` prefix so the backend sees `/health`, `/alphas`, … |
| **API port** | Defaults to **8000**; change [`vite.config.ts`](vite.config.ts) `server.proxy['/api'].target` if your API listens elsewhere |
| **Prod build** | `npm run build` then `npm run preview` (or `npm run start` for `0.0.0.0` + `$PORT`) |

### Step 3 — Production / custom API URL

Vite reads **`VITE_*`** variables from **`.env` / `.env.local`** (and mode-specific variants) at **build** (and `vite dev`) time — not from the shunya API’s `.env`. Use **`.env.local`** for machine-specific overrides (keep it out of git).

```bash
VITE_API_BASE=https://api.yourdomain.com npm run build
```

[`src/api/client.ts`](src/api/client.ts) uses `import.meta.env.VITE_API_BASE ?? '/api'`. Use a **full origin** when the UI and API are on different hosts; use the default **`/api`** when your host (e.g. nginx) reverse-proxies `/api` to FastAPI.

**Railway / CI:** set `VITE_API_BASE` in the environment **before** `npm run build` so the client bundle embeds the correct API URL.

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server + HMR |
| `npm run build` | `tsc -b` then production build → `dist/` |
| `npm run preview` | Local preview of production build |
| `npm run start` | `vite preview` on `0.0.0.0` (uses `$PORT` or **4173**) — useful for Railway-style checks |
| `npm run lint` | ESLint |

---

## Project layout

```
src/
  api/           # apiFetch, endpoints, DTO types (mirrors FastAPI models)
  alphaEditor/   # Monaco completions, inline hints, wrap helpers
  components/    # Shell, charts, dashboard widgets, editors, trade/…
  hooks/         # e.g. useTradeDesk
  lib/           # Shared helpers (watchlist, chart adapters, tradeDeskStore, …)
  pages/         # Route screens (Studio, backtests, data, settings, portfolios, live, execution, risk, …)
  App.tsx        # Router
  main.tsx       # QueryClientProvider, Mantine, Router
```

---

## Types and OpenAPI

`src/api/types.ts` is maintained by hand to match the Python API. After breaking backend changes, regenerate or update types (e.g. `openapi-typescript http://127.0.0.1:8000/openapi.json -o src/api/types.gen.ts` and merge).

---

## Stack

- **UI:** React 19, React Router 7, Mantine 9  
- **Data:** TanStack Query v5  
- **Forms:** react-hook-form, zod  
- **Charts:** Recharts, lightweight-charts  
- **Editor:** Monaco (`@monaco-editor/react`) for alpha source  

---

## Watchlist

The home **Watchlist** uses **`POST /market/snapshot`** and stores tickers in **`localStorage`** (`shunya_watchlist_v1`). It is browser-local only until a server-backed watchlist exists.

---

## Documentation

- This file — setup, proxy, hosting
- **[CHANGELOG.md](CHANGELOG.md)** — UI-facing changes and roadmap notes

---

## License

Align with the **[shunya](https://github.com/Kaushikdey647/shunya)** repository (MIT) unless stated otherwise.
