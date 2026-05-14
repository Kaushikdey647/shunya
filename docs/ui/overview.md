# Web application overview

The **`ui/`** directory is a **React + Vite** front end for the Shunya **FastAPI** service. Backtests and alpha execution run on the **server**; the browser is for editing, configuration, and visualization.

## UI preview

Static captures live under `docs/*.png` in the repository (MkDocs serves them on the doc site).

![Home: macro strip, movers, and recent backtests](../dashboard.png)

| Area | Screenshot walkthrough |
|------|-------------------------|
| Research (home, data, settings) | [Research](research.md) — dashboard on **Home** |
| Alpha Studio, universes, backtests | [Studio](studio.md) — workspace and result charts |
| Running a simulation from the browser | [Backtest from the web UI](../how-to/backtest-from-ui.md) |
| Portfolios, live desk, execution | [Trade](trade.md) — blend workspace, Sentinel, EMS slicer |

## Feature map

| Area | What you get |
|------|----------------|
| **Home** | Macro strip, movers, headlines, recent backtests, browser watchlist, health |
| **Alpha Studio** | Monaco editor, lint and optional AI assist (Ollama via API), inline DSL hints, backtest enqueue, results below the editor (metrics, optional AI review, Recharts / lightweight-charts) |
| **Backtests** | Create jobs, list, detail pages with charts |
| **Data** | Coverage and dashboard views backed by the API |
| **Instruments** | Search and OHLCV-style detail (`/instruments/:symbol`); equities can be added to a saved universe |
| **Universes** | List and edit saved equity universes (`/universe`, `/universe/:id`) backed by **`/universes`** |
| **Trade** | Portfolios (blend configs); Live cockpit; Account (Alpaca when API + token); Execution and Risk surfaces (client mock state in `localStorage` until more live routes exist) |
| **Settings** | App runtime flags and Alpaca / trade-desk configuration where applicable |

## Design

Bloomberg-inspired dark terminal chrome: amber primary, warm neutrals, IBM Plex Mono for monospace (see `ui/src/mantine/theme.ts`).

## Legacy routes

Paths under **`/alphas/*`** redirect to **`/studio/*`** for compatibility.

## Source layout (high level)

| Path | Role |
|------|------|
| `src/api/` | `apiFetch`, endpoints, DTO types |
| `src/alphaEditor/` | Monaco completions, inline hints, wrap helpers |
| `src/components/` | Shell, charts, editors, trade widgets |
| `src/pages/` | Route screens |
| `App.tsx` | React Router routes |

## Install and run

The UI ships only inside this repository (not on PyPI). Install **Node.js 20+** and **npm**, then from the repo root: `cd ui`, **`npm ci`**, **`npm run dev`**. Start the FastAPI service on **port 8000** before relying on the dev proxy. Step-by-step: [Install](../install.md) and [Local development: API, worker, and UI](../how-to/local-dev-api-ui.md).

## Further reading

- [Research](research.md), [Studio](studio.md), [Trade](trade.md), [Configuration](configuration.md)
