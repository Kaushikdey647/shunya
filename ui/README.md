# Shunya UI (`ui/`)

![Shunya UI](docs/banner.png)

[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)

**React + Vite** front-end for [Shunya](https://github.com/Kaushikdey647/shunya): Alpha Studio (Monaco), backtests, data and instrument views, and trade-oriented screens backed by the repo’s FastAPI service (no backtest execution in the browser).

Research dashboard (Research → Dashboard at `/dashboard`; `/` is a static landing):

![Shunya UI — research dashboard](docs/dashboard.png)

**Documentation** (setup paths, API, web app overview, env and proxy notes):

**[kaushikdey647.github.io/shunya](https://kaushikdey647.github.io/shunya/)** — see [Web application](https://kaushikdey647.github.io/shunya/ui/overview/) and [Quickstart](https://kaushikdey647.github.io/shunya/quickstart/).

## Local development

Start the API first (for example `./scripts/local-dev-all.sh` from the repo root, optionally `./scripts/local-dev-all.sh --seed-alphas`, or follow the [Quickstart](https://kaushikdey647.github.io/shunya/quickstart/) and [Bootstrap scripts](https://kaushikdey647.github.io/shunya/how-to/bootstrap-scripts/) guide). Then:

```bash
cd ui
npm ci
npm run dev
```

Confirm the API: `curl -sSf http://127.0.0.1:8000/healthz`. In dev, Vite proxies `/api` to port **8000** by default ([`vite.config.ts`](vite.config.ts) → adjust `server.proxy` if needed). See [`.env.example`](.env.example) for `VITE_*` variables.

| Command | Purpose |
|---------|---------|
| `npm run dev` | Dev server + HMR |
| `npm run build` | Production build → `dist/` |
| `npm run preview` / `npm run start` | Preview production build |

## Contributing & license

Follow the root repo’s [`CONTRIBUTING.md`](https://github.com/Kaushikdey647/shunya/blob/main/CONTRIBUTING.md). **MIT** — same as [Shunya](https://github.com/Kaushikdey647/shunya/blob/main/LICENSE).
