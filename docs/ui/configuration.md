# Web UI configuration

## Environment files

Vite loads **`.env`**, **`.env.local`**, **`.env.[mode]`**, etc. from the **`ui/`** directory at **dev and build** time. Only variables prefixed with **`VITE_`** are exposed to client code via `import.meta.env`.

See **`ui/.env.example`** for starting values.

## API base URL

`src/api/client.ts` uses:

`import.meta.env.VITE_API_BASE ?? '/api'`

| Scenario | Value |
|----------|--------|
| **Local dev with default proxy** | `VITE_API_BASE=/api` — `ui/vite.config.ts` proxies `/api` to `http://127.0.0.1:8000` and rewrites the path prefix off. |
| **Different API host in dev** | Set `VITE_API_BASE=http://127.0.0.1:8000` (or your URL) and adjust or disable the proxy as needed; ensure **CORS** on the API (`SHUNYA_CORS_ORIGINS`). |
| **Production, split hosts** | `VITE_API_BASE=https://api.yourdomain.com` at **`npm run build`** time (full origin). Configure CORS on the API for the UI origin. |

**Railway / CI:** set `VITE_API_BASE` **before** `npm run build` so the bundle embeds the correct API origin.

## Vite dev proxy

From **`ui/vite.config.ts`**:

- Requests to **`/api`** forward to **`http://127.0.0.1:8000`**
- Path rewrite strips the **`/api`** prefix so the backend sees `/health`, `/alphas`, etc.

Change **`server.proxy['/api'].target`** if your API listens on another port.

## Scripts

| Command | Use |
|---------|-----|
| `npm run dev` | Development server + HMR |
| `npm run build` | Typecheck + production bundle → `dist/` |
| `npm run preview` | Preview production build locally |
| `npm run start` | `vite preview` on `0.0.0.0` with `$PORT` (hosting smoke tests) |

## Types vs OpenAPI

`src/api/types.ts` is maintained manually to mirror FastAPI models. After breaking API changes, regenerate or merge from `openapi.json` (see **`ui/README.md`** for suggested tooling).

## See also

- [Local dev: API, worker, and UI](../how-to/local-dev-api-ui.md)
