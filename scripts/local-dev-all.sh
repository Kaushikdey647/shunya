#!/usr/bin/env bash
# Start local TimescaleDB (Docker), sync Python deps, migrate, then API + Vite UI.
# Optional: --seed-alphas inserts example alpha rows (see docs/how-to/bootstrap-scripts.md).
# Requires: Docker (with compose), uv, Node.js 20+, npm. Frees port 5432 for the compose service.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

SEED_ALPHAS=0
while [[ $# -gt 0 && "${1}" == -* ]]; do
  case "${1}" in
    --seed-alphas)
      SEED_ALPHAS=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: local-dev-all.sh [options]

Starts Docker TimescaleDB, uv sync (dev + api + timescale), migrations, FastAPI
on http://127.0.0.1:8000, then the Vite UI in ui/.

Options:
  --seed-alphas   After migrate, run scripts/bootstrap_example_alphas.py so GET /alphas
                  lists the bundled example alphas (idempotent skips existing names).
  -h, --help      Show this message

Data ingest (OHLCV, index jobs) is not run automatically; see bootstrap-scripts
in the docs site (how-to) or scripts/README.md in the repo.
EOF
      exit 0
      ;;
    *)
      die "unknown option: ${1} (try --help)"
      ;;
  esac
  shift
done
[[ $# -eq 0 ]] || die "unexpected arguments: $*"

command -v docker >/dev/null 2>&1 || die "docker not found; install Docker Desktop or the engine."
docker compose version >/dev/null 2>&1 || die "docker compose not available."
command -v uv >/dev/null 2>&1 || die "uv not found; see https://docs.astral.sh/uv/"
command -v node >/dev/null 2>&1 || die "node not found; install Node.js 20+."
command -v npm >/dev/null 2>&1 || die "npm not found."

major="$(node -p "parseInt(process.versions.node.split('.')[0], 10)" 2>/dev/null || echo 0)"
if [ "${major}" -lt 20 ]; then
  die "Node.js 20+ required (found $(node -v 2>/dev/null || echo unknown))."
fi

if [ ! -f .env.example ]; then
  die "missing .env.example at repo root"
fi

if [ ! -f .env ]; then
  info "creating .env from .env.example"
  cp .env.example .env
fi

if ! grep -qE '^[[:space:]]*(DATABASE_URL|SHUNYA_DATABASE_URL|SHUNYA_API_DATABASE_URL)=' .env; then
  info "appending default DATABASE_URL for local Docker Timescale (postgres@127.0.0.1:5432/shunya)"
  printf '\n# Added by scripts/local-dev-all.sh\nDATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/shunya\n' >> .env
fi

pick_database_url() {
  uv run python -c "
from pathlib import Path
from dotenv import dotenv_values
p = Path('.env')
v = dotenv_values(p) if p.is_file() else {}
u = (v.get('DATABASE_URL') or v.get('SHUNYA_DATABASE_URL') or v.get('SHUNYA_API_DATABASE_URL') or '').strip()
if not u:
    raise SystemExit('no DATABASE_URL / SHUNYA_DATABASE_URL / SHUNYA_API_DATABASE_URL in .env')
print(u)
"
}

info "starting TimescaleDB (docker compose service: timescaledb)"
docker compose up -d timescaledb

info "waiting for Postgres to accept connections"
ready=0
for _ in $(seq 1 60); do
  if docker compose exec -T timescaledb pg_isready -U postgres -d shunya >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
[ "${ready}" -eq 1 ] || die "timescaledb did not become ready in time (docker compose logs timescaledb?)"

info "syncing Python dependencies (dev, api, timescale)"
uv sync --extra dev --extra api --extra timescale

export DATABASE_URL="$(pick_database_url)"
info "running Timescale migrations"
uv run shunya-timescale migrate

if [[ "${SEED_ALPHAS}" -eq 1 ]]; then
  info "seeding example alphas (scripts/bootstrap_example_alphas.py)"
  uv run python scripts/bootstrap_example_alphas.py
fi

API_PID=""
cleanup() {
  if [ -n "${API_PID}" ] && kill -0 "${API_PID}" 2>/dev/null; then
    info "stopping API (pid ${API_PID})"
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

info "starting FastAPI on http://127.0.0.1:8000"
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 &
API_PID=$!

info "waiting for API /healthz"
ok=0
for _ in $(seq 1 45); do
  if curl -sSf "http://127.0.0.1:8000/healthz" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 1
done
[ "${ok}" -eq 1 ] || die "API did not respond on /healthz; check logs above."

if [ ! -d ui/node_modules ]; then
  info "installing UI dependencies (npm ci)"
  (cd ui && npm ci)
else
  info "ui/node_modules present; skipping npm ci (remove ui/node_modules to force a clean install)"
fi

info "starting Vite dev server (Ctrl+C stops UI and this script will stop the API)"
(cd ui && exec npm run dev)
