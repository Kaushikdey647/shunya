#!/bin/sh
# One-shot Compose service: migrate, then optionally run SP100 + example alphas + ts_data ingest.
# See docker-compose.yml service "bootstrap".
set -e
cd /app

if [ "${SHUNYA_COMPOSE_AUTO_BOOTSTRAP:-1}" = "0" ]; then
  echo "SHUNYA_COMPOSE_AUTO_BOOTSTRAP=0: running migrations only (no ingest scripts)."
  uv run shunya-timescale migrate
  exit 0
fi

echo "Running SQL migrations (shunya-timescale migrate)…"
uv run shunya-timescale migrate

echo "Checking whether Timescale data is already populated…"
set +e
uv run python docker/compose_bootstrap_probe.py
probe_rc=$?
set -e

if [ "$probe_rc" -eq 0 ]; then
  echo "OHLCV already present; ensuring SP100 classifications + fundamentals_daily for universe overview…"
  uv run python scripts/gapfill_sp100_universe_metadata.py
  exit 0
fi
if [ "$probe_rc" -ne 1 ]; then
  echo "compose_bootstrap_probe.py exited with $probe_rc" >&2
  exit "$probe_rc"
fi

echo "Running bootstrap_sp100_timescale.py (migrations already applied)…"
uv run python scripts/bootstrap_sp100_timescale.py --skip-migrate

echo "Running bootstrap_example_alphas.py…"
uv run python scripts/bootstrap_example_alphas.py

echo "Running bootstrap_ts_data.py…"
uv run python scripts/bootstrap_ts_data.py

echo "Ensuring SP100 universe overview data (classifications + fundamentals_daily)…"
uv run python scripts/gapfill_sp100_universe_metadata.py

echo "Compose bootstrap ingest finished successfully."
