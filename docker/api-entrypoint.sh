#!/bin/sh
set -e
cd /app
if [ "${RUN_MIGRATIONS:-1}" = "1" ]; then
  uv run shunya-timescale migrate
fi
exec "$@"
