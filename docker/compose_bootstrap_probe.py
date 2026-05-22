#!/usr/bin/env python3
"""
Exit status for Docker Compose one-shot bootstrap (see docker/compose-bootstrap.sh).

- 0: DB already has SP100-scale daily yfinance OHLCV — skip ingest scripts.
- 1: Run bootstrap scripts.
- 2: Unexpected error (connection, missing env).

Criteria match ``scripts/bootstrap_sp100_timescale.py`` defaults: ``interval`` = ``1d``,
``source`` = ``STORED_OHLCV_DEFAULT_UPSTREAM_ID`` (Yahoo OHLCV rows in ``ohlcv_bars``).
Requires benchmark ``^OEX`` bars and at least
``SHUNYA_COMPOSE_BOOTSTRAP_MIN_SP100_BARS`` distinct SP100 members with at least one
such bar (default 50).
"""

from __future__ import annotations

import os
import sys

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID

# Keep in sync with scripts/bootstrap_sp100_timescale.py
_DEFAULT_INTERVAL = "1d"
_DEFAULT_SOURCE = STORED_OHLCV_DEFAULT_UPSTREAM_ID
_BENCHMARK = "^OEX"
_DEFAULT_MIN_SP100 = 50


def main() -> int:
    url = (os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL") or "").strip()
    if not url:
        print("compose_bootstrap_probe: set DATABASE_URL or SHUNYA_DATABASE_URL", file=sys.stderr)
        return 2

    try:
        min_sp100 = int(os.environ.get("SHUNYA_COMPOSE_BOOTSTRAP_MIN_SP100_BARS", str(_DEFAULT_MIN_SP100)))
    except ValueError:
        print("compose_bootstrap_probe: SHUNYA_COMPOSE_BOOTSTRAP_MIN_SP100_BARS must be an int", file=sys.stderr)
        return 2

    try:
        import psycopg  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        print("compose_bootstrap_probe: psycopg required", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        with psycopg.connect(url, connect_timeout=30) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'ohlcv_bars'
                    )
                    """
                )
                if not cur.fetchone()[0]:
                    print("compose_bootstrap_probe: ohlcv_bars missing — run bootstrap")
                    return 1
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'symbol_index_membership'
                    )
                    """
                )
                if not cur.fetchone()[0]:
                    print("compose_bootstrap_probe: symbol_index_membership missing — run bootstrap")
                    return 1

                cur.execute(
                    """
                    SELECT COUNT(*)::bigint
                    FROM ohlcv_bars ob
                    JOIN symbols s ON s.id = ob.symbol_id
                    WHERE s.ticker = %s AND ob.interval = %s AND ob.source = %s
                    """,
                    (_BENCHMARK, _DEFAULT_INTERVAL, _DEFAULT_SOURCE),
                )
                bench_count = cur.fetchone()[0]
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT sim.symbol_id)::bigint
                    FROM symbol_index_membership sim
                    INNER JOIN ohlcv_bars ob ON ob.symbol_id = sim.symbol_id
                    WHERE sim.index_code = 'SP100'
                      AND ob.interval = %s
                      AND ob.source = %s
                    """,
                    (_DEFAULT_INTERVAL, _DEFAULT_SOURCE),
                )
                sp100_with_bars = cur.fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        print(f"compose_bootstrap_probe: error: {exc}", file=sys.stderr)
        return 2

    populated = bench_count > 0 and sp100_with_bars >= min_sp100
    if populated:
        print(
            f"compose_bootstrap_probe: populated "
            f"(benchmark {_BENCHMARK!r} bars={bench_count}, "
            f"SP100 symbols with {_DEFAULT_INTERVAL!r}/{_DEFAULT_SOURCE!r} bars={sp100_with_bars}, "
            f"min_required={min_sp100}) — skip ingest"
        )
        return 0

    print(
        f"compose_bootstrap_probe: not populated "
        f"(benchmark bars={bench_count}, SP100_with_bars={sp100_with_bars}, min_required={min_sp100})"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
