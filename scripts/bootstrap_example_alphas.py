#!/usr/bin/env python3
"""
Insert example alphas from ``examples.alphas.ALL_ALPHAS`` into ``api_alphas``.

After ``shunya-timescale migrate`` and with ``DATABASE_URL`` set, these rows
show up on the Shunya UI alphas list (the UI calls ``GET /alphas`` on the
backtest API, which reads the same table).

Requires: ``uv sync --extra api --extra timescale`` from the repo root.

Example::

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
    uv run python scripts/bootstrap_example_alphas.py
    uv run python scripts/bootstrap_example_alphas.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Short descriptions aligned with ``examples/alphas/README.md`` where applicable.
_ALPHA_DESCRIPTIONS: dict[str, str] = {
    "sma_ratio_20": "Trend: close / SMA(20).",
    "mean_reversion_20": "Short-term mean reversion (20-bar window).",
    "breakout_20": "20-bar momentum via delayed-close ratio.",
    "sma20_deviation_rank": "Inverted cross-sectional rank of close − SMA(20), scaled to ~[−1, 1].",
    "sma20_deviation_zscore": "Inverted cross-sectional z-score of close − SMA(20).",
    "fundamental_quality_yf": "Fundamentals-based signal (yfinance path); see examples notebook.",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not connect to the database.",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of ALL_ALPHAS keys (default: all).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from examples.alphas import ALL_ALPHAS

    keys = sorted(ALL_ALPHAS.keys())
    if args.only.strip():
        want = {k.strip() for k in args.only.split(",") if k.strip()}
        unknown = want - set(keys)
        if unknown:
            print(f"Unknown keys (not in ALL_ALPHAS): {sorted(unknown)}", file=sys.stderr)
            return 2
        keys = [k for k in keys if k in want]

    rows: list[tuple[str, str, str]] = []
    for key in keys:
        import_ref = f"examples.alphas.{key}:alpha"
        desc = _ALPHA_DESCRIPTIONS.get(key)
        rows.append((key, import_ref, desc or ""))

    if args.dry_run:
        for name, ref, desc in rows:
            extra = f" — {desc}" if desc else ""
            print(f"would insert: name={name!r} import_ref={ref!r}{extra}")
        print(f"dry-run: {len(rows)} alpha(s)")
        return 0

    from backtest_api.repositories.alphas import insert_alpha, list_alphas
    from backtest_api.schemas.models import AlphaCreate

    existing = {a.name for a in list_alphas(limit=5000, offset=0)}
    inserted = 0
    skipped = 0
    for name, import_ref, desc in rows:
        if name in existing:
            print(f"skip (exists): {name}")
            skipped += 1
            continue
        body = AlphaCreate(
            name=name,
            description=desc or None,
            import_ref=import_ref,
        )
        out = insert_alpha(body)
        print(f"inserted: {name} id={out.id}")
        inserted += 1
        existing.add(name)

    print(f"done: inserted={inserted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
