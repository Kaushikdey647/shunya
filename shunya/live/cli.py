"""``shunya-live``: migrate, cron tick, dry-run, kill / resume, status."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def _broker_and_store() -> tuple[Any, Any]:
    from shunya.integration.alpaca_settings import build_trading_client, load_alpaca_settings_from_env
    from shunya.live.broker import AlpacaBroker
    from shunya.live.control import read_control
    from shunya.live.store import PostgresLiveStore

    store = PostgresLiveStore()
    settings = load_alpaca_settings_from_env()
    if not settings.paper and os.environ.get("SHUNYA_LIVE_ALLOW_REAL_MONEY") != "1":
        raise SystemExit("Refusing live (real-money) Alpaca keys; set SHUNYA_LIVE_ALLOW_REAL_MONEY=1.")
    client = build_trading_client(settings)
    return AlpacaBroker(client, lambda: read_control(store)), store


def _dump(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch ``shunya-live`` subcommands."""
    p = argparse.ArgumentParser(prog="shunya-live", description="Shunya daily live runner (Alpaca)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="Apply live_* table migrations")
    t = sub.add_parser("tick", help="One scheduler firing (run from cron every ~15 min)")
    t.add_argument("--now", default=None, help="Override current time (ISO 8601, for testing)")
    d = sub.add_parser("plan", help="Dry-run: compute targets and orders for a book without submitting")
    d.add_argument("--book", required=True, help="Book id")
    h = sub.add_parser("halt", help="Engage the kill switch")
    h.add_argument("--level", choices=("halt", "cancel", "flatten"), default="cancel")
    h.add_argument("--reason", required=True)
    r = sub.add_parser("resume", help="Re-arm trading")
    r.add_argument("--reason", required=True)
    sub.add_parser("status", help="Kill switch, heartbeat, account, books, ledger drift")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "migrate":
        from shunya.live.store import apply_live_migrations

        apply_live_migrations()
        return 0

    broker, store = _broker_and_store()
    if args.cmd == "tick":
        from shunya.live.runner import tick

        now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        out = tick(store, broker, now=now)
        _dump(out)
        return 0 if out.get("status") in ("ok", "halted") else 1
    if args.cmd == "plan":
        from shunya.live.runner import RunnerConfig, plan_book, plan_sessions

        book = store.get_book(args.book)
        if book is None:
            print(f"book {args.book} not found", file=sys.stderr)
            return 2
        plan = plan_sessions(broker, datetime.now(timezone.utc))
        bp = plan_book(
            book, store.list_sleeves(book.id), store.get_sleeve_positions(book.id), broker, plan, cfg=RunnerConfig()
        )
        _dump({"signal_date": plan.signal.day, "session_date": plan.next.day, **bp.summary()})
        return 0
    if args.cmd == "halt":
        from shunya.live.control import engage_kill

        _dump(engage_kill(store, broker, level=args.level, reason=args.reason, actor="cli"))
        return 0
    if args.cmd == "resume":
        from shunya.live.control import resume

        resume(store, reason=args.reason, actor="cli")
        return 0
    if args.cmd == "status":
        from shunya.live.status import build_status

        _dump(build_status(store, broker))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
