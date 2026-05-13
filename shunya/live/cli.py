"""CLI for one-shot paper cycles (requires Alpaca env keys)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from importlib import import_module
from typing import Any, Optional

import pandas as pd

from shunya.integration.alpaca_settings import (
    build_stock_historical_data_client,
    build_trading_client,
    load_alpaca_settings_from_env,
)
from shunya.live.desk import InstitutionalPaperDesk, new_correlation_id
from shunya.live.demo import build_demo_target_blend_pcs


def _load_pcs_factory(spec: str) -> Any:
    """``module.path:callable`` → callable (no arguments)."""
    if ":" not in spec:
        raise ValueError("Expected import spec 'module:attr'")
    mod_name, attr = spec.split(":", 1)
    mod = import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Shunya paper trading desk (Alpaca)")
    sub = p.add_subparsers(dest="cmd", required=True)

    cyc = sub.add_parser("paper-cycle", help="Run one PCS → risk → OMS → EMS cycle")
    cyc.add_argument("--capital", type=float, default=10_000.0)
    cyc.add_argument("--date", type=str, required=True, help="Execution date YYYY-MM-DD")
    cyc.add_argument("--correlation-id", type=str, default=None)
    cyc.add_argument(
        "--pcs-factory",
        type=str,
        default=None,
        help="Import path 'module:callable' returning PortfolioConstructionService (default: built-in demo)",
    )
    cyc.add_argument("--demo", action="store_true", help="Use built-in two-ticker demo PCS")
    cyc.add_argument("--twap-bins", type=int, default=4)
    cyc.add_argument("--require-market-open", action="store_true")

    args = p.parse_args(argv)
    if args.cmd != "paper-cycle":
        return 2

    settings = load_alpaca_settings_from_env()
    tc = build_trading_client(settings)
    dc = build_stock_historical_data_client(settings)

    if args.demo or not args.pcs_factory:
        pcs = build_demo_target_blend_pcs()
    else:
        fn = _load_pcs_factory(args.pcs_factory)
        pcs = fn()
    desk = InstitutionalPaperDesk(
        tc,
        dc,
        settings,
        twap_bins=args.twap_bins,
        require_market_open=args.require_market_open,
    )
    cid = args.correlation_id or new_correlation_id()
    dt = pd.Timestamp(args.date)

    async def _run() -> Any:
        return await desk.run_with_pcs(pcs, capital=args.capital, execution_date=dt, correlation_id=cid)

    out = asyncio.run(_run())
    json.dump(out.as_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
