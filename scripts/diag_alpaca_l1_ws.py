#!/usr/bin/env python3
"""Open the instrument Alpaca L1 WebSocket and print JSON frames (diagnostics).

Usage (API on localhost:8000)::

    uv run python scripts/diag_alpaca_l1_ws.py http://127.0.0.1:8000 AAPL 30

Expect ``hello`` immediately, then ``quote`` / ``trade``, ``subscription``, or ``error``
(including ``code: alpaca_upstream`` when Alpaca sends a control error).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from urllib.parse import urlparse


def _http_base_to_ws(base: str) -> str:
    u = urlparse(base)
    if u.scheme not in ("http", "https"):
        raise SystemExit(f"unsupported base URL scheme: {u.scheme!r} (use http:// or https://)")
    host = u.netloc or "127.0.0.1"
    path = (u.path or "").rstrip("/")
    scheme = "wss" if u.scheme == "https" else "ws"
    return f"{scheme}://{host}{path}"


async def _run(ws_url: str, seconds: float) -> None:
    try:
        import websockets
    except ImportError as e:  # pragma: no cover
        raise SystemExit("install websockets (e.g. uv sync with api extra)") from e

    print(f"connecting {ws_url!r} …", file=sys.stderr)
    async with websockets.connect(ws_url, max_size=None) as ws:

        async def reader() -> None:
            async for raw in ws:
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    print(raw[:500])
                    continue
                print(json.dumps(obj, indent=2, default=str))

        read_task = asyncio.create_task(reader())
        try:
            await asyncio.wait_for(asyncio.sleep(seconds), timeout=seconds + 1)
        finally:
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "base",
        help="API base URL, e.g. http://127.0.0.1:8000",
    )
    p.add_argument("symbol", help="Ticker, e.g. AAPL")
    p.add_argument(
        "seconds",
        nargs="?",
        type=float,
        default=30.0,
        help="How long to read frames (default: 30)",
    )
    args = p.parse_args()
    sym = args.symbol.strip().upper()
    path = f"/instruments/{sym}/stream/alpaca-l1"
    ws_root = _http_base_to_ws(args.base)
    ws_url = f"{ws_root}{path}"
    asyncio.run(_run(ws_url, max(1.0, args.seconds)))


if __name__ == "__main__":
    main()
