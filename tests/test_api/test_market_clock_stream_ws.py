"""Market clock WebSocket: hello, immediate tick, and hub fan-out."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


async def _quiet_worker(stop: asyncio.Event) -> None:
    await stop.wait()


def test_market_clock_stream_hello_and_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.main.backtest_worker_loop", _quiet_worker)
    monkeypatch.setenv("SHUNYA_MARKET_CLOCK_TICK_SECONDS", "5")

    with TestClient(create_app()) as client:
        with client.websocket_connect("/settings/market-clock/stream") as ws:
            h = ws.receive_json()
            assert h["type"] == "hello"
            assert h["schema"] == 1
            t = ws.receive_json()
            assert t["type"] == "tick"
            assert t["schema"] == 1
            assert "utc_iso" in t and "us_line" in t and "in_line" in t
            assert "us_listed_rth_open" in t and "alpaca_l1_us_equities_stream_allowed" in t


def test_market_clock_stream_receives_hub_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub loop broadcasts ticks; subscriber should see a second frame after hello + immediate."""
    monkeypatch.setattr("api.main.backtest_worker_loop", _quiet_worker)
    monkeypatch.setenv("SHUNYA_MARKET_CLOCK_TICK_SECONDS", "0.25")

    with TestClient(create_app()) as client:
        with client.websocket_connect("/settings/market-clock/stream") as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json()["type"] == "tick"
            time.sleep(0.35)
            t2 = ws.receive_json()
            assert t2["type"] == "tick"
            assert t2["schema"] == 1


def test_market_clock_internal_queue_receives_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.main.backtest_worker_loop", _quiet_worker)
    monkeypatch.setenv("SHUNYA_MARKET_CLOCK_TICK_SECONDS", "0.25")

    async def consume() -> str:
        from api.services.market_clock_hub import (
            create_market_clock_subscription,
            release_market_clock_subscription,
        )

        q = await create_market_clock_subscription()
        try:
            snap = await asyncio.wait_for(q.get(), timeout=3.0)
            return snap.utc_iso
        finally:
            await release_market_clock_subscription(q)

    with TestClient(create_app()) as client:
        utc_iso = client.portal.call(consume)

    assert isinstance(utc_iso, str) and len(utc_iso) > 10
