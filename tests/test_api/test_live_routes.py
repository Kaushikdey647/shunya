"""Hosted live API (token, kill/resume, books/sleeves, deploy validation) and local bridge fallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from api.live_main import create_live_app
from shunya.live.control import read_control
from tests.test_live.fakes import FakeBroker, MemStore

ALPHA = "import jax.numpy as jnp\n\ndef alpha(ctx):\n    return ctx.close.latest.astype(jnp.float32)\n"
H = {"X-Shunya-Live-Token": "live-tok"}


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("SHUNYA_LIVE_TOKEN", "live-tok")
    store = MemStore()
    broker = FakeBroker(lambda: read_control(store))
    with TestClient(create_live_app(store=store, broker=broker)) as c:
        yield c, store, broker


def test_token_required(hosted, monkeypatch) -> None:
    c, _, _ = hosted
    assert c.get("/healthz").status_code == 200
    assert c.get("/live/status").status_code == 401
    assert c.get("/live/status", headers={"X-Shunya-Live-Token": "nope"}).status_code == 401
    monkeypatch.delenv("SHUNYA_LIVE_TOKEN")
    assert c.get("/live/status", headers=H).status_code == 503


def test_kill_and_resume(hosted) -> None:
    c, store, broker = hosted
    broker.pos = {"AAA": 5}
    r = c.post("/live/kill", headers=H, json={"level": "flatten", "reason": "test"})
    assert r.status_code == 200 and r.json()["positions_closed"] == 1
    st = c.get("/live/status", headers=H).json()
    assert st["control"]["state"] == "halted" and st["control"]["level"] == "flatten"
    assert c.post("/live/resume", headers=H, json={"reason": "ok"}).json() == {"state": "armed"}
    assert not store.control.halted
    assert [a["action"] for a in c.get("/live/audit", headers=H).json()] == ["resume", "kill"]


def test_books_sleeves_and_versions(hosted) -> None:
    c, store, _ = hosted
    bad = c.post("/live/strategy-versions", headers=H,
                 json={"name": "x", "alpha_source": ALPHA, "finstrat_config": {"neutralization": "sector"}, "tickers": ["A"]})
    assert bad.status_code == 422
    vid = c.post("/live/strategy-versions", headers=H,
                 json={"name": "close", "alpha_source": ALPHA, "tickers": ["aapl", " msft "]}).json()["id"]
    assert store.versions[vid].tickers == ["AAPL", "MSFT"]

    bid = c.post("/live/books", headers=H, json={"name": "main", "capital_fraction": 0.5}).json()["id"]
    assert c.post("/live/books", headers=H, json={"name": "second"}).status_code == 409
    sid = c.post(f"/live/books/{bid}/sleeves", headers=H, json={"version_id": vid, "weight": 0.7}).json()["id"]
    over = c.post(f"/live/books/{bid}/sleeves", headers=H, json={"version_id": vid, "weight": 0.5})
    assert over.status_code == 422
    assert c.patch(f"/live/books/{bid}/sleeves/{sid}", headers=H, json={"status": "paused"}).status_code == 200
    books = c.get("/live/books", headers=H).json()
    assert books[0]["sleeves"][0]["status"] == "paused"
    st = c.get("/live/status", headers=H).json()
    assert st["books"][0]["sleeves"][0]["name"] == "close" and st["errors"] == []


# --- local bridge -------------------------------------------------------------------------


@pytest.fixture
def local(monkeypatch):
    from api.main import create_app
    from api.settings import get_settings

    async def _no_worker(stop):  # noqa: ARG001
        return None

    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "desk")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setenv("SHUNYA_LIVE_URL", "http://hosted.invalid")
    monkeypatch.setenv("SHUNYA_LIVE_TOKEN", "live-tok")
    monkeypatch.setattr("api.main.backtest_worker_loop", _no_worker)
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        yield c, app
    get_settings.cache_clear()


def _hosted_down(*a, **k):
    raise httpx.ConnectError("refused")


def test_bridge_kill_falls_back_to_direct_alpaca(local, monkeypatch) -> None:
    c, app = local
    monkeypatch.setattr("api.routers.live_bridge.httpx.request", _hosted_down)
    client = MagicMock()
    client.cancel_orders.return_value = [1, 2]
    client.close_all_positions.return_value = [1]
    app.state.trade_desk_runtime = MagicMock(trading_client=client)
    r = c.post("/live/kill", headers={"X-Shunya-Trade-Desk-Token": "desk"}, json={"level": "flatten", "reason": "x"})
    body = r.json()
    assert r.status_code == 200 and body["route"] == "direct_alpaca"
    assert body["orders_cancelled"] == 2 and body["positions_closed"] == 1
    client.close_all_positions.assert_called_once_with(cancel_orders=True)


def test_bridge_kill_uses_hosted_when_up(local, monkeypatch) -> None:
    c, app = local
    calls = []

    def fake_request(method, url, **kw):
        calls.append((method, url, kw["headers"]))
        return httpx.Response(200, json={"level": "cancel", "flag_set": True, "orders_cancelled": 0})

    monkeypatch.setattr("api.routers.live_bridge.httpx.request", fake_request)
    r = c.post("/live/kill", headers={"X-Shunya-Trade-Desk-Token": "desk"}, json={"level": "cancel", "reason": "x"})
    assert r.json()["route"] == "hosted"
    assert calls == [("POST", "http://hosted.invalid/live/kill", {"X-Shunya-Live-Token": "live-tok"})]


def test_bridge_requires_desk_token_and_reports_hosted_down(local, monkeypatch) -> None:
    c, _ = local
    assert c.get("/live/status").status_code == 401
    monkeypatch.setattr("api.routers.live_bridge.httpx.request", _hosted_down)
    st = c.get("/live/status", headers={"X-Shunya-Trade-Desk-Token": "desk"}).json()
    assert st["hosted_reachable"] is False and "unreachable" in st["hosted_error"]
    assert c.get("/live/books", headers={"X-Shunya-Trade-Desk-Token": "desk"}).status_code == 502
