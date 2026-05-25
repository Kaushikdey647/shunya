"""Unit tests for :mod:`api.services.alpaca_l1_feed_hub`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from alpaca.data.enums import DataFeed

from api.services.alpaca_l1_feed_hub import (
    AlpacaL1FeedHub,
    SymbolLimitExceeded,
    get_alpaca_l1_hub,
)
from shunya.integration.alpaca_settings import AlpacaRuntimeSettings


def _clear_hubs() -> None:
    import api.services.alpaca_l1_feed_hub as hubm

    with hubm._hubs_lock:
        hubm._hubs.clear()


class _RecordingWs:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _FakeStream:
    def __init__(self) -> None:
        self.unsub_quotes: list[str] = []
        self.unsub_trades: list[str] = []
        self.stopped = False
        self._running = False
        self.quote_subs: list[str] = []
        self.trade_subs: list[str] = []

    def subscribe_quotes(self, _h: Any, *syms: str) -> None:
        self.quote_subs.extend(syms)

    def subscribe_trades(self, _h: Any, *syms: str) -> None:
        self.trade_subs.extend(syms)

    def register_trade_corrections(self, _h: Any) -> None:
        pass

    def register_trade_cancels(self, _h: Any) -> None:
        pass

    def unsubscribe_quotes(self, *syms: str) -> None:
        self.unsub_quotes.extend(syms)

    def unsubscribe_trades(self, *syms: str) -> None:
        self.unsub_trades.extend(syms)

    async def stop_ws(self) -> None:
        self.stopped = True

    async def _run_forever(self) -> None:  # noqa: SLF001
        await asyncio.Future()


@pytest.fixture(autouse=True)
def clear_hubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_ALPACA_L1_RECONNECT_COOLDOWN_SEC", "0")
    _clear_hubs()
    yield
    _clear_hubs()


def test_hub_fanout_two_clients_same_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.services.alpaca_l1_feed_hub as hubm

    async def _body() -> None:
        fake = _FakeStream()
        monkeypatch.setattr(hubm, "build_stock_data_stream", lambda *a, **k: fake)

        rt = AlpacaRuntimeSettings(api_key_id="k1", secret_key="s1", paper=True)
        hub = AlpacaL1FeedHub(rt, feed=DataFeed.IEX, max_symbols=30)
        ws1, ws2 = _RecordingWs(), _RecordingWs()
        await hub.attach(ws1, "AAPL")
        await hub.attach(ws2, "AAPL")
        assert hub._stream is fake  # noqa: SLF001
        q = {"S": "AAPL", "t": "2024-01-01T12:00:00Z", "bp": 1.0, "bs": 1.0, "ap": 2.0, "as": 2.0}
        await hub._on_quote(q)  # noqa: SLF001
        assert len(ws1.sent) == 1 and len(ws2.sent) == 1
        assert ws1.sent[0]["type"] == "quote" and ws1.sent[0]["symbol"] == "AAPL"

        await hub.detach(ws1, "AAPL")
        await hub._on_quote(q)  # noqa: SLF001
        assert len(ws1.sent) == 1 and len(ws2.sent) == 2

        await hub.detach(ws2, "AAPL")
        assert fake.stopped is True

    asyncio.run(_body())


def test_hub_second_symbol_while_alpaca_running_no_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """alpaca-py uses run_coroutine_threadsafe(...).result() when _running — must not block loop."""
    import api.services.alpaca_l1_feed_hub as hubm

    async def _body() -> None:
        fake = _FakeStream()
        monkeypatch.setattr(hubm, "build_stock_data_stream", lambda *a, **k: fake)

        rt = AlpacaRuntimeSettings(api_key_id="k-run", secret_key="s-run", paper=True)
        hub = AlpacaL1FeedHub(rt, feed=DataFeed.IEX, max_symbols=30)
        await hub.attach(_RecordingWs(), "AAPL")
        fake._running = True
        await asyncio.wait_for(hub.attach(_RecordingWs(), "MSFT"), timeout=5.0)
        assert "MSFT" in fake.quote_subs and "MSFT" in fake.trade_subs

    asyncio.run(_body())


def test_hub_broadcasts_error_when_alpaca_runner_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.services.alpaca_l1_feed_hub as hubm

    class _FakeShort(_FakeStream):
        async def _run_forever(self) -> None:  # noqa: SLF001
            return

    async def _body() -> None:
        fake = _FakeShort()
        monkeypatch.setattr(hubm, "build_stock_data_stream", lambda *a, **k: fake)

        rt = AlpacaRuntimeSettings(api_key_id="k-dead", secret_key="s-dead", paper=True)
        hub = AlpacaL1FeedHub(rt, feed=DataFeed.IEX, max_symbols=30)
        ws = _RecordingWs()
        await hub.attach(ws, "AAPL")
        for _ in range(100):
            if hub._stream is None:
                break
            await asyncio.sleep(0.01)
        assert hub._stream is None
        assert hub._runner_task is None
        assert any(m.get("type") == "error" and m.get("code") == "alpaca_market_data_stopped" for m in ws.sent)

    asyncio.run(_body())


def test_hub_symbol_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.services.alpaca_l1_feed_hub as hubm

    async def _body() -> None:
        fake = _FakeStream()
        monkeypatch.setattr(hubm, "build_stock_data_stream", lambda *a, **k: fake)

        rt = AlpacaRuntimeSettings(api_key_id="k2", secret_key="s2", paper=True)
        hub = AlpacaL1FeedHub(rt, feed=DataFeed.IEX, max_symbols=1)
        w1, w2 = _RecordingWs(), _RecordingWs()
        await hub.attach(w1, "AAPL")
        assert await hub.would_reject_new_symbol("MSFT") is True
        with pytest.raises(SymbolLimitExceeded):
            await hub.attach(w2, "MSFT")
        await hub.detach(w1, "AAPL")

    asyncio.run(_body())


def test_get_alpaca_l1_hub_singleton() -> None:
    rt = AlpacaRuntimeSettings(api_key_id="k3", secret_key="s3", paper=True)
    a = get_alpaca_l1_hub(rt, feed=DataFeed.IEX)
    b = get_alpaca_l1_hub(rt, feed=DataFeed.IEX)
    assert a is b


def test_hub_fanout_alpaca_upstream_control(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.services.alpaca_l1_feed_hub as hubm

    async def _body() -> None:
        fake = _FakeStream()
        monkeypatch.setattr(hubm, "build_stock_data_stream", lambda *a, **k: fake)

        rt = AlpacaRuntimeSettings(api_key_id="k-up", secret_key="s-up", paper=True)
        hub = AlpacaL1FeedHub(rt, feed=DataFeed.IEX, max_symbols=30)
        ws = _RecordingWs()
        await hub.attach(ws, "AAPL")

        await hub._on_alpaca_control({"T": "error", "code": 406, "msg": "not allowed"})  # noqa: SLF001
        assert any(
            m.get("type") == "error" and m.get("code") == "alpaca_upstream" and "406" in m.get("message", "")
            for m in ws.sent
        )

        await hub._on_alpaca_control({"T": "subscription", "quotes": ["AAPL"], "trades": ["AAPL"]})  # noqa: SLF001
        assert any(m.get("type") == "subscription" and m.get("alpaca", {}).get("quotes") == ["AAPL"] for m in ws.sent)

    asyncio.run(_body())
