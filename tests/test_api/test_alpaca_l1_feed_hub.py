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

    def subscribe_quotes(self, _h: Any, *_syms: str) -> None:
        pass

    def subscribe_trades(self, _h: Any, *_syms: str) -> None:
        pass

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
def clear_hubs() -> None:
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
