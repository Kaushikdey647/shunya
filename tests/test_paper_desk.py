"""Tests for :mod:`shunya.live.desk` (broker mocked; no network)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from shunya.integration.alpaca_settings import AlpacaRuntimeSettings
from shunya.live.desk import InstitutionalPaperDesk


class _FakeStream:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        self._task = None

    def start_background(self, loop) -> None:  # noqa: ANN001
        return

    async def stop(self) -> None:
        return


class _FakeEMS:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        pass

    async def run(self) -> None:
        return


@pytest.fixture
def mock_trading_client() -> MagicMock:
    tc = MagicMock()
    tc.get_all_positions.return_value = []
    acct = MagicMock()
    acct.equity = "100000"
    acct.buying_power = "100000"
    tc.get_account.return_value = acct
    tc.get_orders.return_value = []
    return tc


def test_run_with_targets_no_intents_after_vet(monkeypatch: pytest.MonkeyPatch, mock_trading_client: MagicMock) -> None:
    monkeypatch.setattr("shunya.live.desk.AlpacaOMSTradeStream", _FakeStream)
    monkeypatch.setattr("shunya.live.desk.EMSParentRunner", _FakeEMS)
    settings = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    dc = MagicMock()
    desk = InstitutionalPaperDesk(mock_trading_client, dc, settings, twap_bins=2)

    async def _go() -> None:
        res = await desk.run_with_targets(
            {"SPY": 0.0},
            universe=["SPY"],
            prices={"SPY": 450.0},
            correlation_id="t1",
        )
        assert res.correlation_id == "t1"
        assert res.parent_intents == []

    asyncio.run(_go())
