"""Alpaca trading WebSocket bridge → OMS (:class:`~shunya.oms.service.InstitutionalOMS`)."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from shunya.integration.alpaca_settings import (
    AlpacaRuntimeSettings,
    build_trading_stream,
    load_alpaca_settings_from_env,
)
from shunya.oms.service import InstitutionalOMS

logger = logging.getLogger(__name__)


class AlpacaOMSTradeStream:
    """
    Wraps :class:`alpaca.trading.stream.TradingStream` and forwards updates to an OMS.

    Uses Alpaca's internal ``_run_forever`` coroutine so the stream shares the app's
    asyncio loop (``TradingStream.run`` calls ``asyncio.run`` and must not be nested).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        *,
        paper: bool = True,
        oms: Optional[InstitutionalOMS] = None,
        settings: Optional[AlpacaRuntimeSettings] = None,
    ) -> None:
        if settings is not None:
            rt = settings
        elif api_key and secret_key:
            rt = AlpacaRuntimeSettings(api_key_id=api_key, secret_key=secret_key, paper=paper)
        else:
            rt = load_alpaca_settings_from_env(default_paper=paper)
        self._oms = oms
        self._stream = build_trading_stream(rt)
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def oms(self) -> Optional[InstitutionalOMS]:
        return self._oms

    @oms.setter
    def oms(self, value: Optional[InstitutionalOMS]) -> None:
        self._oms = value

    async def _on_trade(self, data: object) -> None:
        if self._oms is None:
            return
        self._oms.apply_alpaca_trade_update(data)

    def start_background(self, loop: asyncio.AbstractEventLoop) -> None:
        """Subscribe and run the websocket loop as an asyncio task on ``loop``."""
        self._stream.subscribe_trade_updates(self._on_trade)
        self._task = loop.create_task(self._stream._run_forever(), name="alpaca-oms-trade-stream")  # noqa: SLF001

    async def stop(self) -> None:
        await self._stream.stop_ws()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
