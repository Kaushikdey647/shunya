"""Async EMS worker: schedule child limit orders for one parent."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List

from shunya.ems.broker_gateway import BrokerGateway
from shunya.ems.ids import child_client_order_id
from shunya.ems.micro_price import MicroPriceUrgency, limit_price_for_child

logger = logging.getLogger(__name__)


@dataclass
class EMSParentRunner:
    """
    Execute a TWAP/VWAP-style slice list against a :class:`BrokerGateway`.

    Each slice submits a DAY limit, waits ``child_timeout_seconds``, cancels any
    remainder, then escalates limit price (mid → touch → cross) on the **unfilled**
    slice quantity only.
    """

    gateway: BrokerGateway
    parent_id: str
    symbol: str
    side: str
    slice_quantities: List[int]
    interval_seconds: float = 60.0
    child_timeout_seconds: float = 45.0
    poll_interval_seconds: float = 2.0
    urgencies: tuple[MicroPriceUrgency, ...] = (
        MicroPriceUrgency.MID,
        MicroPriceUrgency.PASSIVE_TOUCH,
        MicroPriceUrgency.CROSS,
    )
    _tasks: List[asyncio.Task[None]] = field(default_factory=list, repr=False)

    async def run(self) -> None:
        """Run all slices sequentially (interval spacing between slices)."""
        for i, qty in enumerate(self.slice_quantities):
            if i > 0:
                await asyncio.sleep(float(self.interval_seconds))
            if qty <= 0:
                continue
            await self._execute_slice(i, int(qty))

    async def _execute_slice(self, slice_index: int, qty: int) -> None:
        remaining = int(qty)
        for attempt, urgency in enumerate(self.urgencies):
            if remaining <= 0:
                return
            cid = child_client_order_id(self.parent_id, slice_index, attempt)
            quote = await self.gateway.get_latest_quote(self.symbol)
            if quote.bid <= 0 and quote.ask <= 0:
                logger.warning("EMS: no quote for %s; skipping slice %s", self.symbol, slice_index)
                return
            lim = limit_price_for_child(quote, side=self.side, urgency=urgency)
            oid = await self.gateway.submit_limit_order(
                symbol=self.symbol,
                side=self.side,
                qty=remaining,
                limit_price=lim,
                client_order_id=cid,
            )
            if not oid:
                continue
            loop = asyncio.get_event_loop()
            deadline = loop.time() + float(self.child_timeout_seconds)
            while loop.time() < deadline:
                await asyncio.sleep(float(self.poll_interval_seconds))
                open_ids = await self.gateway.get_open_order_ids_for_client_order_id(cid)
                if not open_ids:
                    return
            filled = 0.0
            try:
                filled = await self.gateway.get_order_filled_qty(oid)
            except Exception:  # noqa: BLE001
                logger.debug("EMS: could not read fill snapshot for %s", oid, exc_info=True)
            filled_i = int(min(remaining, max(0.0, filled)))
            remaining = max(0, remaining - filled_i)
            for o in await self.gateway.get_open_order_ids_for_client_order_id(cid):
                try:
                    await self.gateway.cancel_order_by_id(o)
                except Exception:  # noqa: BLE001
                    logger.debug("EMS: cancel failed for %s", o, exc_info=True)

    def start_background(self) -> asyncio.Task[None]:
        t = asyncio.create_task(self.run(), name=f"ems-parent-{self.parent_id}")
        self._tasks.append(t)
        return t

    async def wait_all(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks)
            self._tasks.clear()
