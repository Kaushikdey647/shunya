"""Async broker access for EMS (quotes + limit submit/cancel)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from .micro_price import QuoteL1

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient


@runtime_checkable
class BrokerGateway(Protocol):
    """Minimal EMS-facing broker surface (REST; async wrappers)."""

    async def get_latest_quote(self, symbol: str) -> QuoteL1: ...

    async def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
        time_in_force: str = "day",
    ) -> str: ...

    async def cancel_order_by_id(self, order_id: str) -> None: ...

    async def get_open_order_ids_for_client_order_id(self, client_order_id: str) -> list[str]: ...

    async def get_order_filled_qty(self, order_id: str) -> float: ...


class AlpacaBrokerGateway:
    """Alpaca :class:`~alpaca.trading.client.TradingClient` + data client wrapped for EMS."""

    def __init__(
        self,
        trading_client: "TradingClient",
        data_client: "StockHistoricalDataClient",
    ) -> None:
        self._trading = trading_client
        self._data = data_client

    async def get_latest_quote(self, symbol: str) -> QuoteL1:
        from alpaca.data.requests import StockLatestQuoteRequest

        def _sync() -> QuoteL1:
            qmap = self._data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
            if isinstance(qmap, dict):
                q = qmap[symbol]
            else:
                q = getattr(qmap, symbol, qmap)
            bid = float(getattr(q, "bid_price", 0.0) or 0.0)
            ask = float(getattr(q, "ask_price", 0.0) or 0.0)
            return QuoteL1(bid=bid, ask=ask)

        return await asyncio.to_thread(_sync)

    async def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
        time_in_force: str = "day",
    ) -> str:
        s = OrderSide.BUY if str(side).upper() == "BUY" else OrderSide.SELL
        tif = TimeInForce.DAY
        if str(time_in_force).lower() == "gtc":
            tif = TimeInForce.GTC

        def _sync() -> str:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=s,
                limit_price=round(float(limit_price), 2),
                time_in_force=tif,
                client_order_id=client_order_id,
            )
            o = self._trading.submit_order(req)
            return str(getattr(o, "id", "") or "")

        return await asyncio.to_thread(_sync)

    async def cancel_order_by_id(self, order_id: str) -> None:
        def _sync() -> None:
            self._trading.cancel_order_by_id(order_id)

        await asyncio.to_thread(_sync)

    async def get_open_order_ids_for_client_order_id(self, client_order_id: str) -> list[str]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        def _sync() -> list[str]:
            orders = self._trading.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
            )
            out: list[str] = []
            for o in orders:
                if str(getattr(o, "client_order_id", "") or "") == client_order_id:
                    oid = getattr(o, "id", None)
                    if oid is not None:
                        out.append(str(oid))
            return out

        return await asyncio.to_thread(_sync)

    async def get_order_filled_qty(self, order_id: str) -> float:
        def _sync() -> float:
            o = self._trading.get_order_by_id(order_id)
            return float(getattr(o, "filled_qty", 0.0) or 0.0)

        return await asyncio.to_thread(_sync)
