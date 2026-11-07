"""Alpaca access for the live runner; every order submit re-checks the kill switch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient

    from shunya.live.store import Control

_log = logging.getLogger(__name__)

TZ_NY = ZoneInfo("America/New_York")


class KillSwitchEngaged(RuntimeError):
    """Raised when an order is attempted while the kill switch is not armed."""


@dataclass(frozen=True)
class Session:
    """One exchange trading day with tz-aware open/close."""

    day: date
    open: datetime
    close: datetime


@dataclass(frozen=True)
class AccountView:
    """Subset of the Alpaca account used by the runner and status page."""

    equity: float
    last_equity: float
    cash: float
    buying_power: float
    trading_blocked: bool
    status: str


@dataclass(frozen=True)
class PositionView:
    """Broker position (signed shares)."""

    symbol: str
    qty: float
    market_value: float
    current_price: float
    avg_entry_price: float


@dataclass(frozen=True)
class OrderView:
    """Broker order state."""

    client_order_id: str
    alpaca_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    time_in_force: str


class Broker(Protocol):
    """What the runner needs from a broker (Alpaca in production, fakes in tests)."""

    def clock(self) -> dict:
        """Exchange clock snapshot."""
        ...

    def sessions(self, start: date, end: date) -> List[Session]:
        """Trading days in ``[start, end]``."""
        ...

    def account(self) -> AccountView:
        """Equity, cash, buying power."""
        ...

    def positions(self) -> List[PositionView]:
        """Open positions."""
        ...

    def open_orders(self) -> List[OrderView]:
        """Working orders."""
        ...

    def order_by_client_id(self, client_order_id: str) -> Optional[OrderView]:
        """Order by our id, or ``None``."""
        ...

    def submit_opg(self, *, symbol: str, side: str, qty: int, client_order_id: str) -> OrderView:
        """Submit a market-on-open order (must honour the kill switch)."""
        ...

    def cancel_all(self) -> int:
        """Cancel all open orders."""
        ...

    def close_all(self) -> int:
        """Liquidate all positions."""
        ...


def _f(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _enum_value(v: object) -> str:
    return str(getattr(v, "value", v))


def _order_view(o: object) -> OrderView:
    fap = getattr(o, "filled_avg_price", None)
    return OrderView(
        client_order_id=str(getattr(o, "client_order_id", "")),
        alpaca_id=str(getattr(o, "id", "")),
        symbol=str(getattr(o, "symbol", "")),
        side=_enum_value(getattr(o, "side", "")),
        qty=_f(getattr(o, "qty", 0)),
        status=_enum_value(getattr(o, "status", "")),
        filled_qty=_f(getattr(o, "filled_qty", 0)),
        filled_avg_price=_f(fap) if fap is not None else None,
        time_in_force=_enum_value(getattr(o, "time_in_force", "")),
    )


def _as_ny(d: date, v: object) -> datetime:
    """Alpaca calendar open/close come back naive (exchange local) or as ``HH:MM`` strings."""
    if isinstance(v, datetime):
        return v.replace(tzinfo=TZ_NY) if v.tzinfo is None else v.astimezone(TZ_NY)
    if isinstance(v, time):
        return datetime.combine(d, v, tzinfo=TZ_NY)
    hh, mm = str(v).split(":")[:2]
    return datetime.combine(d, time(int(hh), int(mm)), tzinfo=TZ_NY)


class AlpacaBroker:
    """:class:`Broker` over ``alpaca-py``; ``control_fn`` is consulted before every submit."""

    def __init__(self, client: "TradingClient", control_fn: Callable[[], "Control"]) -> None:
        self._c = client
        self._control_fn = control_fn

    def clock(self) -> dict:
        """Exchange clock for status pages: ``timestamp``, ``is_open``, ``next_open``, ``next_close``."""
        c = self._c.get_clock()
        return {
            "timestamp": c.timestamp.isoformat(),
            "is_open": bool(c.is_open),
            "next_open": c.next_open.isoformat(),
            "next_close": c.next_close.isoformat(),
        }

    def sessions(self, start: date, end: date) -> List[Session]:
        """Trading days in ``[start, end]`` with real open/close (handles holidays and half-days)."""
        from alpaca.trading.requests import GetCalendarRequest

        out: List[Session] = []
        for c in self._c.get_calendar(GetCalendarRequest(start=start, end=end)):
            d = c.date if isinstance(c.date, date) else date.fromisoformat(str(c.date))
            out.append(Session(day=d, open=_as_ny(d, c.open), close=_as_ny(d, c.close)))
        return out

    def account(self) -> AccountView:
        """Equity, cash, buying power."""
        a = self._c.get_account()
        return AccountView(
            equity=_f(a.equity),
            last_equity=_f(a.last_equity),
            cash=_f(a.cash),
            buying_power=_f(a.buying_power),
            trading_blocked=bool(a.trading_blocked),
            status=_enum_value(a.status),
        )

    def positions(self) -> List[PositionView]:
        """All open positions."""
        return [
            PositionView(
                symbol=p.symbol,
                qty=_f(p.qty),
                market_value=_f(p.market_value),
                current_price=_f(p.current_price),
                avg_entry_price=_f(p.avg_entry_price),
            )
            for p in self._c.get_all_positions()
        ]

    def open_orders(self) -> List[OrderView]:
        """Working (open) orders."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._c.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return [_order_view(o) for o in orders]

    def order_by_client_id(self, client_order_id: str) -> Optional[OrderView]:
        """Order by our id, ``None`` when Alpaca does not know it."""
        from alpaca.common.exceptions import APIError

        try:
            return _order_view(self._c.get_order_by_client_id(client_order_id))
        except APIError as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def submit_opg(self, *, symbol: str, side: str, qty: int, client_order_id: str) -> OrderView:
        """Market-on-open order (whole shares). Duplicate ``client_order_id`` returns the existing order."""
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        ctl = self._control_fn()
        if ctl.halted:
            raise KillSwitchEngaged(f"kill switch {ctl.level or 'halt'}: {ctl.reason or ''}".strip())
        if qty <= 0 or int(qty) != qty:
            raise ValueError(f"OPG orders need a positive whole share qty, got {qty!r}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.OPG,
            client_order_id=client_order_id,
        )
        try:
            return _order_view(self._c.submit_order(req))
        except APIError as exc:
            if "client_order_id must be unique" in str(exc).lower():
                existing = self.order_by_client_id(client_order_id)
                if existing is not None:
                    _log.info("order %s already submitted; reusing", client_order_id)
                    return existing
            raise

    def cancel_all(self) -> int:
        """Cancel every open order; return how many cancel requests Alpaca accepted."""
        return len(self._c.cancel_orders() or [])

    def close_all(self) -> int:
        """Liquidate every position (cancels open orders first)."""
        return len(self._c.close_all_positions(cancel_orders=True) or [])
