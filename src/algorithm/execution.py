"""Alpaca execution guardrails: clock, buying power, asset checks, idempotent orders."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order
from alpaca.trading.requests import MarketOrderRequest

logger = logging.getLogger(__name__)


@dataclass
class OrderAttempt:
    symbol: str
    client_order_id: str
    side: str
    notional: float
    success: bool
    error: Optional[str] = None
    order: Optional[Order] = None
    initial_status: Optional[str] = None
    final_status: Optional[str] = None
    filled_qty: Optional[float] = None
    filled_avg_price: Optional[float] = None
    status_error: Optional[str] = None


@dataclass
class ExecutionReport:
    """Structured result from a rebalance / dry-run."""

    correlation_id: str
    as_of: pd.Timestamp
    data_source: Optional[str]
    panel_tickers: List[str]
    targets_usd: Dict[str, float]
    current_usd: Dict[str, float]
    deltas_usd: Dict[str, float]
    tradecapital_requested: float
    tradecapital_effective_gross: float
    buying_power: Optional[float]
    clock_is_open: Optional[bool]
    dry_run: bool
    order_attempts: List[OrderAttempt] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status_observation_enabled: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly summary (no ORM objects)."""
        return {
            "correlation_id": self.correlation_id,
            "as_of": str(self.as_of),
            "data_source": self.data_source,
            "panel_tickers": list(self.panel_tickers),
            "targets_usd": dict(self.targets_usd),
            "current_usd": dict(self.current_usd),
            "deltas_usd": dict(self.deltas_usd),
            "tradecapital_requested": self.tradecapital_requested,
            "tradecapital_effective_gross": self.tradecapital_effective_gross,
            "buying_power": self.buying_power,
            "clock_is_open": self.clock_is_open,
            "dry_run": self.dry_run,
            "order_attempts": [
                {
                    "symbol": o.symbol,
                    "client_order_id": o.client_order_id,
                    "side": o.side,
                    "notional": o.notional,
                    "success": o.success,
                    "error": o.error,
                    "order_id": str(o.order.id) if o.order and o.order.id else None,
                    "initial_status": o.initial_status,
                    "final_status": o.final_status,
                    "filled_qty": o.filled_qty,
                    "filled_avg_price": o.filled_avg_price,
                    "status_error": o.status_error,
                }
                for o in self.order_attempts
            ],
            "warnings": list(self.warnings),
            "status_observation_enabled": self.status_observation_enabled,
        }


def _fetch_buying_power(client: TradingClient) -> float:
    acct = client.get_account()
    bp = getattr(acct, "buying_power", None)
    if bp is None:
        return 0.0
    return float(bp)


def _clock_open(client: TradingClient) -> bool:
    clock = client.get_clock()
    return bool(clock.is_open)


class AlpacaExecutionAdapter:
    """
    Pre-trade validation and order submission with small retries on transient failures.

    Does **not** compute alpha; operates on final signed dollar deltas per symbol.
    """

    def __init__(
        self,
        client: TradingClient,
        *,
        max_submit_retries: int = 3,
        retry_base_seconds: float = 0.5,
        buying_power_buffer: float = 0.98,
    ) -> None:
        if max_submit_retries < 1:
            raise ValueError("max_submit_retries must be >= 1")
        if not (0.0 < buying_power_buffer <= 1.0):
            raise ValueError("buying_power_buffer must be in (0, 1]")
        self._client = client
        self._max_submit_retries = max_submit_retries
        self._retry_base_seconds = retry_base_seconds
        self._buying_power_buffer = buying_power_buffer

    @property
    def buying_power_fraction(self) -> float:
        """Fraction of Alpaca buying_power used as gross cap (0–1)."""
        return self._buying_power_buffer

    def assert_market_open(self) -> None:
        if not _clock_open(self._client):
            raise RuntimeError("Market is closed; refusing to submit orders (clock guard).")

    def buying_power(self) -> float:
        return _fetch_buying_power(self._client)

    def validate_asset(self, symbol: str, *, delta_usd: float) -> None:
        """Raise if asset is untradable or constraints likely block the order."""
        asset = self._client.get_asset(symbol)
        if not getattr(asset, "tradable", False):
            raise ValueError(f"{symbol!r} is not tradable on Alpaca")
        if getattr(asset, "fractionable", None) is False and delta_usd != 0:
            # Notional market orders often need fractionable US equities
            logger.warning("Asset %s is not marked fractionable; notional order may fail", symbol)
        if delta_usd < 0 and not getattr(asset, "shortable", True):
            logger.warning(
                "%s is not marked shortable on Alpaca; ensure sells only reduce longs or cover shorts",
                symbol,
            )

    def submit_delta_orders(
        self,
        deltas: Dict[str, float],
        *,
        min_order_notional: float,
        dry_run: bool,
        correlation_id: str,
    ) -> List[OrderAttempt]:
        attempts: List[OrderAttempt] = []
        for sym, d in deltas.items():
            ad = abs(float(d))
            if ad < min_order_notional:
                continue
            oid = f"{correlation_id}-{sym}-{uuid.uuid4().hex[:12]}"
            side = OrderSide.BUY if d > 0 else OrderSide.SELL
            if dry_run:
                attempts.append(
                    OrderAttempt(
                        symbol=sym,
                        client_order_id=oid,
                        side=side.value,
                        notional=round(ad, 2),
                        success=True,
                        order=None,
                        initial_status="dry_run",
                        final_status="dry_run",
                    )
                )
                continue
            self.validate_asset(sym, delta_usd=float(d))
            req = MarketOrderRequest(
                symbol=sym,
                notional=round(ad, 2),
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=oid[:48],
            )
            attempt = OrderAttempt(
                symbol=sym,
                client_order_id=oid,
                side=side.value,
                notional=round(ad, 2),
                success=False,
                order=None,
            )
            order: Optional[Order] = None
            last_err: Optional[str] = None
            for attempt_n in range(self._max_submit_retries):
                try:
                    order = self._client.submit_order(req)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt_n < self._max_submit_retries - 1:
                        time.sleep(self._retry_base_seconds * (2**attempt_n))
            if last_err is None and order is not None:
                attempt.success = True
                attempt.order = order
                st = getattr(order, "status", None)
                if st is not None:
                    st_s = str(st)
                    attempt.initial_status = st_s
                    attempt.final_status = st_s
            else:
                attempt.error = last_err
                logger.error("Order failed for %s after retries: %s", sym, last_err)
            attempts.append(attempt)
        return attempts

    def observe_submitted_orders(
        self,
        attempts: List[OrderAttempt],
        *,
        max_polls: int = 3,
        poll_interval_seconds: float = 0.5,
    ) -> List[OrderAttempt]:
        """
        Poll submitted orders to capture latest observed status/fill fields.

        Bounded polling is intended for paper/research visibility, not guaranteed
        fill reconciliation.
        """
        if max_polls < 1:
            return attempts

        terminal_statuses = {
            "filled",
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "done_for_day",
            "stopped",
            "suspended",
        }

        for attempt in attempts:
            if not attempt.success or attempt.order is None:
                continue
            oid = getattr(attempt.order, "id", None)
            if oid is None:
                continue
            last_status: Optional[str] = attempt.final_status
            last_err: Optional[str] = None
            for poll_idx in range(max_polls):
                try:
                    obs = self._client.get_order_by_id(oid)
                    st = getattr(obs, "status", None)
                    if st is not None:
                        last_status = str(st)
                        attempt.final_status = last_status
                    fq = getattr(obs, "filled_qty", None)
                    if fq is not None:
                        attempt.filled_qty = float(fq)
                    fap = getattr(obs, "filled_avg_price", None)
                    if fap is not None:
                        attempt.filled_avg_price = float(fap)
                    last_err = None
                    if last_status in terminal_statuses:
                        break
                except Exception as e:
                    last_err = str(e)

                if poll_idx < max_polls - 1:
                    time.sleep(poll_interval_seconds)
            attempt.status_error = last_err
        return attempts
