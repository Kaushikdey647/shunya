"""
Kite Connect v5 execution adapter (KiteTrigger).

Translates :class:`~shunya.algorithm.orders.OrderSpec` into ``kiteconnect.KiteConnect``
``place_order`` calls with support for Regular, Cover Order, AMO, Iceberg, and Auction
varieties, plus Market, Limit, SL, and SL-M order types.

Requires ``kiteconnect>=5.0`` (optional dependency).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .execution import OrderAttempt
from .orders import ExecutionAdapter, OrderSide, OrderSpec, OrderType, OrderVariety

logger = logging.getLogger(__name__)

_VARIETY_MAP: Dict[OrderVariety, str] = {
    OrderVariety.REGULAR: "regular",
    OrderVariety.COVER: "co",
    OrderVariety.AMO: "amo",
    OrderVariety.ICEBERG: "iceberg",
    OrderVariety.AUCTION: "auction",
}

_ORDER_TYPE_MAP: Dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}

_TERMINAL_STATUSES = {"COMPLETE", "REJECTED", "CANCELLED"}


class KiteExecutionAdapter:
    """
    Execution adapter for Zerodha Kite Connect v5.

    Implements :class:`~shunya.algorithm.orders.ExecutionAdapter`.
    """

    def __init__(
        self,
        kite_client: Any,
        *,
        default_exchange: str = "NSE",
        default_product: str = "CNC",
        max_submit_retries: int = 3,
        retry_base_seconds: float = 0.5,
    ) -> None:
        if max_submit_retries < 1:
            raise ValueError("max_submit_retries must be >= 1")
        self._kite = kite_client
        self._default_exchange = default_exchange
        self._default_product = default_product
        self._max_retries = max_submit_retries
        self._retry_base = retry_base_seconds

    def submit_orders(
        self,
        orders: List[OrderSpec],
        *,
        dry_run: bool,
        correlation_id: str,
    ) -> List[OrderAttempt]:
        attempts: List[OrderAttempt] = []
        for spec in orders:
            oid = f"{correlation_id}-{spec.symbol}-{uuid.uuid4().hex[:12]}"
            side_str = spec.side.value
            notional = spec.notional or round(spec.quantity * (spec.price or 0.0), 2)

            if dry_run:
                attempts.append(OrderAttempt(
                    symbol=spec.symbol,
                    client_order_id=oid,
                    side=side_str,
                    notional=notional,
                    success=True,
                    initial_status="dry_run",
                    final_status="dry_run",
                ))
                continue

            variety = _VARIETY_MAP.get(spec.variety, "regular")
            order_type = _ORDER_TYPE_MAP.get(spec.order_type, "MARKET")
            exchange = spec.exchange or self._default_exchange
            product = spec.product or self._default_product
            transaction_type = "BUY" if spec.side is OrderSide.BUY else "SELL"

            if spec.variety is OrderVariety.COVER and spec.trigger_price is not None:
                self._validate_trigger_range(
                    transaction_type, exchange, spec.symbol, spec.trigger_price
                )

            kw: Dict[str, Any] = {
                "variety": variety,
                "exchange": exchange,
                "tradingsymbol": spec.symbol,
                "transaction_type": transaction_type,
                "quantity": spec.quantity,
                "product": product,
                "order_type": order_type,
            }
            if spec.price is not None:
                kw["price"] = spec.price
            if spec.trigger_price is not None:
                kw["trigger_price"] = spec.trigger_price
            if spec.validity:
                kw["validity"] = spec.validity
            if spec.validity_ttl is not None:
                kw["validity_ttl"] = spec.validity_ttl
            if spec.iceberg_legs is not None:
                kw["iceberg_legs"] = spec.iceberg_legs
            if spec.iceberg_quantity is not None:
                kw["iceberg_quantity"] = spec.iceberg_quantity
            if spec.market_protection is not None:
                kw["market_protection"] = spec.market_protection
            if spec.tag is not None:
                kw["tag"] = spec.tag[:20]

            attempt = OrderAttempt(
                symbol=spec.symbol,
                client_order_id=oid,
                side=side_str,
                notional=notional,
                success=False,
            )

            broker_order_id: Optional[str] = None
            last_err: Optional[str] = None
            for attempt_n in range(self._max_retries):
                try:
                    broker_order_id = self._kite.place_order(**kw)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt_n < self._max_retries - 1:
                        time.sleep(self._retry_base * (2 ** attempt_n))

            if last_err is None and broker_order_id is not None:
                attempt.success = True
                attempt.initial_status = "OPEN"
                attempt.final_status = "OPEN"
            else:
                attempt.error = last_err
                logger.error(
                    "Kite order failed for %s after retries: %s", spec.symbol, last_err
                )
            attempts.append(attempt)
        return attempts

    def observe_submitted_orders(
        self,
        attempts: List[OrderAttempt],
        *,
        max_polls: int = 3,
        poll_interval_seconds: float = 0.5,
    ) -> List[OrderAttempt]:
        """Poll Kite order_history to capture latest status and fill fields."""
        if max_polls < 1:
            return attempts

        all_orders: Optional[List[Dict[str, Any]]] = None
        try:
            all_orders = self._kite.orders()
        except Exception as e:
            logger.warning("Could not fetch Kite orders for observation: %s", e)
            return attempts

        if not all_orders:
            return attempts

        oid_map: Dict[str, Dict[str, Any]] = {}
        for o in all_orders:
            oid = o.get("order_id")
            if oid:
                oid_map[str(oid)] = o

        for attempt in attempts:
            if not attempt.success:
                continue
            matched = None
            for oid, o in oid_map.items():
                tag = o.get("tag", "")
                if tag and attempt.client_order_id and tag in attempt.client_order_id:
                    matched = o
                    break

            if matched is None:
                continue

            order_id = matched.get("order_id")
            if order_id is None:
                continue

            for poll_idx in range(max_polls):
                try:
                    history = self._kite.order_history(order_id)
                    if history:
                        latest = history[-1]
                        status = latest.get("status", "")
                        attempt.final_status = status
                        fq = latest.get("filled_quantity")
                        if fq is not None:
                            attempt.filled_qty = float(fq)
                        fap = latest.get("average_price")
                        if fap is not None:
                            attempt.filled_avg_price = float(fap)
                        if status in _TERMINAL_STATUSES:
                            break
                except Exception as e:
                    attempt.status_error = str(e)

                if poll_idx < max_polls - 1:
                    time.sleep(poll_interval_seconds)

        return attempts

    def get_positions(self) -> Dict[str, float]:
        """Current net positions as ``{tradingsymbol: signed_value}``."""
        positions = self._kite.positions()
        net_list = positions.get("net", [])
        result: Dict[str, float] = {}
        for p in net_list:
            sym = p.get("tradingsymbol", "")
            qty = float(p.get("quantity", 0))
            last_price = float(p.get("last_price", 0))
            result[sym] = qty * last_price
        return result

    def is_market_open(self) -> bool:
        """Best-effort liveness probe via LTP fetch."""
        try:
            self._kite.ltp(f"{self._default_exchange}:NIFTY 50")
            return True
        except Exception:
            return False

    def buying_power(self) -> float:
        """Available equity margin from Kite."""
        try:
            margins = self._kite.margins("equity")
            available = margins.get("available", {})
            return float(available.get("live_balance", 0))
        except Exception as e:
            logger.warning("Could not read Kite margin: %s", e)
            return 0.0

    def cancel_open_orders(self) -> None:
        """Best-effort cancellation of all open orders."""
        try:
            orders = self._kite.orders()
        except Exception as e:
            logger.warning("Could not fetch orders for cancellation: %s", e)
            return

        for o in orders:
            status = (o.get("status") or "").upper()
            if status in _TERMINAL_STATUSES:
                continue
            oid = o.get("order_id")
            variety = o.get("variety", "regular")
            if oid is None:
                continue
            try:
                if variety == "co":
                    self._kite.exit_order(variety=variety, order_id=oid)
                else:
                    self._kite.cancel_order(variety=variety, order_id=oid)
            except Exception as e:
                logger.warning("Failed to cancel Kite order %s: %s", oid, e)

    def _validate_trigger_range(
        self,
        transaction_type: str,
        exchange: str,
        symbol: str,
        trigger_price: float,
    ) -> None:
        """Validate trigger_price against Kite's allowed CO trigger range."""
        try:
            instrument = f"{exchange}:{symbol}"
            ranges = self._kite.trigger_range(
                transaction_type.lower(), instrument
            )
            data = ranges.get(instrument, {})
            lower = data.get("lower", 0)
            upper = data.get("upper", float("inf"))
            if not (lower <= trigger_price <= upper):
                logger.warning(
                    "Trigger price %.2f for %s outside allowed range [%.2f, %.2f]; "
                    "order may be rejected",
                    trigger_price, symbol, lower, upper,
                )
        except Exception as e:
            logger.warning("Could not validate trigger range for %s: %s", symbol, e)
