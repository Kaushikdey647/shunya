"""
Kite Connect historical market data provider for :class:`~shunya.data.fints.finTs`.

Requires ``kiteconnect>=5.0`` (optional dependency).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd

from .timeframes import (
    BarIndexPolicy,
    BarSpec,
    BarUnit,
    default_bar_index_policy,
    default_bar_spec,
    normalize_history_index,
)

logger = logging.getLogger(__name__)


def _bar_spec_to_kite_interval(spec: BarSpec) -> str:
    """Map :class:`BarSpec` to a Kite Connect ``historical_data`` interval string."""
    u, s = spec.unit, spec.step
    if u == BarUnit.MINUTES:
        allowed = {1: "minute", 3: "3minute", 5: "5minute", 10: "10minute",
                   15: "15minute", 30: "30minute", 60: "60minute"}
        if s not in allowed:
            raise ValueError(
                f"Kite minute interval: step must be one of {sorted(allowed)}, got {s}"
            )
        return allowed[s]
    if u == BarUnit.HOURS:
        if s != 1:
            raise ValueError(f"Kite hourly interval: only step=1 supported (use 60minute), got {s}")
        return "60minute"
    if u == BarUnit.DAYS:
        if s != 1:
            raise ValueError(f"Kite daily interval: only step=1 supported, got {s}")
        return "day"
    if u == BarUnit.WEEKS:
        if s != 1:
            raise ValueError(f"Kite weekly interval: only step=1 supported, got {s}")
        return "week"
    if u == BarUnit.MONTHS:
        if s != 1:
            raise ValueError(f"Kite monthly interval: only step=1 supported, got {s}")
        return "month"
    raise ValueError(f"Unsupported BarSpec for Kite historical_data: {spec!r}")


class KiteHistoricalMarketDataProvider:
    """
    Historical OHLCV bars from Kite Connect (Zerodha).

    Implements the :class:`~shunya.data.providers.MarketDataProvider` protocol.

    Requires an authenticated ``kiteconnect.KiteConnect`` instance (or API key +
    access token via env vars ``KITE_API_KEY`` / ``KITE_ACCESS_TOKEN``).

    ``instrument_map`` maps trading symbols (e.g. ``"INFY"``) to Kite
    ``instrument_token`` integers. If omitted, tokens are resolved lazily via
    ``kite.instruments(exchange)`` and cached for the provider lifetime.
    """

    def __init__(
        self,
        *,
        kite_client: Optional[object] = None,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        instrument_map: Optional[Dict[str, int]] = None,
        default_exchange: str = "NSE",
    ) -> None:
        if kite_client is not None:
            self._kite = kite_client
        else:
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:
                raise ImportError(
                    "kiteconnect>=5.0 is required for KiteHistoricalMarketDataProvider. "
                    "Install with: pip install 'shunya-py[kite]'"
                ) from exc
            key = api_key or os.environ.get("KITE_API_KEY")
            token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
            if not key:
                raise ValueError(
                    "Kite API key required. Set KITE_API_KEY env var or pass api_key="
                )
            kc = KiteConnect(api_key=key)
            if token:
                kc.set_access_token(token)
            self._kite = kc

        self._default_exchange = default_exchange
        self._instrument_map: Dict[str, int] = dict(instrument_map or {})
        self._instruments_fetched = bool(instrument_map)

    def _resolve_instrument_token(self, symbol: str) -> int:
        if symbol in self._instrument_map:
            return self._instrument_map[symbol]
        if not self._instruments_fetched:
            self._fetch_instruments()
        token = self._instrument_map.get(symbol)
        if token is None:
            raise ValueError(
                f"Cannot resolve instrument_token for {symbol!r}. "
                f"Provide instrument_map= or ensure the symbol exists on {self._default_exchange}."
            )
        return token

    def _fetch_instruments(self) -> None:
        instruments = self._kite.instruments(self._default_exchange)
        for inst in instruments:
            ts = inst.get("tradingsymbol") or inst.get("tradingsymbol", "")
            token = inst.get("instrument_token")
            if ts and token is not None:
                self._instrument_map[str(ts)] = int(token)
        self._instruments_fetched = True
        logger.info(
            "Fetched %d instruments from Kite exchange=%s",
            len(self._instrument_map),
            self._default_exchange,
        )

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
    ) -> pd.DataFrame:
        if not ticker_list:
            return pd.DataFrame()

        spec = bar_spec if bar_spec is not None else default_bar_spec()
        idx_policy = bar_index_policy if bar_index_policy is not None else default_bar_index_policy()
        interval = _bar_spec_to_kite_interval(spec)

        s = pd.Timestamp(start)
        e = pd.Timestamp(end)

        frames: List[pd.DataFrame] = []
        keys: List[str] = []

        for sym in ticker_list:
            token = self._resolve_instrument_token(sym)
            try:
                candles = self._kite.historical_data(
                    token,
                    from_date=s.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=e.strftime("%Y-%m-%d %H:%M:%S"),
                    interval=interval,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Kite historical_data request failed for {sym} "
                    f"(token={token}, interval={interval}). "
                    "Check session validity and instrument permissions."
                ) from exc

            if not candles:
                raise ValueError(f"Kite historical_data returned no candles for {sym}")

            records = []
            idx = []
            for c in candles:
                records.append({
                    "Open": float(c["open"]),
                    "High": float(c["high"]),
                    "Low": float(c["low"]),
                    "Close": float(c["close"]),
                    "Volume": float(c["volume"]),
                })
                idx.append(pd.Timestamp(c["date"]))

            part = pd.DataFrame(records, index=idx).sort_index()
            part = normalize_history_index(part, spec, policy=idx_policy)
            frames.append(part)
            keys.append(sym)

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1 and len(ticker_list) == 1:
            return frames[0]
        out = pd.concat({k: f for k, f in zip(keys, frames, strict=True)}, axis=1)
        return out
