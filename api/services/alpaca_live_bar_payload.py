"""Normalize Alpaca live ``Bar`` payloads for instrument WebSocket clients."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _as_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def alpaca_bar_to_ohlcv_dict(bar: Any) -> dict[str, Any]:
    """
    Map an Alpaca-py :class:`~alpaca.data.models.bars.Bar` (or compatible mapping) to
    REST-aligned OHLCV fields for JSON (no ``type`` wrapper).
    """
    if isinstance(bar, Mapping):
        # Raw msgpack-style keys (defensive)
        ts = bar.get("t") or bar.get("timestamp")
        if isinstance(ts, datetime):
            time_s = _as_utc_iso(ts)
        elif isinstance(ts, str) and ts.strip():
            time_s = ts.strip()
        else:
            raise ValueError("bar missing timestamp")
        return {
            "time": time_s,
            "open": float(bar.get("o", bar.get("open"))),
            "high": float(bar.get("h", bar.get("high"))),
            "low": float(bar.get("l", bar.get("low"))),
            "close": float(bar.get("c", bar.get("close"))),
            "volume": float(bar.get("v", bar.get("volume", 0) or 0)),
        }

    ts = getattr(bar, "timestamp", None)
    if not isinstance(ts, datetime):
        raise ValueError("bar missing timestamp")
    vol = getattr(bar, "volume", None)
    return {
        "time": _as_utc_iso(ts),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(vol) if vol is not None else None,
    }
