"""Normalize Alpaca live L1 ``Quote`` / ``Trade`` payloads for instrument WebSocket clients."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _as_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _exchange_str(val: Any) -> str | None:
    if val is None:
        return None
    return str(val)


def alpaca_quote_to_dict(q: Any) -> dict[str, Any]:
    """Map Alpaca-py ``Quote`` (or mapping) to JSON body (no ``type`` wrapper)."""
    if isinstance(q, Mapping):
        ts = q.get("t") or q.get("timestamp")
        if isinstance(ts, datetime):
            t_s = _as_utc_iso(ts)
        elif isinstance(ts, str) and ts.strip():
            t_s = ts.strip()
        else:
            raise ValueError("quote missing timestamp")
        sym = q.get("S") or q.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            raise ValueError("quote missing symbol")
        return {
            "symbol": sym.strip().upper(),
            "time": t_s,
            "bid_price": float(q.get("bp", q.get("bid_price"))),
            "bid_size": float(q.get("bs", q.get("bid_size", 0) or 0)),
            "ask_price": float(q.get("ap", q.get("ask_price"))),
            "ask_size": float(q.get("as", q.get("ask_size", 0) or 0)),
            "bid_exchange": _exchange_str(q.get("bx", q.get("bid_exchange"))),
            "ask_exchange": _exchange_str(q.get("ax", q.get("ask_exchange"))),
            "conditions": q.get("c", q.get("conditions")),
            "tape": q.get("z", q.get("tape")),
        }

    ts = getattr(q, "timestamp", None)
    if not isinstance(ts, datetime):
        raise ValueError("quote missing timestamp")
    return {
        "symbol": str(getattr(q, "symbol", "")).strip().upper(),
        "time": _as_utc_iso(ts),
        "bid_price": float(q.bid_price),
        "bid_size": float(q.bid_size),
        "ask_price": float(q.ask_price),
        "ask_size": float(q.ask_size),
        "bid_exchange": _exchange_str(getattr(q, "bid_exchange", None)),
        "ask_exchange": _exchange_str(getattr(q, "ask_exchange", None)),
        "conditions": getattr(q, "conditions", None),
        "tape": getattr(q, "tape", None),
    }


def alpaca_trade_to_dict(t: Any) -> dict[str, Any]:
    """Map Alpaca-py ``Trade`` (or mapping) to JSON body (no ``type`` wrapper)."""
    if isinstance(t, Mapping):
        ts = t.get("t") or t.get("timestamp")
        if isinstance(ts, datetime):
            t_s = _as_utc_iso(ts)
        elif isinstance(ts, str) and ts.strip():
            t_s = ts.strip()
        else:
            raise ValueError("trade missing timestamp")
        sym = t.get("S") or t.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            raise ValueError("trade missing symbol")
        tid = t.get("i", t.get("id"))
        return {
            "symbol": sym.strip().upper(),
            "time": t_s,
            "price": float(t.get("p", t.get("price"))),
            "size": float(t.get("s", t.get("size", 0) or 0)),
            "id": int(tid) if tid is not None else None,
            "exchange": _exchange_str(t.get("x", t.get("exchange"))),
            "conditions": t.get("c", t.get("conditions")),
            "tape": t.get("z", t.get("tape")),
        }

    ts = getattr(t, "timestamp", None)
    if not isinstance(ts, datetime):
        raise ValueError("trade missing timestamp")
    tid = getattr(t, "id", None)
    return {
        "symbol": str(getattr(t, "symbol", "")).strip().upper(),
        "time": _as_utc_iso(ts),
        "price": float(t.price),
        "size": float(t.size),
        "id": int(tid) if tid is not None else None,
        "exchange": _exchange_str(getattr(t, "exchange", None)),
        "conditions": getattr(t, "conditions", None),
        "tape": getattr(t, "tape", None),
    }


def alpaca_trade_correction_to_dict(c: Any) -> dict[str, Any]:
    """Map Alpaca-py ``TradeCorrection`` to JSON body (no ``type`` wrapper)."""
    ts = getattr(c, "timestamp", None)
    if not isinstance(ts, datetime):
        raise ValueError("correction missing timestamp")
    oc = getattr(c, "original_conditions", None) or []
    cc = getattr(c, "corrected_conditions", None) or []
    return {
        "symbol": str(getattr(c, "symbol", "")).strip().upper(),
        "time": _as_utc_iso(ts),
        "original_id": getattr(c, "original_id", None),
        "original_price": float(getattr(c, "original_price", 0)),
        "original_size": float(getattr(c, "original_size", 0)),
        "original_conditions": list(oc) if not isinstance(oc, str) else [oc],
        "corrected_id": getattr(c, "corrected_id", None),
        "corrected_price": float(getattr(c, "corrected_price", 0)),
        "corrected_size": float(getattr(c, "corrected_size", 0)),
        "corrected_conditions": list(cc) if not isinstance(cc, str) else [cc],
        "tape": getattr(c, "tape", None),
    }


def alpaca_trade_cancel_to_dict(c: Any) -> dict[str, Any]:
    """Map Alpaca-py ``TradeCancel`` to JSON body (no ``type`` wrapper)."""
    ts = getattr(c, "timestamp", None)
    if not isinstance(ts, datetime):
        raise ValueError("cancel missing timestamp")
    return {
        "symbol": str(getattr(c, "symbol", "")).strip().upper(),
        "time": _as_utc_iso(ts),
        "price": float(getattr(c, "price", 0)),
        "size": float(getattr(c, "size", 0)),
        "id": getattr(c, "id", None),
        "exchange": _exchange_str(getattr(c, "exchange", None)),
        "action": getattr(c, "action", None),
        "tape": getattr(c, "tape", None),
    }
