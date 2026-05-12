"""Stable ``client_order_id`` values linking EMS child orders to OMS parents."""

from __future__ import annotations


def child_client_order_id(parent_id: str, slice_index: int, urgency_attempt: int = 0) -> str:
    """
    Return a client order id for slice ``slice_index`` (zero-based).

    When ``urgency_attempt > 0``, a suffix is appended so Alpaca accepts a fresh
    order after cancel/replace escalation on the same slice.
    """
    base = f"{parent_id}:{int(slice_index)}"
    if int(urgency_attempt) <= 0:
        return base
    return f"{base}u{int(urgency_attempt)}"


def parse_child_client_order_id(client_order_id: str) -> tuple[str, int] | None:
    """Return ``(parent_id, slice_index)`` when parseable."""
    s = str(client_order_id)
    base = s.split("u", 1)[0]
    if ":" not in base:
        return None
    a, b = base.split(":", 1)
    try:
        return a, int(b)
    except ValueError:
        return None


def parent_root_from_client_order_id(client_order_id: str | None) -> str | None:
    """OMS parent id extracted from EMS ``client_order_id`` (ignores urgency suffix)."""
    if not client_order_id:
        return None
    base = str(client_order_id).split("u", 1)[0]
    return base.split(":", 1)[0] if ":" in base else base
