"""Kill switch: sticky flag in Postgres plus broker-side cancel / flatten."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from shunya.live.store import Control

if TYPE_CHECKING:
    from shunya.live.broker import Broker
    from shunya.live.store import PostgresLiveStore

_log = logging.getLogger(__name__)

KillLevel = Literal["halt", "cancel", "flatten"]
KILL_LEVELS: tuple[str, ...] = ("halt", "cancel", "flatten")


def read_control(store: "PostgresLiveStore") -> Control:
    """Fail closed: any error reading the flag reports ``halted``."""
    try:
        return store.get_control()
    except Exception as exc:  # noqa: BLE001
        _log.error("kill-switch state unreadable; treating as halted: %s", exc)
        return Control(state="halted", level="halt", reason=f"control unreadable: {exc}")


def engage_kill(
    store: "PostgresLiveStore",
    broker: Optional["Broker"],
    *,
    level: str,
    reason: str,
    actor: str,
) -> Dict[str, Any]:
    """Set the flag **first**, then cancel orders (``cancel``) and close positions (``flatten``).

    Broker steps run even when the flag write fails (DB down): stopping orders matters more.
    """
    if level not in KILL_LEVELS:
        raise ValueError(f"level must be one of {KILL_LEVELS}")
    out: Dict[str, Any] = {"level": level, "flag_set": False}
    try:
        store.set_control(state="halted", level=level, reason=reason, actor=actor)
        out["flag_set"] = True
    except Exception as exc:  # noqa: BLE001
        out["flag_error"] = str(exc)
        _log.error("kill switch flag write failed: %s", exc)
    try:
        if level in ("cancel", "flatten") and broker is not None:
            out["orders_cancelled"] = broker.cancel_all()
        if level == "flatten" and broker is not None:
            out["positions_closed"] = broker.close_all()
    except Exception as exc:  # noqa: BLE001 — flag is already set; report broker failure
        out["broker_error"] = str(exc)
        _log.error("kill switch broker step failed: %s", exc)
    try:
        store.audit(actor, "kill", {"reason": reason, **out})
    except Exception as exc:  # noqa: BLE001
        _log.error("kill switch audit write failed: %s", exc)
    return out


def resume(store: "PostgresLiveStore", *, reason: str, actor: str) -> None:
    """Re-arm trading (only explicit operator action clears a kill)."""
    store.set_control(state="armed", level=None, reason=reason, actor=actor)
    store.audit(actor, "resume", {"reason": reason})
