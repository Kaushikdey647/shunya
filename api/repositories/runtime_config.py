"""Read/write ``api_runtime_config`` JSONB (operator tunables, non-secrets)."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

from api.db import cursor_dict

logger = logging.getLogger(__name__)


def fetch_runtime_payload() -> Optional[dict[str, Any]]:
    """Return stored payload dict, or ``None`` if DB unavailable / table missing."""
    try:
        with cursor_dict() as cur:
            cur.execute("SELECT payload FROM api_runtime_config WHERE id = 1")
            row = cur.fetchone()
            if not row:
                return {}
            raw = row.get("payload")
            if raw is None:
                return {}
            if isinstance(raw, dict):
                return dict(raw)
            if isinstance(raw, str):
                return dict(json.loads(raw))
            return dict(raw)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.debug("runtime_config fetch skipped: %s", exc)
        return None


def save_runtime_payload(payload: Mapping[str, Any]) -> None:
    """Replace row 1 payload (caller validates keys)."""
    with cursor_dict() as cur:
        cur.execute(
            """
            UPDATE api_runtime_config
            SET payload = %s::jsonb, updated_at = now()
            WHERE id = 1
            """,
            (json.dumps(dict(payload)),),
        )
