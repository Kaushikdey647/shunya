from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from psycopg.rows import dict_row

from backtest_api.db import resolve_database_url
from backtest_api.schemas.models import AlphaCreate, AlphaOut, AlphaPatch, FinStratConfig


def _row_to_out(row: dict[str, Any]) -> AlphaOut:
    fc = row["finstrat_config"]
    if isinstance(fc, str):
        fc = json.loads(fc)
    return AlphaOut(
        id=str(row["id"]),
        name=row["name"],
        description=row.get("description"),
        import_ref=row.get("import_ref"),
        source_code=row.get("source_code"),
        finstrat_config=dict(fc) if fc is not None else {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def insert_alpha(body: AlphaCreate) -> AlphaOut:
    import psycopg
    from psycopg import errors as pg_errors

    cfg = body.finstrat_config.model_dump(mode="json", exclude_none=True)
    try:
        with psycopg.connect(resolve_database_url()) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO api_alphas (name, description, import_ref, source_code, finstrat_config)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING id, name, description, import_ref, source_code, finstrat_config, created_at, updated_at
                    """,
                    (body.name, body.description, body.import_ref, body.source_code, json.dumps(cfg)),
                )
                row = cur.fetchone()
            conn.commit()
    except pg_errors.UniqueViolation as exc:
        raise RuntimeError("duplicate_alpha_name") from exc
    if row is None:
        raise RuntimeError("insert_alpha: INSERT did not return row")
    return _row_to_out(row)


def list_alphas(limit: int = 100, offset: int = 0) -> list[AlphaOut]:
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, name, description, import_ref, source_code, finstrat_config, created_at, updated_at
                FROM api_alphas
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
    return [_row_to_out(r) for r in rows]


def get_alpha(alpha_id: str) -> Optional[AlphaOut]:
    import psycopg

    try:
        UUID(alpha_id)
    except ValueError:
        return None
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, name, description, import_ref, source_code, finstrat_config, created_at, updated_at
                FROM api_alphas WHERE id = %s
                """,
                (alpha_id,),
            )
            row = cur.fetchone()
    return _row_to_out(row) if row else None


def update_alpha(alpha_id: str, patch: AlphaPatch) -> Optional[AlphaOut]:
    import psycopg

    try:
        UUID(alpha_id)
    except ValueError:
        return None
    data = patch.model_dump(exclude_unset=True)
    fields: list[str] = []
    params: list[Any] = []
    if "name" in data:
        fields.append("name = %s")
        params.append(data["name"])
    if "description" in data:
        fields.append("description = %s")
        params.append(data["description"])
    if "import_ref" in data:
        fields.append("import_ref = %s")
        params.append(data["import_ref"])
    if "source_code" in data:
        fields.append("source_code = %s")
        params.append(data["source_code"])
    if "finstrat_config" in data and data["finstrat_config"] is not None:
        fields.append("finstrat_config = %s::jsonb")
        params.append(
            json.dumps(
                FinStratConfig.model_validate(data["finstrat_config"]).model_dump(
                    mode="json", exclude_none=True
                )
            )
        )
    if not fields:
        return get_alpha(alpha_id)
    fields.append("updated_at = %s")
    params.append(datetime.now(timezone.utc))
    params.append(alpha_id)
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                UPDATE api_alphas SET {", ".join(fields)}
                WHERE id = %s
                RETURNING id, name, description, import_ref, source_code, finstrat_config, created_at, updated_at
                """,
                tuple(params),
            )
            row = cur.fetchone()
        conn.commit()
    return _row_to_out(row) if row else None


def get_alpha_raw(alpha_id: str) -> Optional[dict[str, Any]]:
    """Row dict for worker: id, import_ref, finstrat_config (parsed)."""
    import psycopg

    try:
        UUID(alpha_id)
    except ValueError:
        return None
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, import_ref, source_code, finstrat_config FROM api_alphas WHERE id = %s",
                (alpha_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    fc = row["finstrat_config"]
    if isinstance(fc, str):
        fc = json.loads(fc)
    row["finstrat_config"] = dict(fc) if fc else {}
    return row


def delete_alpha(alpha_id: str) -> bool:
    import psycopg
    from psycopg import errors as pg_errors

    try:
        UUID(alpha_id)
    except ValueError:
        return False
    try:
        with psycopg.connect(resolve_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM api_alphas WHERE id = %s", (alpha_id,))
                n = cur.rowcount
            conn.commit()
    except pg_errors.ForeignKeyViolation:
        raise RuntimeError("foreign_key_violation") from None
    return n > 0
