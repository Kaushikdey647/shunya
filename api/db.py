"""Postgres connection helpers for API repositories."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from api.settings import get_settings


def resolve_database_url() -> str:
    """Raise ``RuntimeError`` when no API database URL is configured in settings or environment."""
    s = get_settings()
    if s.database_url:
        return str(s.database_url)
    url = os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Set SHUNYA_API_DATABASE_URL, DATABASE_URL, or SHUNYA_DATABASE_URL for API persistence."
        )
    return str(url)


@contextmanager
def cursor_dict() -> Iterator[psycopg.Cursor]:
    """Yield a psycopg dict-row cursor on a fresh connection to the API database."""
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
