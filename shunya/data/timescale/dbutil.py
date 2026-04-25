"""Connection helpers and migration runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

_DEFAULT_MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if not url:
        raise ValueError(
            "Set DATABASE_URL or SHUNYA_DATABASE_URL (e.g. postgresql://postgres:postgres@localhost:5432/shunya)"
        )
    return url


def migrations_dir() -> Path:
    return Path(os.environ.get("SHUNYA_MIGRATIONS_DIR", str(_DEFAULT_MIGRATIONS)))


def migration_files() -> list[Path]:
    d = migrations_dir()
    if not d.is_dir():
        raise FileNotFoundError(f"Migrations directory not found: {d}")
    return sorted(p for p in d.glob("*.sql") if p.is_file())


def iter_migration_sql() -> Iterator[tuple[str, str]]:
    for path in migration_files():
        yield path.name, path.read_text(encoding="utf-8")


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Timescale features require the optional 'timescale' extra: pip install 'shunya-py[timescale]'"
        ) from exc


def apply_migrations(dsn: str | None = None) -> None:
    """Run all ``*.sql`` files in migrations dir in lexicographic order."""
    _require_psycopg()
    import psycopg

    url = dsn or get_database_url()
    for name, sql in iter_migration_sql():
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(sql)
        # noqa: print for CLI feedback
        print(f"applied migration: {name}")
