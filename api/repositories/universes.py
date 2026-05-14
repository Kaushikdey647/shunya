"""CRUD and analytics for user-defined api_universes / api_universe_members."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from psycopg.rows import dict_row

from api.db import resolve_database_url
from api.schemas.models import (
    UniverseBreakdownSlice,
    UniverseCreate,
    UniverseMemberOut,
    UniverseOut,
    UniversePatch,
    UniverseSummaryOut,
)


def _parse_uuid(uid: str) -> Optional[UUID]:
    try:
        return UUID(str(uid).strip())
    except ValueError:
        return None


def universe_exists(universe_id: str) -> bool:
    u = _parse_uuid(universe_id)
    if u is None:
        return False
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM api_universes WHERE id = %s LIMIT 1", (str(u),))
            return cur.fetchone() is not None


def constituent_tickers(universe_id: str) -> list[str]:
    u = _parse_uuid(universe_id)
    if u is None:
        return []
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker
                FROM api_universe_members m
                JOIN symbols s ON s.id = m.symbol_id
                WHERE m.universe_id = %s
                ORDER BY s.ticker
                """,
                (str(u),),
            )
            return [str(r[0]) for r in cur.fetchall()]


def _row_universe_out(row: dict[str, Any]) -> UniverseOut:
    return UniverseOut(
        id=str(row["id"]),
        name=row["name"],
        description=row.get("description"),
        member_count=int(row.get("member_count") or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def insert_universe(body: UniverseCreate) -> UniverseOut:
    import psycopg
    from psycopg import errors as pg_errors

    try:
        with psycopg.connect(resolve_database_url()) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO api_universes (name, description)
                    VALUES (%s, %s)
                    RETURNING id, name, description, created_at, updated_at,
                              (SELECT COUNT(*)::int FROM api_universe_members m WHERE m.universe_id = id) AS member_count
                    """,
                    (body.name.strip(), body.description),
                )
                row = cur.fetchone()
            conn.commit()
    except pg_errors.UniqueViolation as exc:
        raise RuntimeError("duplicate_universe_name") from exc
    if row is None:
        raise RuntimeError("insert_universe: no row")
    return _row_universe_out(row)


def list_universes(*, limit: int = 100, offset: int = 0) -> list[UniverseOut]:
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT u.id, u.name, u.description, u.created_at, u.updated_at,
                       (SELECT COUNT(*)::int FROM api_universe_members m WHERE m.universe_id = u.id) AS member_count
                FROM api_universes u
                ORDER BY u.updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
    return [_row_universe_out(r) for r in rows]


def get_universe(universe_id: str) -> Optional[UniverseOut]:
    u = _parse_uuid(universe_id)
    if u is None:
        return None
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT u.id, u.name, u.description, u.created_at, u.updated_at,
                       (SELECT COUNT(*)::int FROM api_universe_members m WHERE m.universe_id = u.id) AS member_count
                FROM api_universes u
                WHERE u.id = %s
                """,
                (str(u),),
            )
            row = cur.fetchone()
    return _row_universe_out(row) if row else None


def update_universe(universe_id: str, patch: UniversePatch) -> Optional[UniverseOut]:
    u = _parse_uuid(universe_id)
    if u is None:
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
    if not fields:
        return get_universe(universe_id)
    fields.append("updated_at = %s")
    params.append(datetime.now(timezone.utc))
    params.append(str(u))
    import psycopg
    from psycopg import errors as pg_errors

    try:
        with psycopg.connect(resolve_database_url()) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    UPDATE api_universes SET {", ".join(fields)}
                    WHERE id = %s
                    RETURNING id, name, description, created_at, updated_at,
                              (SELECT COUNT(*)::int FROM api_universe_members m WHERE m.universe_id = id) AS member_count
                    """,
                    tuple(params),
                )
                row = cur.fetchone()
            conn.commit()
    except pg_errors.UniqueViolation:
        raise RuntimeError("duplicate_universe_name") from None
    return _row_universe_out(row) if row else None


def delete_universe(universe_id: str) -> bool:
    u = _parse_uuid(universe_id)
    if u is None:
        return False
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_universes WHERE id = %s", (str(u),))
            n = cur.rowcount
        conn.commit()
    return n > 0


def list_members(
    universe_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[UniverseMemberOut], int]:
    u = _parse_uuid(universe_id)
    if u is None:
        return [], 0
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT COUNT(*)::int AS cnt FROM api_universe_members WHERE universe_id = %s",
                (str(u),),
            )
            cr = cur.fetchone()
            total = int(cr["cnt"]) if cr else 0
            cur.execute(
                """
                SELECT s.ticker,
                       sc.long_name,
                       sc.sector_disp,
                       sc.industry_disp
                FROM api_universe_members m
                JOIN symbols s ON s.id = m.symbol_id
                LEFT JOIN LATERAL (
                    SELECT c.long_name, c.sector_disp, c.industry_disp
                    FROM symbol_classifications c
                    WHERE c.symbol_id = s.id
                    ORDER BY c.as_of DESC
                    LIMIT 1
                ) sc ON true
                WHERE m.universe_id = %s
                ORDER BY s.ticker
                LIMIT %s OFFSET %s
                """,
                (str(u), limit, offset),
            )
            rows = cur.fetchall()
    out = [
        UniverseMemberOut(
            ticker=str(r["ticker"]),
            long_name=r.get("long_name"),
            sector_disp=r.get("sector_disp"),
            industry_disp=r.get("industry_disp"),
        )
        for r in rows
    ]
    return out, total


def _latest_quote_types(cur: Any, tickers: list[str]) -> dict[str, Optional[str]]:
    if not tickers:
        return {}
    cur.execute(
        """
        SELECT s.ticker, sc.quote_type
        FROM symbols s
        LEFT JOIN LATERAL (
            SELECT c.quote_type
            FROM symbol_classifications c
            WHERE c.symbol_id = s.id
            ORDER BY c.as_of DESC
            LIMIT 1
        ) sc ON true
        WHERE s.ticker = ANY(%s)
        """,
        (tickers,),
    )
    return {str(r[0]): r[1] for r in cur.fetchall()}


def _symbol_ids_equity_only(cur: Any, tickers: list[str]) -> dict[str, int]:
    """Return ticker -> symbol_id for symbols that exist and pass equity filter."""
    if not tickers:
        return {}
    qt = _latest_quote_types(cur, tickers)
    cur.execute(
        "SELECT id, ticker FROM symbols WHERE ticker = ANY(%s)",
        (tickers,),
    )
    id_by_ticker: dict[str, int] = {}
    for sid, tk in cur.fetchall():
        id_by_ticker[str(tk)] = int(sid)
    out: dict[str, int] = {}
    for t in tickers:
        if t not in id_by_ticker:
            continue
        q = qt.get(t)
        if q is not None and str(q).strip().upper() not in ("", "EQUITY"):
            raise ValueError(f"non_equity_ticker:{t}:{q}")
        out[t] = id_by_ticker[t]
    return out


def add_members(universe_id: str, tickers: list[str]) -> int:
    """Insert members; skips unknown tickers (caller may treat as error). Returns rows inserted."""
    u = _parse_uuid(universe_id)
    if u is None:
        return 0
    if not tickers:
        return 0
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            try:
                mapping = _symbol_ids_equity_only(cur, tickers)
            except ValueError:
                conn.rollback()
                raise
            if not mapping:
                conn.commit()
                return 0
            n = 0
            for sym_id in mapping.values():
                cur.execute(
                    """
                    INSERT INTO api_universe_members (universe_id, symbol_id)
                    VALUES (%s, %s)
                    ON CONFLICT (universe_id, symbol_id) DO NOTHING
                    """,
                    (str(u), sym_id),
                )
                n += cur.rowcount
        conn.commit()
    return n


def add_members_strict(universe_id: str, tickers: list[str]) -> int:
    """Like add_members but raises if any ticker is unknown."""
    u = _parse_uuid(universe_id)
    if u is None:
        raise ValueError("invalid_universe_id")
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            mapping = _symbol_ids_equity_only(cur, tickers)
        missing = [t for t in tickers if t not in mapping]
        if missing:
            raise ValueError("unknown_tickers:" + ",".join(missing[:20]))
        with conn.cursor() as cur:
            n = 0
            for sym_id in mapping.values():
                cur.execute(
                    """
                    INSERT INTO api_universe_members (universe_id, symbol_id)
                    VALUES (%s, %s)
                    ON CONFLICT (universe_id, symbol_id) DO NOTHING
                    """,
                    (str(u), sym_id),
                )
                n += cur.rowcount
        conn.commit()
    return n


def remove_members(universe_id: str, tickers: list[str]) -> int:
    u = _parse_uuid(universe_id)
    if u is None:
        return 0
    if not tickers:
        return 0
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM api_universe_members m
                USING symbols s
                WHERE m.symbol_id = s.id AND m.universe_id = %s AND s.ticker = ANY(%s)
                """,
                (str(u), tickers),
            )
            n = cur.rowcount
        conn.commit()
    return n


def replace_members(universe_id: str, tickers: list[str]) -> None:
    u = _parse_uuid(universe_id)
    if u is None:
        raise ValueError("invalid_universe_id")
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_universe_members WHERE universe_id = %s", (str(u),))
            if not tickers:
                conn.commit()
                return
            mapping = _symbol_ids_equity_only(cur, tickers)
            missing = [t for t in tickers if t not in mapping]
            if missing:
                raise ValueError("unknown_tickers:" + ",".join(missing[:20]))
            for sym_id in mapping.values():
                cur.execute(
                    "INSERT INTO api_universe_members (universe_id, symbol_id) VALUES (%s, %s)",
                    (str(u), sym_id),
                )
        conn.commit()


def universe_summary(universe_id: str) -> Optional[UniverseSummaryOut]:
    u = _parse_uuid(universe_id)
    if u is None:
        return None
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT COUNT(*)::int AS n FROM api_universe_members WHERE universe_id = %s",
                (str(u),),
            )
            row0 = cur.fetchone()
            member_count = int(row0["n"]) if row0 else 0
            if member_count == 0:
                return UniverseSummaryOut(
                    member_count=0,
                    classified_for_breakdown_count=0,
                    sector_breakdown=[],
                    industry_breakdown=[],
                    fundamentals_coverage_count=0,
                    median_market_cap=None,
                    mean_trailing_pe=None,
                    median_beta=None,
                )

            cur.execute(
                """
                SELECT sc.sector_disp, sc.sector_key, sc.industry_disp, sc.industry_key
                FROM api_universe_members m
                JOIN symbols s ON s.id = m.symbol_id
                LEFT JOIN LATERAL (
                    SELECT c.sector_disp, c.sector_key, c.industry_disp, c.industry_key
                    FROM symbol_classifications c
                    WHERE c.symbol_id = s.id
                    ORDER BY c.as_of DESC
                    LIMIT 1
                ) sc ON true
                WHERE m.universe_id = %s
                """,
                (str(u),),
            )
            class_rows = cur.fetchall()

    def _excluded_sector(label: Optional[str], key: Optional[str]) -> bool:
        if key is None or (isinstance(key, str) and not str(key).strip()):
            return True
        ld = (label or "").strip().lower()
        if ld in ("", "other", "unknown"):
            return True
        return False

    def _excluded_industry(label: Optional[str], key: Optional[str]) -> bool:
        if key is None or (isinstance(key, str) and not str(key).strip()):
            return True
        ld = (label or "").strip().lower()
        if ld in ("", "other", "unknown"):
            return True
        return False

    sector_counts: dict[str, int] = {}
    industry_counts: dict[str, int] = {}
    for r in class_rows:
        sd = r.get("sector_disp") or r.get("sector_key") or "Unknown"
        id_ = r.get("industry_disp") or r.get("industry_key") or "Unknown"
        sk = r.get("sector_key")
        ik = r.get("industry_key")
        if not _excluded_sector(str(sd) if sd else None, str(sk) if sk else None):
            sector_counts[str(sd)] = sector_counts.get(str(sd), 0) + 1
        if not _excluded_industry(str(id_) if id_ else None, str(ik) if ik else None):
            industry_counts[str(id_)] = industry_counts.get(str(id_), 0) + 1

    classified = sum(sector_counts.values())
    denom = classified if classified > 0 else 1
    sector_breakdown = [
        UniverseBreakdownSlice(label=k, count=v, fraction=round(v / denom, 6))
        for k, v in sorted(sector_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    ind_denom = sum(industry_counts.values()) or 1
    industry_breakdown = [
        UniverseBreakdownSlice(label=k, count=v, fraction=round(v / ind_denom, 6))
        for k, v in sorted(industry_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    # Latest fundamentals_daily per member symbol
    caps: list[float] = []
    pes: list[float] = []
    betas: list[float] = []
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (d.symbol_id)
                    d.market_cap, d.trailing_pe, d.beta
                FROM fundamentals_daily d
                JOIN api_universe_members m ON m.symbol_id = d.symbol_id AND m.universe_id = %s
                ORDER BY d.symbol_id, d.as_of_ts DESC
                """,
                (str(u),),
            )
            for r in cur.fetchall():
                mc = r.get("market_cap")
                pe = r.get("trailing_pe")
                b = r.get("beta")
                if mc is not None and float(mc) > 0 and str(mc).lower() != "nan":
                    caps.append(float(mc))
                if pe is not None and float(pe) > 0 and str(pe).lower() != "nan":
                    pes.append(float(pe))
                if b is not None and str(b).lower() != "nan":
                    betas.append(float(b))

    def _med(xs: list[float]) -> Optional[float]:
        if not xs:
            return None
        return float(statistics.median(xs))

    def _mean(xs: list[float]) -> Optional[float]:
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    return UniverseSummaryOut(
        member_count=member_count,
        classified_for_breakdown_count=classified,
        sector_breakdown=sector_breakdown,
        industry_breakdown=industry_breakdown,
        fundamentals_coverage_count=len(caps) + len(pes) + len(betas),
        median_market_cap=_med(caps),
        mean_trailing_pe=_mean(pes),
        median_beta=_med(betas),
    )
