"""Postgres persistence for the live daily runner (``live_*`` tables)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

LIVE_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def live_database_url() -> str:
    """``SHUNYA_LIVE_DATABASE_URL`` or ``DATABASE_URL``; raise ``RuntimeError`` when neither is set."""
    url = os.environ.get("SHUNYA_LIVE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set SHUNYA_LIVE_DATABASE_URL or DATABASE_URL for the live runner.")
    return url


def apply_live_migrations(dsn: str | None = None) -> None:
    """Apply ``shunya/live/migrations/*.sql`` (idempotent)."""
    from shunya.data.timescale.dbutil import apply_migrations

    apply_migrations(dsn or live_database_url(), LIVE_MIGRATIONS_DIR)


@dataclass(frozen=True)
class Control:
    """Kill-switch state (singleton row)."""

    state: str
    level: Optional[str] = None
    reason: Optional[str] = None
    actor: Optional[str] = None
    updated_at: Optional[datetime] = None

    @property
    def halted(self) -> bool:
        """True unless the switch is explicitly armed."""
        return self.state != "armed"


@dataclass(frozen=True)
class StrategyVersion:
    """Frozen alpha snapshot run by a sleeve."""

    id: str
    name: str
    alpha_source: str
    finstrat_config: Dict[str, Any]
    tickers: List[str]
    lookback_days: int


@dataclass(frozen=True)
class Sleeve:
    """One strategy version's capital slot inside a book."""

    id: str
    book_id: str
    version: StrategyVersion
    weight: float
    status: str
    ramp_days: int
    runs_completed: int


@dataclass(frozen=True)
class Book:
    """Set of sleeves traded in one Alpaca account."""

    id: str
    name: str
    status: str
    capital_fraction: float
    risk_config: Dict[str, Any] = field(default_factory=dict)
    hwm_equity: Optional[float] = None
    last_equity: Optional[float] = None


@dataclass(frozen=True)
class Run:
    """One book's trading cycle for one session date."""

    id: str
    book_id: str
    session_date: date
    signal_date: date
    status: str
    error: Optional[str] = None
    equity: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class OrderRow:
    """Persisted broker order for a run."""

    run_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: int
    status: str
    alpaca_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    error: Optional[str] = None


def _version(r: Mapping[str, Any]) -> StrategyVersion:
    return StrategyVersion(
        id=str(r["v_id"]),
        name=r["v_name"],
        alpha_source=r["alpha_source"],
        finstrat_config=dict(r["finstrat_config"] or {}),
        tickers=list(r["tickers"]),
        lookback_days=int(r["lookback_days"]),
    )


def _book(r: Mapping[str, Any]) -> Book:
    return Book(
        id=str(r["id"]),
        name=r["name"],
        status=r["status"],
        capital_fraction=float(r["capital_fraction"]),
        risk_config=dict(r["risk_config"] or {}),
        hwm_equity=r["hwm_equity"],
        last_equity=r["last_equity"],
    )


def _run(r: Mapping[str, Any]) -> Run:
    return Run(
        id=str(r["id"]),
        book_id=str(r["book_id"]),
        session_date=r["session_date"],
        signal_date=r["signal_date"],
        status=r["status"],
        error=r["error"],
        equity=r["equity"],
        detail=dict(r["detail"] or {}),
        created_at=r["created_at"],
    )


def _order(r: Mapping[str, Any]) -> OrderRow:
    return OrderRow(
        run_id=str(r["run_id"]),
        client_order_id=r["client_order_id"],
        symbol=r["symbol"],
        side=r["side"],
        qty=int(r["qty"]),
        status=r["status"],
        alpaca_id=r["alpaca_id"],
        filled_qty=float(r["filled_qty"]),
        filled_avg_price=r["filled_avg_price"],
        error=r["error"],
    )


_SLEEVE_SELECT = """
    SELECT s.id, s.book_id, s.weight, s.status, s.ramp_days, s.runs_completed,
           v.id AS v_id, v.name AS v_name, v.alpha_source, v.finstrat_config, v.tickers, v.lookback_days
    FROM live_sleeves s JOIN live_strategy_versions v ON v.id = s.version_id
"""


class PostgresLiveStore:
    """Short-lived connection per call (daily cadence; no pool needed)."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or live_database_url()

    @contextmanager
    def _cur(self) -> Iterator[Any]:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur

    @staticmethod
    def _json(v: Any) -> Any:
        from psycopg.types.json import Jsonb

        return Jsonb(v)

    # --- control / audit / heartbeat -------------------------------------------------

    def get_control(self) -> Control:
        """Read the kill-switch row."""
        with self._cur() as cur:
            cur.execute("SELECT state, level, reason, actor, updated_at FROM live_control WHERE id = 1")
            r = cur.fetchone()
        if r is None:
            return Control(state="halted", reason="live_control row missing")
        return Control(**r)

    def set_control(self, *, state: str, level: Optional[str], reason: Optional[str], actor: str) -> None:
        """Upsert the kill-switch row."""
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_control (id, state, level, reason, actor, updated_at)
                VALUES (1, %s, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, level = EXCLUDED.level,
                    reason = EXCLUDED.reason, actor = EXCLUDED.actor, updated_at = now()
                """,
                (state, level, reason, actor),
            )

    def audit(self, actor: str, action: str, detail: Mapping[str, Any] | None = None) -> None:
        """Append an audit row."""
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO live_audit (actor, action, detail) VALUES (%s, %s, %s)",
                (actor, action, self._json(dict(detail or {}))),
            )

    def list_audit(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Most recent audit rows first."""
        with self._cur() as cur:
            cur.execute("SELECT ts, actor, action, detail FROM live_audit ORDER BY id DESC LIMIT %s", (limit,))
            return list(cur.fetchall())

    def write_heartbeat(self, result: str) -> None:
        """Record the latest tick time and outcome."""
        with self._cur() as cur:
            cur.execute(
                "UPDATE live_heartbeat SET last_tick_at = now(), last_result = %s WHERE id = 1", (result,)
            )

    def get_heartbeat(self) -> Dict[str, Any]:
        """Latest tick time and outcome."""
        with self._cur() as cur:
            cur.execute("SELECT last_tick_at, last_result FROM live_heartbeat WHERE id = 1")
            return dict(cur.fetchone() or {})

    # --- strategy versions / books / sleeves -----------------------------------------

    def create_version(
        self,
        *,
        name: str,
        alpha_source: str,
        finstrat_config: Mapping[str, Any],
        tickers: Sequence[str],
        lookback_days: int,
    ) -> str:
        """Insert a frozen strategy snapshot; return its id."""
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_strategy_versions (name, alpha_source, finstrat_config, tickers, lookback_days)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (name, alpha_source, self._json(dict(finstrat_config)), list(tickers), lookback_days),
            )
            return str(cur.fetchone()["id"])

    def list_versions(self) -> List[Dict[str, Any]]:
        """Version metadata (no source) newest first."""
        with self._cur() as cur:
            cur.execute(
                """
                SELECT id, name, finstrat_config, cardinality(tickers) AS n_tickers, lookback_days, created_at
                FROM live_strategy_versions ORDER BY created_at DESC
                """
            )
            return [dict(r, id=str(r["id"])) for r in cur.fetchall()]

    def create_book(
        self, *, name: str, capital_fraction: float, risk_config: Mapping[str, Any], status: str = "active"
    ) -> str:
        """Insert a book; return its id."""
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_books (name, capital_fraction, risk_config, status)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (name, capital_fraction, self._json(dict(risk_config)), status),
            )
            return str(cur.fetchone()["id"])

    def update_book(self, book_id: str, **fields: Any) -> None:
        """Patch ``status``, ``capital_fraction``, ``risk_config``, ``name``."""
        allowed = {"status", "capital_fraction", "risk_config", "name"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            sets.append(f"{k} = %s")
            vals.append(self._json(v) if k == "risk_config" else v)
        if not sets:
            return
        with self._cur() as cur:
            cur.execute(
                f"UPDATE live_books SET {', '.join(sets)}, updated_at = now() WHERE id = %s",
                (*vals, book_id),
            )

    def update_book_equity(self, book_id: str, *, equity: float, hwm: float) -> None:
        """Store latest equity and high-water mark."""
        with self._cur() as cur:
            cur.execute(
                "UPDATE live_books SET last_equity = %s, hwm_equity = %s, updated_at = now() WHERE id = %s",
                (equity, hwm, book_id),
            )

    def list_books(self, *, active_only: bool = False) -> List[Book]:
        """All books (or only active)."""
        q = "SELECT * FROM live_books"
        if active_only:
            q += " WHERE status = 'active'"
        with self._cur() as cur:
            cur.execute(q + " ORDER BY created_at")
            return [_book(r) for r in cur.fetchall()]

    def get_book(self, book_id: str) -> Optional[Book]:
        """One book or ``None``."""
        with self._cur() as cur:
            cur.execute("SELECT * FROM live_books WHERE id = %s", (book_id,))
            r = cur.fetchone()
        return _book(r) if r else None

    def create_sleeve(self, *, book_id: str, version_id: str, weight: float, ramp_days: int = 1) -> str:
        """Attach a strategy version to a book; return sleeve id."""
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_sleeves (book_id, version_id, weight, ramp_days)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (book_id, version_id, weight, ramp_days),
            )
            return str(cur.fetchone()["id"])

    def update_sleeve(self, sleeve_id: str, **fields: Any) -> None:
        """Patch ``weight``, ``status``, ``ramp_days``."""
        allowed = {"weight", "status", "ramp_days"}
        sets = [(k, v) for k, v in fields.items() if k in allowed and v is not None]
        if not sets:
            return
        with self._cur() as cur:
            cur.execute(
                f"UPDATE live_sleeves SET {', '.join(f'{k} = %s' for k, _ in sets)}, updated_at = now() "
                "WHERE id = %s",
                (*[v for _, v in sets], sleeve_id),
            )

    def increment_sleeve_runs(self, sleeve_ids: Sequence[str]) -> None:
        """Bump ramp counters after a submitted run."""
        if not sleeve_ids:
            return
        with self._cur() as cur:
            cur.execute(
                "UPDATE live_sleeves SET runs_completed = runs_completed + 1 WHERE id = ANY(%s::uuid[])",
                (list(sleeve_ids),),
            )

    def list_sleeves(self, book_id: str) -> List[Sleeve]:
        """Sleeves of a book (excluding retired)."""
        with self._cur() as cur:
            cur.execute(_SLEEVE_SELECT + " WHERE s.book_id = %s AND s.status <> 'retired' ORDER BY s.created_at", (book_id,))
            rows = cur.fetchall()
        return [
            Sleeve(
                id=str(r["id"]),
                book_id=str(r["book_id"]),
                version=_version(r),
                weight=float(r["weight"]),
                status=r["status"],
                ramp_days=int(r["ramp_days"]),
                runs_completed=int(r["runs_completed"]),
            )
            for r in rows
        ]

    # --- runs ---------------------------------------------------------------------------

    def claim_run(self, *, book_id: str, session_date: date, signal_date: date) -> Optional[str]:
        """Insert a ``running`` row; ``None`` when this book/session is already claimed.

        A ``failed`` run is re-claimed (max 3 attempts), as is a ``running`` row untouched for
        30 minutes (crashed process). Re-submission is safe: deltas net out working orders and
        client order ids are deterministic per book/session/symbol.
        """
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_runs (book_id, session_date, signal_date, status, detail)
                VALUES (%s, %s, %s, 'running', '{"attempts": 1}'::jsonb)
                ON CONFLICT (book_id, session_date) DO UPDATE SET status = 'running', error = NULL,
                    detail = live_runs.detail || jsonb_build_object(
                        'attempts', COALESCE((live_runs.detail->>'attempts')::int, 1) + 1),
                    updated_at = now()
                WHERE (live_runs.status = 'failed' AND COALESCE((live_runs.detail->>'attempts')::int, 1) < 3)
                   OR (live_runs.status = 'running' AND live_runs.updated_at < now() - interval '30 minutes')
                RETURNING id
                """,
                (book_id, session_date, signal_date),
            )
            r = cur.fetchone()
        return str(r["id"]) if r else None

    def claim_reconcile(self, run_id: str) -> bool:
        """Atomically move ``submitted`` → ``reconciling``."""
        with self._cur() as cur:
            cur.execute(
                "UPDATE live_runs SET status = 'reconciling', updated_at = now() "
                "WHERE id = %s AND status = 'submitted' RETURNING id",
                (run_id,),
            )
            return cur.fetchone() is not None

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        error: Optional[str] = None,
        equity: Optional[float] = None,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Set status and merge ``detail``."""
        with self._cur() as cur:
            cur.execute(
                """
                UPDATE live_runs SET status = %s, error = COALESCE(%s, error), equity = COALESCE(%s, equity),
                    detail = detail || %s, updated_at = now()
                WHERE id = %s
                """,
                (status, error, equity, self._json(dict(detail or {})), run_id),
            )

    def get_run(self, run_id: str) -> Optional[Run]:
        """One run or ``None``."""
        with self._cur() as cur:
            cur.execute("SELECT * FROM live_runs WHERE id = %s", (run_id,))
            r = cur.fetchone()
        return _run(r) if r else None

    def list_runs(self, *, book_id: Optional[str] = None, status: Optional[str] = None, limit: int = 30) -> List[Run]:
        """Newest first, optional filters."""
        where, vals = [], []
        if book_id:
            where.append("book_id = %s")
            vals.append(book_id)
        if status:
            where.append("status = %s")
            vals.append(status)
        q = "SELECT * FROM live_runs" + (f" WHERE {' AND '.join(where)}" if where else "")
        with self._cur() as cur:
            cur.execute(q + " ORDER BY session_date DESC, created_at DESC LIMIT %s", (*vals, limit))
            return [_run(r) for r in cur.fetchall()]

    # --- targets / orders ---------------------------------------------------------------

    def save_sleeve_targets(self, run_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Rows: ``sleeve_id, symbol, target_usd, target_shares, prev_shares``."""
        if not rows:
            return
        with self._cur() as cur:
            cur.executemany(
                """
                INSERT INTO live_sleeve_targets (run_id, sleeve_id, symbol, target_usd, target_shares, prev_shares)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, sleeve_id, symbol) DO UPDATE SET target_usd = EXCLUDED.target_usd,
                    target_shares = EXCLUDED.target_shares, prev_shares = EXCLUDED.prev_shares
                """,
                [
                    (run_id, r["sleeve_id"], r["symbol"], r["target_usd"], r["target_shares"], r["prev_shares"])
                    for r in rows
                ],
            )

    def list_sleeve_targets(self, run_id: str) -> List[Dict[str, Any]]:
        """All sleeve target rows for a run."""
        with self._cur() as cur:
            cur.execute(
                "SELECT sleeve_id, symbol, target_usd, target_shares, prev_shares FROM live_sleeve_targets "
                "WHERE run_id = %s ORDER BY symbol",
                (run_id,),
            )
            return [dict(r, sleeve_id=str(r["sleeve_id"])) for r in cur.fetchall()]

    def upsert_order(self, o: OrderRow) -> None:
        """Insert or update by ``client_order_id``."""
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_orders (run_id, client_order_id, symbol, side, qty, alpaca_id, status,
                    filled_qty, filled_avg_price, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (client_order_id) DO UPDATE SET alpaca_id = COALESCE(EXCLUDED.alpaca_id, live_orders.alpaca_id),
                    status = EXCLUDED.status, filled_qty = EXCLUDED.filled_qty,
                    filled_avg_price = EXCLUDED.filled_avg_price, error = EXCLUDED.error, updated_at = now()
                """,
                (
                    o.run_id, o.client_order_id, o.symbol, o.side, o.qty, o.alpaca_id, o.status,
                    o.filled_qty, o.filled_avg_price, o.error,
                ),
            )

    def list_orders(self, run_id: str) -> List[OrderRow]:
        """Orders of a run."""
        with self._cur() as cur:
            cur.execute("SELECT * FROM live_orders WHERE run_id = %s ORDER BY symbol", (run_id,))
            return [_order(r) for r in cur.fetchall()]

    # --- sleeve ledger ------------------------------------------------------------------

    def get_sleeve_positions(self, book_id: str) -> Dict[str, Dict[str, float]]:
        """``sleeve_id -> symbol -> shares`` for a book."""
        with self._cur() as cur:
            cur.execute(
                """
                SELECT p.sleeve_id, p.symbol, p.shares FROM live_sleeve_positions p
                JOIN live_sleeves s ON s.id = p.sleeve_id WHERE s.book_id = %s
                """,
                (book_id,),
            )
            out: Dict[str, Dict[str, float]] = {}
            for r in cur.fetchall():
                out.setdefault(str(r["sleeve_id"]), {})[r["symbol"]] = float(r["shares"])
        return out

    def set_sleeve_positions(self, sleeve_id: str, shares: Mapping[str, float]) -> None:
        """Replace a sleeve's ledger (zero rows dropped)."""
        with self._cur() as cur:
            cur.execute("DELETE FROM live_sleeve_positions WHERE sleeve_id = %s", (sleeve_id,))
            rows = [(sleeve_id, s, float(q)) for s, q in shares.items() if abs(float(q)) > 1e-9]
            if rows:
                cur.executemany(
                    "INSERT INTO live_sleeve_positions (sleeve_id, symbol, shares) VALUES (%s, %s, %s)", rows
                )

    def save_snapshot(self, sleeve_id: str, session_date: date, market_value: float) -> None:
        """Upsert a sleeve's end-of-run market value."""
        with self._cur() as cur:
            cur.execute(
                """
                INSERT INTO live_sleeve_snapshots (sleeve_id, session_date, market_value) VALUES (%s, %s, %s)
                ON CONFLICT (sleeve_id, session_date) DO UPDATE SET market_value = EXCLUDED.market_value
                """,
                (sleeve_id, session_date, market_value),
            )

    def list_snapshots(self, book_id: str, limit: int = 120) -> List[Dict[str, Any]]:
        """Sleeve market-value history for a book."""
        with self._cur() as cur:
            cur.execute(
                """
                SELECT n.sleeve_id, n.session_date, n.market_value FROM live_sleeve_snapshots n
                JOIN live_sleeves s ON s.id = n.sleeve_id WHERE s.book_id = %s
                ORDER BY n.session_date DESC LIMIT %s
                """,
                (book_id, limit),
            )
            return [dict(r, sleeve_id=str(r["sleeve_id"])) for r in cur.fetchall()]
