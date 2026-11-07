"""Persistence helpers for OMS fills and parent orders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shunya.oms.fills import ExecutionFill

from .schema import Base, ExecutionFillRow, ParentOrderRow


def utc_now() -> datetime:
    """Current UTC timestamp for ``updated_at`` fields."""
    return datetime.now(timezone.utc)


def create_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine with ``future=True``."""
    from sqlalchemy import create_engine as sa_create_engine

    return sa_create_engine(url, future=True)


def create_all(engine: Engine) -> None:
    """Create OMS tables from :class:`Base` metadata."""
    Base.metadata.create_all(engine)


class OMSRepository:
    """Append-only fills and parent-order upserts."""

    def __init__(self, engine: Engine) -> None:
        """Bind a session factory to the given SQLAlchemy engine."""
        self._engine = engine
        self._session_factory = sessionmaker(engine, expire_on_commit=False, class_=Session)

    def upsert_parent_snapshot(
        self,
        *,
        parent_id: str,
        symbol: str,
        side: str,
        quantity_ordered: int,
        quantity_filled: int,
        state: str,
        client_order_id: Optional[str],
        created_at: datetime,
        order_type: Optional[str] = None,
        limit_price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> None:
        """Insert or update a parent order snapshot row in Postgres."""
        with self._session_factory() as s:
            row = s.get(ParentOrderRow, parent_id)
            if row is None:
                s.add(
                    ParentOrderRow(
                        id=parent_id,
                        symbol=symbol,
                        side=side,
                        quantity_ordered=quantity_ordered,
                        quantity_filled=quantity_filled,
                        state=state,
                        client_order_id=client_order_id,
                        created_at=created_at,
                        order_type=order_type,
                        limit_price=limit_price,
                        time_in_force=time_in_force,
                        broker_order_id=broker_order_id,
                        updated_at=created_at,
                        origin=origin,
                    )
                )
            else:
                row.quantity_filled = quantity_filled
                row.state = state
                row.client_order_id = client_order_id
                if order_type is not None:
                    row.order_type = order_type
                if limit_price is not None:
                    row.limit_price = limit_price
                if time_in_force is not None:
                    row.time_in_force = time_in_force
                if broker_order_id is not None:
                    row.broker_order_id = broker_order_id
                if origin is not None:
                    row.origin = origin
                row.updated_at = utc_now()
            s.commit()

    def get_parent(self, parent_id: str) -> Optional[ParentOrderRow]:
        """Load one ``ParentOrderRow`` by primary key."""
        with self._session_factory() as s:
            return s.get(ParentOrderRow, parent_id)

    def list_parents(
        self,
        *,
        states: Optional[Sequence[str]] = None,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> list[ParentOrderRow]:
        """List parent orders filtered by state and symbol."""
        with self._session_factory() as s:
            stmt = select(ParentOrderRow)
            if states:
                stmt = stmt.where(ParentOrderRow.state.in_(states))
            if symbol:
                stmt = stmt.where(ParentOrderRow.symbol == symbol)
            stmt = stmt.order_by(ParentOrderRow.created_at.desc()).limit(limit)
            return list(s.scalars(stmt).all())

    def list_open_parents(self) -> list[ParentOrderRow]:
        """Return parents in non-terminal working states."""
        return self.list_parents(
            states=("PENDING_SUBMIT", "WORKING", "PARTIAL_FILL"), limit=1000
        )

    def list_recent_fills(
        self, *, limit: int = 200, symbol: Optional[str] = None
    ) -> list[ExecutionFill]:
        """Load recent fills as :class:`ExecutionFill` records."""
        with self._session_factory() as s:
            stmt = select(ExecutionFillRow)
            if symbol:
                stmt = stmt.where(ExecutionFillRow.symbol == symbol)
            stmt = stmt.order_by(ExecutionFillRow.ts.desc()).limit(limit)
            rows = s.scalars(stmt).all()
        return [
            ExecutionFill(
                trade_id=r.trade_id,
                parent_order_id=r.parent_order_id,
                symbol=r.symbol,
                side=r.side,
                quantity=float(r.quantity),
                price=float(r.price),
                fee=float(r.fee),
                ts=r.ts,
                child_client_order_id=r.child_client_order_id,
            )
            for r in rows
        ]

    def insert_fill_idempotent(self, fill: ExecutionFill) -> bool:
        """
        Insert a fill if ``trade_id`` is new.

        Returns True when a row was inserted, False when skipped (duplicate).
        """
        with self._session_factory() as s:
            stmt = (
                pg_insert(ExecutionFillRow)
                .values(
                    trade_id=fill.trade_id,
                    parent_order_id=fill.parent_order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=float(fill.quantity),
                    price=float(fill.price),
                    fee=float(fill.fee),
                    ts=fill.ts,
                    child_client_order_id=fill.child_client_order_id,
                )
                .on_conflict_do_nothing(index_elements=["trade_id"])
            )
            res = s.execute(stmt)
            s.commit()
            return res.rowcount == 1  # type: ignore[union-attr]

    def list_fills_for_parent(self, parent_id: str) -> list[ExecutionFill]:
        """Return all fills for a single parent order id."""
        with self._session_factory() as s:
            rows = s.scalars(
                select(ExecutionFillRow).where(ExecutionFillRow.parent_order_id == parent_id)
            ).all()
        return [
            ExecutionFill(
                trade_id=r.trade_id,
                parent_order_id=r.parent_order_id,
                symbol=r.symbol,
                side=r.side,
                quantity=float(r.quantity),
                price=float(r.price),
                fee=float(r.fee),
                ts=r.ts,
                child_client_order_id=r.child_client_order_id,
            )
            for r in rows
        ]
