"""Persistence helpers for OMS fills and parent orders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shunya.oms.fills import ExecutionFill

from .schema import Base, ExecutionFillRow, ParentOrderRow


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_engine(url: str) -> Engine:
    from sqlalchemy import create_engine as sa_create_engine

    return sa_create_engine(url, future=True)


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)


class OMSRepository:
    """Append-only fills and parent-order upserts."""

    def __init__(self, engine: Engine) -> None:
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
    ) -> None:
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
                    )
                )
            else:
                row.quantity_filled = quantity_filled
                row.state = state
                row.client_order_id = client_order_id
            s.commit()

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
