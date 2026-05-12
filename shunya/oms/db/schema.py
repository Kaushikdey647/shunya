"""SQLAlchemy ORM tables for durable OMS state."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ParentOrderRow(Base):
    __tablename__ = "oms_parent_orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    quantity_ordered: Mapped[int] = mapped_column(Integer)
    quantity_filled: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(32), index=True)
    client_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    fills: Mapped[list["ExecutionFillRow"]] = relationship(back_populates="parent")


class ExecutionFillRow(Base):
    __tablename__ = "oms_execution_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    parent_order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("oms_parent_orders.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    child_client_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    parent: Mapped["ParentOrderRow"] = relationship(back_populates="fills")

    __table_args__ = (UniqueConstraint("trade_id", name="uq_oms_execution_fills_trade_id"),)
