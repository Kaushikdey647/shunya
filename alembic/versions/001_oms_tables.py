"""OMS tables (parent orders + execution fills)."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001_oms_tables"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "oms_parent_orders",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("quantity_ordered", sa.Integer(), nullable=False),
        sa.Column("quantity_filled", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("client_order_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_oms_parent_orders_symbol", "oms_parent_orders", ["symbol"])
    op.create_index("ix_oms_parent_orders_state", "oms_parent_orders", ["state"])

    op.create_table(
        "oms_execution_fills",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_id", sa.String(length=128), nullable=False),
        sa.Column("parent_order_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("child_client_order_id", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent_order_id"],
            ["oms_parent_orders.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trade_id", name="uq_oms_execution_fills_trade_id"),
    )
    op.create_index("ix_oms_execution_fills_trade_id", "oms_execution_fills", ["trade_id"])
    op.create_index(
        "ix_oms_execution_fills_parent_order_id", "oms_execution_fills", ["parent_order_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_oms_execution_fills_parent_order_id", table_name="oms_execution_fills")
    op.drop_index("ix_oms_execution_fills_trade_id", table_name="oms_execution_fills")
    op.drop_table("oms_execution_fills")
    op.drop_index("ix_oms_parent_orders_state", table_name="oms_parent_orders")
    op.drop_index("ix_oms_parent_orders_symbol", table_name="oms_parent_orders")
    op.drop_table("oms_parent_orders")
