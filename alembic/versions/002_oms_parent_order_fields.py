"""Add manual-order fields to oms_parent_orders (order_type, limit_price, tif, broker_order_id, updated_at, origin)."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002_oms_parent_order_fields"
down_revision: Union[str, None] = "001_oms_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("oms_parent_orders", sa.Column("order_type", sa.String(length=16), nullable=True))
    op.add_column("oms_parent_orders", sa.Column("limit_price", sa.Float(), nullable=True))
    op.add_column("oms_parent_orders", sa.Column("time_in_force", sa.String(length=8), nullable=True))
    op.add_column("oms_parent_orders", sa.Column("broker_order_id", sa.String(length=64), nullable=True))
    op.add_column("oms_parent_orders", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("oms_parent_orders", sa.Column("origin", sa.String(length=16), nullable=True))
    op.create_index(
        "ix_oms_parent_orders_broker_order_id", "oms_parent_orders", ["broker_order_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_oms_parent_orders_broker_order_id", table_name="oms_parent_orders")
    op.drop_column("oms_parent_orders", "origin")
    op.drop_column("oms_parent_orders", "updated_at")
    op.drop_column("oms_parent_orders", "broker_order_id")
    op.drop_column("oms_parent_orders", "time_in_force")
    op.drop_column("oms_parent_orders", "limit_price")
    op.drop_column("oms_parent_orders", "order_type")
