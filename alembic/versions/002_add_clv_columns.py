"""Add CLV and settlement columns to predictions

Revision ID: 002
Revises: 001
Create Date: 2026-04-25
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("actual_total", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("closing_line", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("clv", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("predictions", "settled_at")
    op.drop_column("predictions", "clv")
    op.drop_column("predictions", "closing_line")
    op.drop_column("predictions", "actual_total")
