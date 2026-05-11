"""Add alert_records table for persistent alert storage

Revision ID: 003
Revises: 002
Create Date: 2026-05-10
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alert_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("alert_id", sa.String(36), nullable=False, unique=True),
        sa.Column("sport", sa.String(10), nullable=False),
        sa.Column("home_team", sa.String(50), nullable=False),
        sa.Column("away_team", sa.String(50), nullable=False),
        sa.Column("game_date", sa.String(20), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("prob_over", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("edge", sa.Float(), nullable=False),
        sa.Column("total_line", sa.Float(), nullable=False),
        sa.Column("fired", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("webhook_response", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alert_records_alert_id", "alert_records", ["alert_id"])
    op.create_index("ix_alert_records_sport", "alert_records", ["sport"])
    op.create_index("ix_alert_records_created_at", "alert_records", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_alert_records_created_at", table_name="alert_records")
    op.drop_index("ix_alert_records_sport", table_name="alert_records")
    op.drop_index("ix_alert_records_alert_id", table_name="alert_records")
    op.drop_table("alert_records")
