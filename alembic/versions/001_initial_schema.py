"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "games",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sport", sa.String(10), nullable=False),
        sa.Column("home_team", sa.String(50), nullable=False),
        sa.Column("away_team", sa.String(50), nullable=False),
        sa.Column("game_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(20), server_default="scheduled"),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("total_line", sa.Float(), nullable=True),
        sa.Column("result_over", sa.Boolean(), nullable=True),
        sa.Column("external_id", sa.String(50), unique=True, nullable=True),
        sa.Column("venue", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_games_sport", "games", ["sport"])
    op.create_index("ix_games_game_date", "games", ["game_date"])

    op.create_table(
        "predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("sport", sa.String(10), nullable=False),
        sa.Column("prob_over", sa.Float(), nullable=False),
        sa.Column("prob_under", sa.Float(), nullable=False),
        sa.Column("ci_lower", sa.Float(), nullable=False),
        sa.Column("ci_upper", sa.Float(), nullable=False),
        sa.Column("predicted_direction", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("features_snapshot", JSONB(), nullable=True),
        sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
    )
    op.create_index("ix_predictions_game_id", "predictions", ["game_id"])

    op.create_table(
        "model_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("version", sa.String(50), unique=True, nullable=False),
        sa.Column("sport", sa.String(10), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("log_loss", sa.Float(), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column("training_games", sa.Integer(), nullable=True),
        sa.Column("feature_count", sa.Integer(), nullable=True),
        sa.Column("xgb_weight", sa.Float(), server_default="0.5"),
        sa.Column("lgb_weight", sa.Float(), server_default="0.5"),
        sa.Column("trained_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("model_path", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="false"),
        sa.Column("hyperparams", JSONB(), nullable=True),
    )
    op.create_index("ix_model_runs_is_active", "model_runs", ["is_active"])

    op.create_table(
        "player_stats",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("player_id", sa.String(50), nullable=False),
        sa.Column("player_name", sa.String(100), nullable=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=True),
        sa.Column("team_id", sa.String(50), nullable=False),
        sa.Column("sport", sa.String(10), nullable=False),
        sa.Column("game_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stats", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_player_stats_player_id", "player_stats", ["player_id"])
    op.create_index("ix_player_stats_team_id", "player_stats", ["team_id"])
    op.create_index("ix_player_stats_game_date", "player_stats", ["game_date"])

    op.create_table(
        "injury_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("player_id", sa.String(50), nullable=False),
        sa.Column("player_name", sa.String(100), nullable=True),
        sa.Column("team_id", sa.String(50), nullable=False),
        sa.Column("sport", sa.String(10), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("injury_type", sa.String(100), nullable=True),
        sa.Column("expected_return", sa.DateTime(timezone=True), nullable=True),
        sa.Column("impact_score", sa.Float(), nullable=True),
        sa.Column("reported_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_updated", sa.DateTime(timezone=True), onupdate=sa.text("now()")),
    )
    op.create_index("ix_injury_reports_player_id", "injury_reports", ["player_id"])
    op.create_index("ix_injury_reports_team_id", "injury_reports", ["team_id"])


def downgrade() -> None:
    op.drop_table("injury_reports")
    op.drop_table("player_stats")
    op.drop_table("model_runs")
    op.drop_table("predictions")
    op.drop_table("games")
