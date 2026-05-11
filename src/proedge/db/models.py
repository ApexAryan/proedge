import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Game(Base):
    __tablename__ = "games"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sport = Column(String(10), nullable=False, index=True)
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    game_date = Column(DateTime(timezone=True), nullable=False, index=True)
    status = Column(String(20), default="scheduled")
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    total_line = Column(Float, nullable=True)
    result_over = Column(Boolean, nullable=True)
    external_id = Column(String(50), unique=True, nullable=True)
    venue = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    predictions = relationship("Prediction", back_populates="game")
    player_stats = relationship("PlayerStat", back_populates="game")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    sport = Column(String(10), nullable=False)
    prob_over = Column(Float, nullable=False)
    prob_under = Column(Float, nullable=False)
    ci_lower = Column(Float, nullable=False)
    ci_upper = Column(Float, nullable=False)
    predicted_direction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    features_snapshot = Column(JSONB, nullable=True)
    predicted_at = Column(DateTime(timezone=True), server_default=func.now())
    is_correct = Column(Boolean, nullable=True)
    latency_ms = Column(Float, nullable=True)
    actual_total = Column(Float, nullable=True)
    closing_line = Column(Float, nullable=True)
    clv = Column(Float, nullable=True)  # closing line value: positive = beat the close
    settled_at = Column(DateTime(timezone=True), nullable=True)

    game = relationship("Game", back_populates="predictions")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(50), unique=True, nullable=False)
    sport = Column(String(10), nullable=False)
    accuracy = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    training_games = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    xgb_weight = Column(Float, default=0.5)
    lgb_weight = Column(Float, default=0.5)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
    model_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False, index=True)
    hyperparams = Column(JSONB, nullable=True)


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=True)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=True)
    team_id = Column(String(50), nullable=False, index=True)
    sport = Column(String(10), nullable=False)
    game_date = Column(DateTime(timezone=True), nullable=False, index=True)
    stats = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    game = relationship("Game", back_populates="player_stats")


class InjuryReport(Base):
    __tablename__ = "injury_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=True)
    team_id = Column(String(50), nullable=False, index=True)
    sport = Column(String(10), nullable=False)
    status = Column(String(30), nullable=False)  # out, doubtful, questionable, probable, active
    injury_type = Column(String(100), nullable=True)
    expected_return = Column(DateTime(timezone=True), nullable=True)
    impact_score = Column(Float, nullable=True)  # 0–1 estimated team impact
    reported_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(36), unique=True, nullable=False, index=True)
    sport = Column(String(10), nullable=False, index=True)
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    game_date = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # "over" | "under"
    prob_over = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    total_line = Column(Float, nullable=False)
    fired = Column(Boolean, default=False)
    webhook_response = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
