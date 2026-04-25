from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Sport(str, Enum):
    NFL = "nfl"
    NBA = "nba"
    MLB = "mlb"


class PredictionRequest(BaseModel):
    sport: Sport
    home_team: str = Field(..., min_length=2, max_length=50)
    away_team: str = Field(..., min_length=2, max_length=50)
    game_date: datetime
    total_line: float = Field(..., gt=0, description="The posted over/under line")
    home_rest_days: int | None = Field(None, ge=0, le=14)
    away_rest_days: int | None = Field(None, ge=0, le=14)

    # GROUP C — situational context
    wind_speed_mph: float = Field(0.0, ge=0.0, le=100.0, description="Wind speed (NFL/MLB outdoor)")
    temperature_f: float = Field(72.0, ge=-20.0, le=120.0, description="Game-time temperature")
    is_dome: bool = Field(True, description="Indoor/dome stadium (removes weather variance)")
    altitude_feet: float = Field(0.0, ge=0.0, le=8000.0, description="Stadium altitude (Denver=5280)")
    is_playoff: bool = Field(False, description="Playoff game (typically lower-scoring)")

    # GROUP D — market / sharp signals
    line_movement: float = Field(0.0, ge=-20.0, le=20.0, description="Closing minus opening line")
    public_over_pct: float = Field(0.5, ge=0.0, le=1.0, description="Fraction of public bets on Over")
    sharp_over_pct: float = Field(0.5, ge=0.0, le=1.0, description="Fraction of sharp money on Over")
    ref_foul_rate: float = Field(0.0, ge=-10.0, le=10.0, description="NBA ref fouls/game delta from avg")
    ump_walk_rate: float = Field(0.0, ge=-5.0, le=5.0, description="MLB ump walks/game delta from avg")

    # GROUP E — injury counts
    home_key_players_out: int = Field(0, ge=0, le=15, description="Key home players unavailable")
    away_key_players_out: int = Field(0, ge=0, le=15, description="Key away players unavailable")

    # Legacy injury impact (kept for backward compat)
    home_injury_impact: float = Field(0.0, ge=0.0, le=1.0)
    away_injury_impact: float = Field(0.0, ge=0.0, le=1.0)

    include_features: bool = False

    @field_validator("home_team", "away_team")
    @classmethod
    def upper_team(cls, v: str) -> str:
        return v.strip().upper()


class PredictionResponse(BaseModel):
    prediction_id: UUID
    game_id: UUID | None
    sport: Sport
    home_team: str
    away_team: str
    game_date: datetime
    total_line: float
    model_version: str
    prob_over: float
    prob_under: float
    ci_lower: float
    ci_upper: float
    predicted_direction: str
    confidence: float
    latency_ms: float
    features: dict | None = None


class ModelPerformanceResponse(BaseModel):
    version: str
    sport: str
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    training_games: int | None
    feature_count: int | None
    xgb_weight: float
    lgb_weight: float
    trained_at: datetime
    is_active: bool


class DriftReport(BaseModel):
    sport: str
    retrain_triggered: bool
    features_checked: int
    features_drifted: int
    feature_details: dict


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    models_loaded: dict[str, str | None]
    uptime_seconds: float
    version: str
