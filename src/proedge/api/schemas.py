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
    altitude_feet: float = Field(
        0.0, ge=0.0, le=8000.0, description="Stadium altitude (Denver=5280)"
    )
    is_playoff: bool = Field(False, description="Playoff game (typically lower-scoring)")

    # GROUP D — market / sharp signals
    line_movement: float = Field(0.0, ge=-20.0, le=20.0, description="Closing minus opening line")
    public_over_pct: float = Field(
        0.5, ge=0.0, le=1.0, description="Fraction of public bets on Over"
    )
    sharp_over_pct: float = Field(
        0.5, ge=0.0, le=1.0, description="Fraction of sharp money on Over"
    )
    ref_foul_rate: float = Field(
        0.0, ge=-10.0, le=10.0, description="NBA ref fouls/game delta from avg"
    )
    ump_walk_rate: float = Field(
        0.0, ge=-5.0, le=5.0, description="MLB ump walks/game delta from avg"
    )

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
    line_comparison: "LineComparisonResponse | None" = None


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


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    models_loaded: dict[str, str | None]
    uptime_seconds: float
    version: str


class PlayerProjectionResponse(BaseModel):
    projection_id: str
    player_name: str
    team: str
    position: str
    stat_type: str
    line: float
    game_id: str
    home_team: str
    away_team: str
    start_time: datetime | None
    status: str
    is_promo: bool
    odds_type: str  # "standard" | "demon" | "goblin"
    projection_type: str
    sport: str


class GameLineResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    start_time: datetime | None
    stat_type: str
    line: float
    sport: str


class GameSummaryResponse(BaseModel):
    """All lines and props grouped by game, ready to feed into the model."""

    game_id: str
    home_team: str
    away_team: str
    start_time: datetime | None
    sport: str
    total_line: float | None
    spread: float | None  # positive = home favoured by N pts
    game_lines: list[GameLineResponse]
    player_projections: list[PlayerProjectionResponse]
    projected_total: float | None  # sum of point props when no explicit total line


class PrizePicksBoardResponse(BaseModel):
    sport: str
    fetched_at: datetime
    game_count: int
    player_prop_count: int
    game_line_count: int
    games: list[GameSummaryResponse]


class LineComparisonResponse(BaseModel):
    """Cross-source line comparison for a specific matchup."""

    sport: str
    home_team: str
    away_team: str
    book_line: float | None = None
    prizepicks_line: float | None = None
    kalshi_line: float | None = None
    kalshi_nearest_threshold: int | None = None
    kalshi_nearest_prob: float | None = None
    consensus_line: float | None = None
    pp_vs_book: float | None = None
    kalshi_vs_book: float | None = None
    sources: list[str] = Field(default_factory=list)


class KalshiThresholdData(BaseModel):
    """One Kalshi contract (e.g. 'Over 213.5 points') with market prices and model edge."""
    threshold: float
    yes_ask: float          # cost to buy Over contract (bet Over)
    no_ask: float           # cost to buy Under contract (bet Under)
    market_prob: float      # market-implied P(over) = yes midpoint
    model_prob: float | None = None   # model's P(over at this line)
    ev_yes: float | None = None       # expected value of Over bet per $1 risked
    ev_no: float | None = None        # expected value of Under bet per $1 risked
    best_bet: str | None = None       # "over", "under", or None (no edge)
    edge: float | None = None         # magnitude of best edge (positive = value)


class LineMatrixResponse(BaseModel):
    """Full line matrix for a game across Kalshi, PrizePicks, and Underdog."""
    sport: str
    home_team: str
    away_team: str
    kalshi_implied_line: float | None = None
    prizepicks_line: float | None = None
    underdog_line: float | None = None
    underdog_over_american: str | None = None
    underdog_under_american: str | None = None
    thresholds: list[KalshiThresholdData] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class MarketScanRequest(BaseModel):
    sports: list[str] = Field(default_factory=lambda: ["nba", "mlb", "nfl"])


class MarketScanGameResult(BaseModel):
    game_id: UUID | None
    prediction_id: UUID | None
    sport: str
    home_team: str
    away_team: str
    kalshi_implied_line: float
    model_prob_over: float
    model_prob_under: float
    predicted_direction: str
    confidence: float
    best_threshold: float | None = None
    best_bet: str | None = None
    best_edge: float | None = None
    thresholds: list[KalshiThresholdData] = Field(default_factory=list)
    line_movement: float | None = None   # change from first-seen implied line; None = first scan
    ci_lower: float | None = None
    ci_upper: float | None = None
    home_key_players_out: int = 0
    away_key_players_out: int = 0
    live_stats: bool = False


class MarketScanResponse(BaseModel):
    scanned_at: datetime
    sports: list[str]
    total_games: int
    results: list[MarketScanGameResult]


class SettleRequest(BaseModel):
    actual_total: float = Field(..., gt=0, description="Final combined score / runs")
    closing_line: float = Field(..., gt=0, description="Line at game time (as it closed)")


class SettleResponse(BaseModel):
    prediction_id: UUID
    actual_total: float
    closing_line: float
    clv: float
    is_correct: bool
    predicted_direction: str
    message: str
