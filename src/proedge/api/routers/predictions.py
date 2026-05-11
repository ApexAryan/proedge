"""Prediction endpoints — the primary API surface."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.api.schemas import PredictionRequest, PredictionResponse, SettleRequest, SettleResponse
from proedge.db.repositories import AlertRepository, GameRepository, PredictionRepository
from proedge.db.session import get_db
from proedge.monitoring.alerts import get_alert_manager
from proedge.monitoring.metrics import (
    INFERENCE_FEATURE_MISSING,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_PROB_OVER,
)
from proedge.pipeline.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])
_registry = ModelRegistry()
_model_cache: dict[str, object] = {}

# Odds board cache: sport → (timestamp, list[GameOdds])
# Refreshed at most once per 15 minutes to stay within the 500 req/month free quota.
_ODDS_CACHE_TTL = 900  # 15 minutes
_odds_cache: dict[str, tuple[float, list]] = {}


def _get_model(sport: str):
    if sport not in _model_cache:
        try:
            _model_cache[sport] = _registry.load(sport)
        except FileNotFoundError:
            return None
    return _model_cache[sport]


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction(req: PredictionRequest, db: AsyncSession = Depends(get_db)):
    t0 = time.perf_counter()
    sport = req.sport.value

    model = _get_model(sport)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No trained model available for sport '{sport}'. Run training first.",
        )

    meta = _registry.load_meta(sport)
    model_version = meta.get("version", "unknown")
    feature_names: list[str] = meta.get("feature_names", [])
    feature_medians: dict[str, float] = meta.get("feature_medians", {})

    from proedge.config import get_settings
    from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher

    _settings = get_settings()
    if _settings.odds_api_key:
        try:
            cached = _odds_cache.get(sport)
            now = time.time()
            if cached is None or (now - cached[0]) > _ODDS_CACHE_TTL:
                board = OddsFetcher(api_key=_settings.odds_api_key, timeout=5.0).fetch_game_odds(
                    sport
                )
                _odds_cache[sport] = (now, board)
            else:
                board = cached[1]

            home_lower = req.home_team.lower()
            away_lower = req.away_team.lower()
            for game in board:
                ht, at = game.home_team.lower(), game.away_team.lower()
                if (home_lower in ht or ht in home_lower) and (
                    away_lower in at or at in away_lower
                ):
                    if game.total_line is not None:
                        req = req.model_copy(update={"total_line": game.total_line})
                    break
        except Exception:
            pass  # best-effort; keep caller-supplied line

    # Auto-populate injury counts from ESPN if caller didn't provide them
    home_out = req.home_key_players_out
    away_out = req.away_key_players_out
    if home_out == 0 and away_out == 0:
        try:
            from proedge.pipeline.ingestion.injuries import InjuryFetcher

            fetcher = InjuryFetcher(timeout=5.0)
            reports = fetcher.fetch_all(sport)
            home_out = reports.get(
                req.home_team, type("_", (), {"key_players_out": 0})()
            ).key_players_out
            away_out = reports.get(
                req.away_team, type("_", (), {"key_players_out": 0})()
            ).key_players_out
        except Exception:
            pass  # injury fetch is best-effort; fall back to 0

    # Build inference feature row with available context
    X = _build_inference_features(req, feature_names, home_out, away_out, feature_medians)

    # Validate feature dimensions — warn on mismatch to surface train-serve skew
    expected = set(feature_names)
    actual = set(X.columns)
    missing = expected - actual
    if missing:
        logger.warning(
            "sport=%s: %d features present in training but missing at inference: %s",
            sport,
            len(missing),
            sorted(missing)[:10],
        )
        try:
            INFERENCE_FEATURE_MISSING.labels(sport=sport).inc(len(missing))
        except Exception:
            pass

    predictions = model.predict_with_intervals(X)
    pred = predictions[0]

    direction = "over" if pred["prob_over"] >= 0.5 else "under"
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Persist game and prediction
    game_repo = GameRepository(db)
    pred_repo = PredictionRepository(db)

    game = await game_repo.create(
        sport=sport,
        home_team=req.home_team,
        away_team=req.away_team,
        game_date=req.game_date,
        total_line=req.total_line,
        status="scheduled",
    )

    features_snapshot = (
        {col: float(X[col].iloc[0]) for col in feature_names if col in X.columns}
        if req.include_features
        else None
    )

    db_pred = await pred_repo.create(
        game_id=game.id,
        model_version=model_version,
        sport=sport,
        prob_over=pred["prob_over"],
        prob_under=pred["prob_under"],
        ci_lower=pred["ci_lower"],
        ci_upper=pred["ci_upper"],
        predicted_direction=direction,
        confidence=pred["confidence"],
        features_snapshot=features_snapshot,
        latency_ms=latency_ms,
    )

    # Emit Prometheus metrics
    PREDICTION_COUNT.labels(sport=sport, direction=direction).inc()
    PREDICTION_CONFIDENCE.labels(sport=sport).observe(pred["confidence"])
    PREDICTION_PROB_OVER.labels(sport=sport).observe(pred["prob_over"])

    # Fire alert if confidence exceeds threshold; persist to DB for durability
    try:
        alert = get_alert_manager().evaluate(
            {
                "sport": sport,
                "home_team": req.home_team,
                "away_team": req.away_team,
                "game_date": str(req.game_date),
                "prob_over": pred["prob_over"],
                "prob_under": pred["prob_under"],
                "confidence": pred["confidence"],
                "total_line": req.total_line,
                "predicted_direction": direction,
            }
        )
        if alert is not None:
            alert_repo = AlertRepository(db)
            await alert_repo.create(
                alert_id=alert.alert_id,
                sport=alert.sport,
                home_team=alert.home_team,
                away_team=alert.away_team,
                game_date=alert.game_date,
                direction=alert.direction,
                prob_over=alert.prob_over,
                confidence=alert.confidence,
                edge=alert.edge,
                total_line=alert.total_line,
                fired=alert.fired,
                webhook_response=alert.webhook_response,
                created_at=alert.created_at,
            )
    except Exception:
        pass  # alerts are best-effort

    return PredictionResponse(
        prediction_id=db_pred.id,
        game_id=game.id,
        sport=req.sport,
        home_team=req.home_team,
        away_team=req.away_team,
        game_date=req.game_date,
        total_line=req.total_line,
        model_version=model_version,
        prob_over=pred["prob_over"],
        prob_under=pred["prob_under"],
        ci_lower=pred["ci_lower"],
        ci_upper=pred["ci_upper"],
        predicted_direction=direction,
        confidence=pred["confidence"],
        latency_ms=latency_ms,
        features=features_snapshot,
    )


@router.get("/recent", response_model=list[dict])
async def get_recent_predictions(
    sport: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Recent predictions for the dashboard, ordered newest-first."""
    pred_repo = PredictionRepository(db)
    preds = await pred_repo.get_recent(sport=sport, limit=limit)
    return [
        {
            "prediction_id": str(p.id),
            "game_id": str(p.game_id),
            "sport": p.sport,
            "model_version": p.model_version,
            "prob_over": p.prob_over,
            "prob_under": p.prob_under,
            "predicted_direction": p.predicted_direction,
            "confidence": p.confidence,
            "predicted_at": p.predicted_at.isoformat() if p.predicted_at else None,
            "is_correct": p.is_correct,
            "clv": p.clv,
            "actual_total": p.actual_total,
            "closing_line": p.closing_line,
            "settled_at": p.settled_at.isoformat() if p.settled_at else None,
        }
        for p in preds
    ]


@router.get("/alerts/recent", response_model=list[dict])
async def get_recent_alerts(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Recent high-confidence alerts. Reads from DB for durability; falls back to in-memory."""
    try:
        alert_repo = AlertRepository(db)
        records = await alert_repo.get_recent(limit=limit)
        return [
            {
                "alert_id": r.alert_id,
                "sport": r.sport,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "game_date": r.game_date,
                "direction": r.direction,
                "confidence": r.confidence,
                "prob_over": r.prob_over,
                "edge": r.edge,
                "total_line": r.total_line,
                "fired": r.fired,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    except Exception:
        # DB unavailable — serve from in-memory deque
        mgr = get_alert_manager()
        return [
            {
                "alert_id": a.alert_id,
                "sport": a.sport,
                "home_team": a.home_team,
                "away_team": a.away_team,
                "game_date": a.game_date,
                "direction": a.direction,
                "confidence": a.confidence,
                "prob_over": a.prob_over,
                "edge": a.edge,
                "total_line": a.total_line,
                "fired": a.fired,
                "created_at": a.created_at.isoformat(),
            }
            for a in mgr.recent(limit)
        ]


@router.get("/{game_id}", response_model=list[PredictionResponse])
async def get_predictions_for_game(game_id: str, db: AsyncSession = Depends(get_db)):
    import uuid as _uuid

    try:
        gid = _uuid.UUID(game_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid game_id UUID")

    pred_repo = PredictionRepository(db)
    game_repo = GameRepository(db)

    game = await game_repo.get_by_id(gid)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    preds = await pred_repo.get_by_game(gid)
    return [
        PredictionResponse(
            prediction_id=p.id,
            game_id=game.id,
            sport=game.sport,
            home_team=game.home_team,
            away_team=game.away_team,
            game_date=game.game_date,
            total_line=game.total_line or 0,
            model_version=p.model_version,
            prob_over=p.prob_over,
            prob_under=p.prob_under,
            ci_lower=p.ci_lower,
            ci_upper=p.ci_upper,
            predicted_direction=p.predicted_direction,
            confidence=p.confidence,
            latency_ms=p.latency_ms or 0,
        )
        for p in preds
    ]


@router.post("/{prediction_id}/settle", response_model=SettleResponse)
async def settle_prediction(
    prediction_id: str,
    req: SettleRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record final score and closing line; compute CLV and correctness."""
    import uuid as _uuid

    try:
        pid = _uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction_id UUID")

    pred_repo = PredictionRepository(db)
    game_repo = GameRepository(db)
    pred = await pred_repo.get_by_id(pid)
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    game = await game_repo.get_by_id(pred.game_id)
    bet_line = game.total_line if game and game.total_line else req.closing_line
    await pred_repo.settle(
        prediction_id=pid,
        actual_total=req.actual_total,
        closing_line=req.closing_line,
        predicted_direction=pred.predicted_direction,
        bet_line=bet_line,
    )

    result_over = req.actual_total > req.closing_line
    is_correct = (pred.predicted_direction == "over") == result_over
    clv = (
        (req.closing_line - bet_line)
        if pred.predicted_direction == "over"
        else (bet_line - req.closing_line)
    )

    clv_desc = f"+{clv:.1f}" if clv >= 0 else f"{clv:.1f}"
    return SettleResponse(
        prediction_id=pid,
        actual_total=req.actual_total,
        closing_line=req.closing_line,
        clv=round(clv, 2),
        is_correct=is_correct,
        predicted_direction=pred.predicted_direction,
        message=f"{'✓ Correct' if is_correct else '✗ Wrong'} | CLV {clv_desc} | actual={req.actual_total} close={req.closing_line}",
    )


def _build_inference_features(
    req: PredictionRequest,
    feature_names: list[str],
    home_key_out: int = 0,
    away_key_out: int = 0,
    feature_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Constructs a single-row feature DataFrame for inference.
    Known signals from the request are filled explicitly. Unknown rolling
    features default to their training-set median (stored in model meta)
    rather than 0 so the model sees a realistic baseline for unseen matchups.
    """
    # Seed with training medians — much better default than zero for rolling stats
    medians = feature_medians or {}
    row: dict[str, float] = {f: medians.get(f, 0.0) for f in feature_names}

    # Core game context
    row["total_line"] = req.total_line
    row["home_advantage"] = 1.0

    if req.home_rest_days is not None:
        row["home_rest_days"] = float(req.home_rest_days)
    if req.away_rest_days is not None:
        row["away_rest_days"] = float(req.away_rest_days)

    if req.home_rest_days is not None and req.away_rest_days is not None:
        row["home_rest_advantage"] = float(req.home_rest_days - req.away_rest_days)
        row["home_back_to_back"] = float(req.home_rest_days <= 1)
        row["away_back_to_back"] = float(req.away_rest_days <= 1)

    # GROUP C — situational context
    row["wind_speed_mph"] = req.wind_speed_mph
    row["temperature_f"] = req.temperature_f
    row["is_dome"] = float(req.is_dome)
    row["altitude_feet"] = req.altitude_feet
    row["is_playoff"] = float(req.is_playoff)
    row["altitude_boost"] = req.altitude_feet / 5280.0
    row["dome_flag"] = float(req.is_dome)
    row["wind_under_signal"] = float(req.wind_speed_mph > 15)
    row["wind_severity"] = req.wind_speed_mph / 30.0
    row["cold_game"] = float(req.temperature_f < 40)
    row["hot_game"] = float(req.temperature_f > 85)

    # GROUP D — market / sharp signals
    row["line_movement"] = req.line_movement
    row["public_over_pct"] = req.public_over_pct
    row["sharp_over_pct"] = req.sharp_over_pct
    row["ref_foul_rate"] = req.ref_foul_rate
    row["ump_walk_rate"] = req.ump_walk_rate
    row["sharp_vs_public"] = req.sharp_over_pct - req.public_over_pct
    row["line_move_magnitude"] = abs(req.line_movement)
    row["line_move_direction"] = float(np.sign(req.line_movement))

    # GROUP E — injury counts (prefer auto-fetched values when caller sent 0)
    h_out = max(req.home_key_players_out, home_key_out)
    a_out = max(req.away_key_players_out, away_key_out)
    row["home_key_players_out"] = float(h_out)
    row["away_key_players_out"] = float(a_out)
    row["injury_pts_impact"] = (h_out - a_out) * -3.0

    # Legacy
    row["home_injury_impact"] = req.home_injury_impact
    row["away_injury_impact"] = req.away_injury_impact

    return pd.DataFrame([row])
