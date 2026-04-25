"""Prediction endpoints — the primary API surface."""
from __future__ import annotations

import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.api.schemas import PredictionRequest, PredictionResponse
from proedge.db.repositories import GameRepository, PredictionRepository
from proedge.db.session import get_db
from proedge.monitoring.metrics import (
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_PROB_OVER,
)
from proedge.pipeline.models.registry import ModelRegistry

router = APIRouter(prefix="/predictions", tags=["predictions"])
_registry = ModelRegistry()
_model_cache: dict[str, object] = {}


def _get_model(sport: str):
    if sport not in _model_cache:
        try:
            _model_cache[sport] = _registry.load(sport)
        except FileNotFoundError:
            return None
    return _model_cache[sport]


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction(
    req: PredictionRequest, db: AsyncSession = Depends(get_db)
):
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

    # Build inference feature row with available context
    X = _build_inference_features(req, feature_names)

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


def _build_inference_features(
    req: PredictionRequest, feature_names: list[str]
) -> pd.DataFrame:
    """
    Constructs a single-row feature DataFrame for inference.
    Known signals from the request are filled; unknowns default to 0
    (model was trained with these as 0 for unseen matchups).
    """
    row: dict[str, float] = {f: 0.0 for f in feature_names}

    # Direct signals from request
    row["total_line"] = req.total_line
    row["home_injury_impact"] = req.home_injury_impact
    row["away_injury_impact"] = req.away_injury_impact
    row["home_advantage"] = 1.0

    if req.home_rest_days is not None:
        row["home_rest_days"] = float(req.home_rest_days)
    if req.away_rest_days is not None:
        row["away_rest_days"] = float(req.away_rest_days)

    if req.home_rest_days is not None and req.away_rest_days is not None:
        row["home_rest_advantage"] = float(req.home_rest_days - req.away_rest_days)
        row["home_back_to_back"] = float(req.home_rest_days <= 1)
        row["away_back_to_back"] = float(req.away_rest_days <= 1)

    return pd.DataFrame([row])
