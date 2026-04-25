"""Model performance and drift monitoring endpoints."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.api.schemas import ModelPerformanceResponse
from proedge.config import get_settings
from proedge.db.repositories import PredictionRepository
from proedge.db.session import get_db
from proedge.pipeline.models.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["models"])
settings = get_settings()
_registry = ModelRegistry()


def _meta_to_response(meta: dict, sport: str) -> ModelPerformanceResponse:
    metrics = meta.get("metrics", {})
    trained_at = meta.get("trained_at", datetime.utcnow().isoformat())
    if isinstance(trained_at, str):
        trained_at = datetime.fromisoformat(trained_at)
    return ModelPerformanceResponse(
        version=meta.get("version", "unknown"),
        sport=sport,
        accuracy=metrics.get("accuracy"),
        log_loss=metrics.get("log_loss"),
        brier_score=metrics.get("brier_score"),
        training_games=metrics.get("training_games") or metrics.get("holdout_games") or meta.get("training_games"),
        feature_count=meta.get("feature_count"),
        xgb_weight=meta.get("xgb_weight", 0.5),
        lgb_weight=meta.get("lgb_weight", 0.5),
        trained_at=trained_at,
        is_active=True,
    )


@router.get("/performance", response_model=list[ModelPerformanceResponse])
async def get_model_performance():
    """Returns metrics for all registered model versions across sports."""
    results = []
    for sport in settings.supported_sports:
        for meta in _registry.list_versions(sport):
            results.append(_meta_to_response(meta, sport))
    return results


@router.get("/performance/{sport}", response_model=list[ModelPerformanceResponse])
async def get_sport_performance(sport: str):
    if sport not in settings.supported_sports:
        raise HTTPException(status_code=400, detail=f"Unsupported sport: {sport}")
    return [_meta_to_response(m, sport) for m in _registry.list_versions(sport)]


@router.get("/accuracy/live", response_model=dict)
async def live_accuracy(db: AsyncSession = Depends(get_db)):
    """Rolling accuracy across all resolved predictions persisted to the DB."""
    pred_repo = PredictionRepository(db)
    result = {}
    for sport in settings.supported_sports:
        meta = _registry.load_meta(sport)
        version = meta.get("version", "unknown")
        result[sport] = await pred_repo.accuracy_by_version(version, sport)
    return result


@router.get("/feature-importance/{sport}", response_model=dict)
async def feature_importance(sport: str):
    if sport not in settings.supported_sports:
        raise HTTPException(status_code=400, detail=f"Unsupported sport: {sport}")
    try:
        model = _registry.load(sport)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No model found for {sport}")
    imp = model.feature_importance()
    return {
        "sport": sport,
        "top_features": imp.head(20)[["xgb", "lgb", "ensemble"]].to_dict(),
    }
