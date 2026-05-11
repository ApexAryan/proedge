"""Model performance and drift monitoring endpoints."""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.api.schemas import ModelPerformanceResponse
from proedge.config import get_settings
from proedge.db.repositories import PredictionRepository
from proedge.db.session import get_db
from proedge.monitoring.metrics import DRIFT_PSI
from proedge.pipeline.models.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["models"])
settings = get_settings()
_registry = ModelRegistry()


def _meta_to_response(meta: dict, sport: str) -> ModelPerformanceResponse:
    metrics = meta.get("metrics", {})
    trained_at = meta.get("trained_at", datetime.now(timezone.utc).isoformat())
    if isinstance(trained_at, str):
        trained_at = datetime.fromisoformat(trained_at)
    return ModelPerformanceResponse(
        version=meta.get("version", "unknown"),
        sport=sport,
        accuracy=metrics.get("accuracy"),
        log_loss=metrics.get("log_loss"),
        brier_score=metrics.get("brier_score"),
        training_games=metrics.get("training_games")
        or metrics.get("holdout_games")
        or meta.get("training_games"),
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
        try:
            meta = _registry.load_meta(sport)
            version = meta.get("version", "unknown")
        except FileNotFoundError:
            result[sport] = {"accuracy": None, "total": 0}
            continue
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


@router.post("/drift-check/{sport}", response_model=dict, status_code=200)
async def run_drift_check(
    sport: str,
    recent_games: int = Query(
        200,
        ge=50,
        description="Number of most-recent games used as the 'current' window",
    ),
):
    """Compare recent feature distributions against the training reference.

    Emits proedge_drift_psi Prometheus metrics (visible in Grafana) and returns
    per-feature PSI / KS results. CPU-heavy — runs in a thread pool.
    """
    if sport not in settings.supported_sports:
        raise HTTPException(status_code=400, detail=f"Unsupported sport: {sport}")

    loop = asyncio.get_running_loop()
    try:
        report = await loop.run_in_executor(None, lambda: _run_drift(sport, recent_games))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return report


def _run_drift(sport: str, recent_games: int) -> dict:
    from proedge.pipeline.features.store import FeatureStore
    from proedge.pipeline.ingestion.historical import HistoricalLoader
    from proedge.pipeline.models.drift import DriftDetector

    meta = _registry.load_meta(sport)
    model = _registry.load(sport)
    feature_cols: list[str] = meta.get("feature_names", [])

    loader = HistoricalLoader()
    df = loader.load(sport)
    store = FeatureStore()
    feature_df = store.compute(df, sport, use_cache=True)

    cols = feature_cols or list(feature_df.columns)
    X = feature_df[cols].fillna(0)
    n = len(X)
    split = max(100, n - recent_games)
    X_ref = X.iloc[:split]
    X_cur = X.iloc[split:]

    if X_ref.empty or X_cur.empty:
        return {"error": "insufficient data for drift check", "features_checked": 0}

    importance = model.feature_importance()["ensemble"]
    detector = DriftDetector(psi_threshold=settings.drift_psi_threshold)
    detector.fit_reference(X_ref, feature_importance=importance)
    report = detector.detect(X_cur)

    for feat, detail in report.get("feature_details", {}).items():
        DRIFT_PSI.labels(sport=sport, feature=feat).set(detail["psi"])

    return report
