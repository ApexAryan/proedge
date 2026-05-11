"""Training management endpoints: manual daily updates and force retrains."""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])

_STATUS_FILE = Path("./data/update_status.json")

# In-process lock so two concurrent retrain requests don't collide
_retrain_locks: dict[str, asyncio.Lock] = {}


def _lock_for(sport: str) -> asyncio.Lock:
    if sport not in _retrain_locks:
        _retrain_locks[sport] = asyncio.Lock()
    return _retrain_locks[sport]


# ── Response schemas ──────────────────────────────────────────────────────────

class UpdateResponse(BaseModel):
    sport: str
    date: str
    games_found: int
    games_added: int
    games_skipped: int
    retrain_triggered: bool
    retrain_metrics: dict
    error: str | None = None
    message: str = ""


class RetrainResponse(BaseModel):
    sport: str
    version: str
    accuracy: float | None = None
    auc: float | None = None
    log_loss: float | None = None
    brier_score: float | None = None
    training_games: int | None = None
    feature_count: int | None = None
    message: str = ""


class TrainingStatusResponse(BaseModel):
    sport: str
    last_update_date: str | None
    last_update_games_added: int | None
    last_retrain_version: str | None
    last_retrain_at: str | None
    total_historical_games: int | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/update/{sport}",
    response_model=UpdateResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch yesterday's completed games and append to training data",
    description=(
        "Pulls completed game results, derives injury counts from box-score DNP comments, "
        "appends to the historical parquet, and clears the feature cache. "
        "Pass `date` (YYYY-MM-DD) to backfill a specific day."
    ),
)
async def run_daily_update(
    sport: str,
    target_date: str | None = Query(
        None,
        alias="date",
        description="YYYY-MM-DD (default: yesterday)",
    ),
    auto_retrain: bool = Query(
        False,
        description=f"Trigger a full retrain if ≥{30} new games have accumulated",
    ),
):
    sport = sport.lower()
    _validate_sport(sport)

    parsed_date: date | None = None
    if target_date:
        try:
            parsed_date = date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid date format: {target_date}")

    from proedge.pipeline.ingestion.daily_updater import DailyUpdater
    updater = DailyUpdater(sport, auto_retrain=auto_retrain)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, updater.run, parsed_date)

    _save_update_status(sport, result.__dict__)

    return UpdateResponse(
        sport=result.sport,
        date=result.date,
        games_found=result.games_found,
        games_added=result.games_added,
        games_skipped=result.games_skipped,
        retrain_triggered=result.retrain_triggered,
        retrain_metrics=result.retrain_metrics,
        error=result.error,
        message=(
            f"Added {result.games_added} new games."
            if result.games_added
            else "No new games found."
        ),
    )


@router.post(
    "/retrain/{sport}",
    response_model=RetrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Force a full model retrain from the current historical data",
)
async def force_retrain(sport: str):
    sport = sport.lower()
    _validate_sport(sport)

    lock = _lock_for(sport)
    if lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Retrain already in progress for {sport}",
        )

    async with lock:
        from proedge.pipeline.training.trainer import train
        loop = asyncio.get_running_loop()
        try:
            metrics = await loop.run_in_executor(None, train, sport)
        except Exception as exc:
            logger.exception("Retrain failed for %s", sport)
            raise HTTPException(status_code=500, detail=str(exc))

    # Reload model in prediction cache
    _reload_model_cache(sport)

    return RetrainResponse(
        sport=sport,
        version=metrics.get("version", "unknown"),
        accuracy=metrics.get("accuracy"),
        auc=metrics.get("auc"),
        log_loss=metrics.get("log_loss"),
        brier_score=metrics.get("brier_score"),
        training_games=metrics.get("training_games"),
        feature_count=metrics.get("feature_count"),
        message=f"Retrain complete — version {metrics.get('version')}",
    )


@router.get(
    "/status/{sport}",
    response_model=TrainingStatusResponse,
    summary="Last update and retrain status for a sport",
)
async def get_training_status(sport: str):
    sport = sport.lower()
    _validate_sport(sport)

    saved = _load_update_status(sport)
    historical_games = _count_historical(sport)

    from proedge.pipeline.models.registry import ModelRegistry
    try:
        meta = ModelRegistry().load_meta(sport)
        retrain_version = meta.get("version")
        retrain_at = meta.get("trained_at")
    except Exception:
        retrain_version = None
        retrain_at = None

    return TrainingStatusResponse(
        sport=sport,
        last_update_date=saved.get("date"),
        last_update_games_added=saved.get("games_added"),
        last_retrain_version=retrain_version,
        last_retrain_at=retrain_at,
        total_historical_games=historical_games,
    )


@router.get(
    "/status",
    summary="Training status for all sports",
)
async def get_all_training_status():
    statuses = await asyncio.gather(
        get_training_status("nba"),
        get_training_status("nfl"),
        get_training_status("mlb"),
    )
    return {s.sport: s.__dict__ for s in statuses}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_sport(sport: str):
    from proedge.config import get_settings
    valid = get_settings().supported_sports
    if sport not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported sport '{sport}'. Choose: {' | '.join(valid)}.",
        )


def _save_update_status(sport: str, data: dict):
    import json
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if _STATUS_FILE.exists():
        try:
            existing = json.loads(_STATUS_FILE.read_text())
        except Exception as exc:
            logger.warning("Could not read update status file, resetting: %s", exc)
    existing[sport] = {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                       for k, v in data.items()}
    _STATUS_FILE.write_text(json.dumps(existing, indent=2))


def _load_update_status(sport: str) -> dict:
    import json
    if _STATUS_FILE.exists():
        try:
            return json.loads(_STATUS_FILE.read_text()).get(sport, {})
        except Exception:
            pass
    return {}


def _count_historical(sport: str) -> int | None:
    import pandas as pd
    path = Path(f"./data/{sport}_historical.parquet")
    if path.exists():
        try:
            return len(pd.read_parquet(path, columns=["game_id"]))
        except Exception:
            pass
    return None


def _reload_model_cache(sport: str):
    try:
        from proedge.api.routers.predictions import _model_cache
        from proedge.pipeline.models.registry import ModelRegistry
        _model_cache[sport] = ModelRegistry().load(sport)
        logger.info("Reloaded %s model into prediction cache", sport.upper())
    except Exception as exc:
        logger.warning("Could not reload model cache for %s: %s", sport, exc)
