import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.api.schemas import HealthResponse
from proedge.config import get_settings
from proedge.db.session import get_db

router = APIRouter(tags=["health"])
settings = get_settings()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    # Check which sports have a registered and loadable model
    from proedge.pipeline.models.registry import ModelRegistry

    registry = ModelRegistry()
    models_loaded: dict[str, str | None] = {}
    for sport in settings.supported_sports:
        try:
            version = registry.latest_version(sport)
            if version:
                # Verify the artifact is actually on disk
                registry.load_meta(sport)
            models_loaded[sport] = version
        except Exception:
            models_loaded[sport] = None

    any_model_ready = any(v is not None for v in models_loaded.values())
    if not db_ok:
        status_str = "degraded"
    elif not any_model_ready:
        status_str = "no_models"
    else:
        status_str = "ok"

    return HealthResponse(
        status=status_str,
        db_connected=db_ok,
        models_loaded=models_loaded,
        uptime_seconds=round(time.time() - _start_time, 1),
        version="0.1.0",
    )


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Database unavailable")
