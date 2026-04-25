import time
from datetime import datetime

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

    # Check which sports have a registered model
    from proedge.pipeline.models.registry import ModelRegistry
    registry = ModelRegistry()
    models_loaded: dict[str, str | None] = {}
    for sport in settings.supported_sports:
        models_loaded[sport] = registry.latest_version(sport)

    return HealthResponse(
        status="ok" if db_ok else "degraded",
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
