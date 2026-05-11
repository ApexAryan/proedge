"""FastAPI application — entry point."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from proedge.api.middleware.auth import APIKeyMiddleware
from proedge.api.routers import backtest, health, lines, performance, predictions, training
from proedge.config import get_settings
from proedge.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger(__name__)
settings = get_settings()


async def _daily_update_loop():
    """
    Background task: runs DailyUpdater for each sport at 6 AM daily.
    Games typically finish by 1–2 AM ET; 6 AM gives ample margin.
    """
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        wait_secs = (next_run - now).total_seconds()
        logger.info(
            "Daily updater sleeping %.0f s until %s", wait_secs, next_run.strftime("%Y-%m-%d %H:%M")
        )
        await asyncio.sleep(wait_secs)

        for sport in settings.supported_sports:
            try:
                from proedge.pipeline.ingestion.daily_updater import DailyUpdater

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda s=sport: DailyUpdater(s, auto_retrain=True).run(),
                )
                logger.info(
                    "Daily update %s: +%d games | retrain=%s | error=%s",
                    sport.upper(),
                    result.games_added,
                    result.retrain_triggered,
                    result.error,
                )
            except Exception as exc:
                logger.exception("Daily update failed for %s: %s", sport, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from proedge.pipeline.models.registry import ModelRegistry
    from proedge.monitoring.metrics import (
        MODEL_ACCURACY,
        MODEL_LOG_LOSS,
        MODEL_BRIER_SCORE,
        ACTIVE_MODEL_VERSION,
    )

    registry = ModelRegistry()
    for sport in settings.supported_sports:
        try:
            from proedge.api.routers.predictions import _model_cache

            _model_cache[sport] = registry.load(sport)
            meta = registry.load_meta(sport)
            version = meta.get("version", "unknown")
            metrics = meta.get("metrics", {})
            if metrics.get("accuracy"):
                MODEL_ACCURACY.labels(sport=sport, model_version=version).set(metrics["accuracy"])
            if metrics.get("log_loss"):
                MODEL_LOG_LOSS.labels(sport=sport, model_version=version).set(metrics["log_loss"])
            if metrics.get("brier_score"):
                MODEL_BRIER_SCORE.labels(sport=sport, model_version=version).set(
                    metrics["brier_score"]
                )
            ACTIVE_MODEL_VERSION.labels(sport=sport, version=version).set(1)
        except FileNotFoundError:
            pass

    # Start background daily update task
    task = asyncio.create_task(_daily_update_loop())
    logger.info("Daily update scheduler started (runs at 06:00 daily)")

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="ProEdge Analytics API",
    description="Adaptive sports over/under prediction engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    latency = time.perf_counter() - t0

    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()
    response.headers["X-Response-Time-Ms"] = str(round(latency * 1000, 2))
    return response


_DASHBOARD_FILE = Path(__file__).parent / "static" / "dashboard.html"


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(_DASHBOARD_FILE)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "ProEdge Analytics API",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/dashboard",
    }


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(performance.router)
app.include_router(lines.router)
app.include_router(training.router)
app.include_router(backtest.router)
