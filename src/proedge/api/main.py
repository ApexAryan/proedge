"""FastAPI application — entry point."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from proedge.api.routers import health, performance, predictions
from proedge.config import get_settings
from proedge.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from proedge.pipeline.models.registry import ModelRegistry
    from proedge.monitoring.metrics import MODEL_ACCURACY, MODEL_LOG_LOSS, MODEL_BRIER_SCORE, ACTIVE_MODEL_VERSION
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
                MODEL_BRIER_SCORE.labels(sport=sport, model_version=version).set(metrics["brier_score"])
            ACTIVE_MODEL_VERSION.labels(sport=sport, version=version).set(1)
        except FileNotFoundError:
            pass
    yield


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


@app.get("/", include_in_schema=False)
async def root():
    return {"name": "ProEdge Analytics API", "docs": "/docs", "health": "/health"}


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(performance.router)
