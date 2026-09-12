"""Restricted public portfolio adapter over ProEdge's real market scanner."""
from __future__ import annotations
import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from fastapi import APIRouter, Header, HTTPException
from proedge.api.routers.predictions import scan_markets
from proedge.api.schemas import MarketScanRequest
from proedge.db.session import AsyncSessionLocal

router = APIRouter(prefix="/demo", tags=["portfolio-demo"])
_lock = asyncio.Lock()
_cached: Any = None
_generated: datetime | None = None

def _auth(token: str, visitor: str) -> None:
    if not os.environ.get("PORTFOLIO_DEMO_TOKEN") or token != os.environ["PORTFOLIO_DEMO_TOKEN"] or len(visitor) != 64: raise HTTPException(401, "Unauthorized")

def _envelope(data: Any, cached: bool) -> dict[str, Any]:
    return {"status": "cached" if cached else "ready", "data": data, "generatedAt": _generated.isoformat() if _generated else None, "cached": cached, "source": "real_cached" if cached else "live_provider", "phase": None, "progress": 100, "error": None}

async def _refresh() -> dict[str, Any]:
    global _cached, _generated
    if _lock.locked():
        return {"status": "running", "data": _cached, "generatedAt": _generated.isoformat() if _generated else None, "cached": _cached is not None, "source": "real_cached" if _cached else "live_provider", "phase": "Another visitor refresh is running", "progress": 10, "retryAfter": 15, "error": None}
    async with _lock:
        if _generated and datetime.now(UTC) - _generated < timedelta(minutes=15): return _envelope(_cached, True)
        async with AsyncSessionLocal() as db:
            result = await scan_markets(MarketScanRequest(sports=["nba", "nfl", "mlb"]), db)
            await db.rollback()
        _cached = result.model_dump(mode="json")
        _generated = datetime.now(UTC)
        return _envelope(_cached, False)

@router.get("/bootstrap")
async def bootstrap(x_demo_token: str = Header(""), x_demo_visitor: str = Header("")) -> dict[str, Any]:
    _auth(x_demo_token, x_demo_visitor)
    # Keep first paint fast after a free-tier cold start. The explicit refresh
    # operation owns provider access and the global cooldown.
    return _envelope(_cached if _cached is not None else {"results": []}, _cached is not None)

@router.post("/refresh")
async def refresh(x_demo_token: str = Header(""), x_demo_visitor: str = Header("")) -> dict[str, Any]:
    _auth(x_demo_token, x_demo_visitor)
    return await _refresh()
