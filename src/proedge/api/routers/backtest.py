"""Backtesting endpoints — replay historical predictions to measure edge and ROI."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])

def _valid_sports() -> set[str]:
    from proedge.config import get_settings
    return set(get_settings().supported_sports)


# ── Response schemas ──────────────────────────────────────────────────────────

class FoldResultResponse(BaseModel):
    fold: int
    start_date: str
    end_date: str
    n_games: int
    accuracy: float
    auc: float
    log_loss: float
    brier_score: float
    roi_flat: float
    roi_kelly: float
    edge_mean: float
    high_conf_accuracy: float | None = None


class BacktestResponse(BaseModel):
    sport: str
    n_folds: int
    min_confidence: float
    total_games: int
    total_bets: int
    overall_accuracy: float
    overall_auc: float
    overall_roi_flat: float
    overall_roi_kelly: float
    sharpe_ratio: float
    max_drawdown: float
    folds: list[FoldResultResponse]
    calibration: dict[str, list[Any]]
    message: str = ""


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/{sport}",
    response_model=BacktestResponse,
    status_code=status.HTTP_200_OK,
    summary="Walk-forward backtest for a sport",
    description=(
        "Runs an expanding-window walk-forward backtest on the current historical data. "
        "Features are computed once with shift(1) to prevent lookahead leakage. "
        "Reports flat-bet ROI at -110 juice, Kelly ROI, Sharpe, and calibration curve. "
        "Warning: this is CPU-heavy — may take several minutes for large datasets."
    ),
)
async def run_backtest(
    sport: str,
    n_folds: int = Query(5, ge=2, le=10, description="Number of walk-forward folds"),
    min_train_games: int = Query(
        500, ge=100, description="Minimum training games before first prediction fold"
    ),
    min_confidence: float = Query(
        0.0, ge=0.0, le=1.0,
        description="Only include predictions with confidence >= this value in ROI calculations",
    ),
):
    sport = sport.lower()
    valid = _valid_sports()
    if sport not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported sport '{sport}'. Choose: {sorted(valid)}",
        )

    from proedge.pipeline.backtesting.backtester import Backtester

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: Backtester(sport).run(
                n_folds=n_folds,
                min_train_games=min_train_games,
                min_confidence=min_confidence,
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Backtest failed for %s", sport)
        raise HTTPException(status_code=500, detail=str(exc))

    folds_resp = [
        FoldResultResponse(
            fold=f.fold,
            start_date=f.start_date,
            end_date=f.end_date,
            n_games=f.n_games,
            accuracy=f.accuracy,
            auc=f.auc,
            log_loss=f.log_loss,
            brier_score=f.brier_score,
            roi_flat=f.roi_flat,
            roi_kelly=f.roi_kelly,
            edge_mean=f.edge_mean,
            high_conf_accuracy=None if math.isnan(f.high_conf_accuracy) else f.high_conf_accuracy,
        )
        for f in result.folds
    ]

    roi_pct = round(result.overall_roi_flat * 100, 2)
    msg = (
        f"{result.total_bets} bets across {result.total_games} games | "
        f"Acc={result.overall_accuracy:.3f} AUC={result.overall_auc:.3f} "
        f"ROI_flat={roi_pct:+.1f}% Sharpe={result.sharpe_ratio:.2f}"
    )

    return BacktestResponse(
        sport=result.sport,
        n_folds=result.n_folds,
        min_confidence=result.min_confidence,
        total_games=result.total_games,
        total_bets=result.total_bets,
        overall_accuracy=result.overall_accuracy,
        overall_auc=result.overall_auc,
        overall_roi_flat=result.overall_roi_flat,
        overall_roi_kelly=result.overall_roi_kelly,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        folds=folds_resp,
        calibration=result.calibration,
        message=msg,
    )
