"""End-to-end training pipeline: ingest → features → train → evaluate → register."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from proedge.config import get_settings
from proedge.monitoring.metrics import DRIFT_PSI, RETRAIN_COUNTER
from proedge.pipeline.features.store import FeatureStore
from proedge.pipeline.ingestion.historical import HistoricalLoader
from proedge.pipeline.models.drift import DriftDetector
from proedge.pipeline.models.ensemble import OverUnderEnsemble
from proedge.pipeline.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

HOLDOUT_FRAC = 0.15  # ~last 15% of games for final evaluation
VAL_FRAC = 0.15      # validation set carved from training portion
MIN_TRAIN_GAMES = 500


def train(sport: str, xgb_weight: float = 0.5, trigger_reason: str = "manual") -> dict:
    logger.info("=== ProEdge Training Pipeline | sport=%s ===", sport)

    # 1. Load historical data
    loader = HistoricalLoader()
    df = loader.load(sport)
    logger.info("Loaded %d historical games for %s", len(df), sport)

    if len(df) < MIN_TRAIN_GAMES:
        raise ValueError(f"Not enough data: {len(df)} games (need {MIN_TRAIN_GAMES})")

    # 2. Feature engineering
    store = FeatureStore()
    feature_df = store.compute(df, sport)
    feature_cols = store.get_feature_columns(feature_df)

    X = feature_df[feature_cols].fillna(0)
    y = feature_df["result_over"].astype(int)

    logger.info("Feature matrix: %d rows × %d features", len(X), len(feature_cols))

    # 3. Time-aware train / val / holdout split
    n = len(X)
    holdout_start = int(n * (1 - HOLDOUT_FRAC))
    val_start = int(holdout_start * (1 - VAL_FRAC))

    X_train = X.iloc[:val_start]
    y_train = y.iloc[:val_start]
    X_val = X.iloc[val_start:holdout_start]
    y_val = y.iloc[val_start:holdout_start]
    X_holdout = X.iloc[holdout_start:]
    y_holdout = y.iloc[holdout_start:]

    logger.info(
        "Split — train: %d | val: %d | holdout: %d",
        len(X_train), len(X_val), len(X_holdout),
    )

    # 4. Train ensemble — pass training_games so small datasets get tighter regularization
    model = OverUnderEnsemble(
        xgb_weight=xgb_weight,
        lgb_weight=1 - xgb_weight,
        training_games=len(X_train),
    )
    model.fit(X_train, y_train, X_val, y_val)

    # 5. Evaluate on holdout (10,000+ games in production)
    holdout_metrics = model.evaluate(X_holdout, y_holdout)
    logger.info(
        "Holdout metrics — accuracy: %.4f | AUC: %.4f | LogLoss: %.4f | Brier: %.4f",
        holdout_metrics["accuracy"],
        holdout_metrics["auc"],
        holdout_metrics["log_loss"],
        holdout_metrics["brier_score"],
    )

    # Directional accuracy lift over baseline (50% naive predictor)
    baseline_accuracy = 0.50
    lift_pct = (holdout_metrics["accuracy"] - baseline_accuracy) / baseline_accuracy * 100
    logger.info("Directional accuracy lift over baseline: +%.1f%%", lift_pct)

    # 6. Set drift detector reference
    detector = DriftDetector(psi_threshold=settings.drift_psi_threshold)
    importance = model.feature_importance()["ensemble"]
    detector.fit_reference(X_train, feature_importance=importance)

    # Compute feature medians from training data — used at inference time as
    # defaults instead of 0.0 so the model sees realistic context for unseen matchups.
    feature_medians = {col: round(float(X_train[col].median()), 6) for col in feature_cols}

    # 7. Register model
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    registry = ModelRegistry()
    model_path = registry.save(
        model=model,
        sport=sport,
        version=version,
        metrics={**holdout_metrics, "lift_pct": round(lift_pct, 2), "training_games": len(X_train), "holdout_games": len(X_holdout)},
        feature_names=feature_cols,
        feature_medians=feature_medians,
    )

    RETRAIN_COUNTER.labels(sport=sport, trigger_reason=trigger_reason).inc()

    _persist_model_run(
        sport=sport,
        version=version,
        model_path=str(model_path),
        metrics=holdout_metrics,
        feature_count=len(feature_cols),
        xgb_weight=xgb_weight,
    )

    result = {
        "sport": sport,
        "version": version,
        "model_path": model_path,
        "feature_count": len(feature_cols),
        "training_games": len(X_train),
        "holdout_games": len(X_holdout),
        **holdout_metrics,
        "lift_pct": round(lift_pct, 2),
    }
    logger.info("Training complete: %s", result)
    return result


def _persist_model_run(
    sport: str,
    version: str,
    model_path: str,
    metrics: dict,
    feature_count: int,
    xgb_weight: float,
) -> None:
    try:
        from sqlalchemy import update as sa_update
        from proedge.db.models import ModelRun
        from proedge.db.session import SyncSessionLocal
        with SyncSessionLocal() as session:
            session.execute(
                sa_update(ModelRun).where(ModelRun.sport == sport).values(is_active=False)
            )
            session.add(ModelRun(
                version=version,
                sport=sport,
                accuracy=metrics.get("accuracy"),
                log_loss=metrics.get("log_loss"),
                brier_score=metrics.get("brier_score"),
                training_games=metrics.get("training_games"),
                feature_count=feature_count,
                xgb_weight=xgb_weight,
                lgb_weight=1.0 - xgb_weight,
                model_path=model_path,
                is_active=True,
            ))
            session.commit()
        logger.info("ModelRun persisted: %s (%s)", version, sport)
    except Exception as exc:
        logger.warning("Could not persist ModelRun to DB: %s", exc)


def check_and_retrain(sport: str, X_current: pd.DataFrame) -> bool:
    """Run drift check on current data; retrain if threshold exceeded."""
    registry = ModelRegistry()
    meta = registry.load_meta(sport)
    if not meta:
        logger.info("No existing model for %s — training from scratch", sport)
        train(sport)
        return True

    model = registry.load(sport)
    feature_cols = meta.get("feature_names", [])

    store = FeatureStore()
    loader = HistoricalLoader()
    df_ref = loader.load(sport)
    feature_df_ref = store.compute(df_ref, sport, use_cache=True)
    X_ref = feature_df_ref[feature_cols].fillna(0)

    detector = DriftDetector(psi_threshold=settings.drift_psi_threshold)
    importance = model.feature_importance()["ensemble"]
    detector.fit_reference(X_ref, feature_importance=importance)

    X_cur = X_current[feature_cols].fillna(0) if feature_cols else X_current
    drift_report = detector.detect(X_cur)

    for feat, detail in drift_report.get("feature_details", {}).items():
        DRIFT_PSI.labels(sport=sport, feature=feat).set(detail["psi"])

    logger.info(
        "Drift check: %d/%d features drifted | retrain=%s",
        drift_report["features_drifted"],
        drift_report["features_checked"],
        drift_report["retrain_triggered"],
    )

    if drift_report["retrain_triggered"]:
        logger.info("Retraining triggered for %s", sport)
        train(sport, trigger_reason="drift_psi")
        return True
    return False


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="ProEdge model trainer")
    parser.add_argument("--sport", choices=["nfl", "nba", "mlb"], required=True)
    parser.add_argument("--xgb-weight", type=float, default=0.5)
    args = parser.parse_args()
    train(args.sport, xgb_weight=args.xgb_weight)
