"""XGBoost + LightGBM ensemble for over/under prediction."""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from proedge.pipeline.models.calibration import IsotonicCalibrator

logger = logging.getLogger(__name__)


_TIGHT_PARAMS = {
    # Applied when training_games < 2000 (e.g. NFL synthetic or first-season real data).
    # Deep trees + many features on small datasets = catastrophic overfitting.
    "xgb": dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=10,
        gamma=0.5,
        reg_alpha=2.0,
        reg_lambda=5.0,
        early_stopping_rounds=20,
    ),
    "lgb": dict(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.03,
        feature_fraction=0.5,
        bagging_fraction=0.7,
        bagging_freq=5,
        min_child_samples=50,
        reg_alpha=2.0,
        reg_lambda=5.0,
    ),
}

_DEFAULT_PARAMS = {
    "xgb": dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
    ),
    "lgb": dict(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
    ),
}

_SMALL_DATA_THRESHOLD = 2000


class OverUnderEnsemble:
    """
    Weighted ensemble of XGBoost and LightGBM classifiers.
    Outputs calibrated over-probability with conformal prediction intervals.
    """

    def __init__(
        self,
        xgb_weight: float = 0.5,
        lgb_weight: float = 0.5,
        training_games: int = 10_000,
    ):
        assert abs(xgb_weight + lgb_weight - 1.0) < 1e-6, "weights must sum to 1"
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.feature_names: list[str] = []
        self.calibrator = IsotonicCalibrator()

        params = _TIGHT_PARAMS if training_games < _SMALL_DATA_THRESHOLD else _DEFAULT_PARAMS
        if training_games < _SMALL_DATA_THRESHOLD:
            logger.info(
                "Small dataset (%d games < %d threshold) — using tighter regularization",
                training_games,
                _SMALL_DATA_THRESHOLD,
            )

        self.xgb_model = xgb.XGBClassifier(
            **params["xgb"],
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.lgb_model = lgb.LGBMClassifier(
            **params["lgb"],
            random_state=42,
            verbosity=-1,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "OverUnderEnsemble":
        self.feature_names = list(X_train.columns)
        X_tr, X_v = X_train.values, X_val.values
        y_tr, y_v = y_train.values, y_val.values

        logger.info("Training XGBoost on %d samples", len(X_tr))
        self.xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            verbose=False,
        )

        logger.info("Training LightGBM on %d samples", len(X_tr))
        self.lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        # Fit calibrator on validation set predictions
        raw_val = self._raw_predict_proba(X_v)
        self.calibrator.fit(raw_val, y_v)

        # Log validation metrics
        cal_val = self.calibrator.transform(raw_val)
        logger.info(
            "Val metrics — AUC: %.4f | LogLoss: %.4f | Brier: %.4f",
            roc_auc_score(y_v, cal_val),
            log_loss(y_v, cal_val),
            brier_score_loss(y_v, cal_val),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated over-probability for each row in X."""
        X_arr = X[self.feature_names].values
        raw = self._raw_predict_proba(X_arr)
        return self.calibrator.transform(raw)

    def predict_with_intervals(
        self, X: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Returns list of {prob_over, prob_under, ci_lower, ci_upper, confidence}."""
        probs_over = self.predict_proba(X)
        results = []
        for p in probs_over:
            lo, hi = self.calibrator.prediction_interval(float(p))
            results.append({
                "prob_over": round(float(p), 4),
                "prob_under": round(1.0 - float(p), 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "confidence": round(abs(float(p) - 0.5) * 2, 4),  # 0=coin flip, 1=certain
            })
        return results

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        y_arr = y.values
        return {
            "accuracy": float((preds == y_arr).mean()),
            "auc": float(roc_auc_score(y_arr, probs)),
            "log_loss": float(log_loss(y_arr, probs)),
            "brier_score": float(brier_score_loss(y_arr, probs)),
        }

    def feature_importance(self) -> pd.DataFrame:
        xgb_imp = pd.Series(
            self.xgb_model.feature_importances_, index=self.feature_names, name="xgb"
        )
        lgb_imp = pd.Series(
            self.lgb_model.feature_importances_, index=self.feature_names, name="lgb"
        )
        df = pd.concat([xgb_imp, lgb_imp], axis=1)
        df["ensemble"] = self.xgb_weight * df["xgb"] + self.lgb_weight * df["lgb"]
        return df.sort_values("ensemble", ascending=False)

    def _raw_predict_proba(self, X: np.ndarray) -> np.ndarray:
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        return self.xgb_weight * xgb_prob + self.lgb_weight * lgb_prob
