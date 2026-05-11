"""Data drift detection using Population Stability Index (PSI) and KS test."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_NO_DRIFT = 0.10
PSI_MINOR_DRIFT = 0.25  # threshold to trigger retraining


def compute_psi(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """Population Stability Index between a reference and current distribution."""
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    ref_pct = ref_counts / max(len(reference), 1)
    cur_pct = cur_counts / max(len(current), 1)

    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test statistic and p-value."""
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)


class DriftDetector:
    """
    Monitors feature distributions against a stored reference window.
    Triggers retraining if PSI exceeds threshold for any critical feature.
    """

    def __init__(
        self,
        psi_threshold: float = PSI_MINOR_DRIFT,
        top_k_features: int = 20,
    ):
        self.psi_threshold = psi_threshold
        self.top_k_features = top_k_features
        self._reference: pd.DataFrame | None = None
        self._critical_features: list[str] = []

    def fit_reference(self, X: pd.DataFrame, feature_importance: pd.Series | None = None) -> None:
        self._reference = X.copy()
        if feature_importance is not None:
            self._critical_features = list(
                feature_importance.nlargest(self.top_k_features).index
            )
        else:
            self._critical_features = list(X.columns[: self.top_k_features])
        logger.info(
            "Drift reference set: %d rows, %d features monitored",
            len(X), len(self._critical_features),
        )

    def detect(self, X_current: pd.DataFrame) -> dict:
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before detect()")

        results: dict[str, dict] = {}
        drift_triggered = False

        for feat in self._critical_features:
            if feat not in self._reference.columns or feat not in X_current.columns:
                continue

            ref_vals = self._reference[feat].dropna().values
            cur_vals = X_current[feat].dropna().values
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            psi = compute_psi(ref_vals, cur_vals)
            ks_stat, ks_p = compute_ks(ref_vals, cur_vals)

            drifted = psi >= self.psi_threshold
            if drifted:
                drift_triggered = True
                logger.warning("Drift detected on '%s': PSI=%.3f", feat, psi)

            results[feat] = {
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_p, 4),
                "drifted": drifted,
                "severity": _psi_severity(psi),
            }

        return {
            "retrain_triggered": drift_triggered,
            "features_checked": len(results),
            "features_drifted": sum(1 for v in results.values() if v["drifted"]),
            "feature_details": results,
        }


def _psi_severity(psi: float) -> str:
    if psi < PSI_NO_DRIFT:
        return "none"
    if psi < PSI_MINOR_DRIFT:
        return "minor"
    return "major"
