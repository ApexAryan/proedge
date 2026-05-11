"""Isotonic regression calibrator with conformal prediction intervals."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """
    Wraps sklearn IsotonicRegression for probability calibration.
    Also stores holdout residuals for conformal prediction intervals.
    """

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._residual_p5: float = 0.0
        self._residual_p95: float = 0.0
        self._is_fitted: bool = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(raw_probs, labels)
        calibrated = self._iso.predict(raw_probs)
        residuals = np.abs(calibrated - labels.astype(float))
        self._residual_p5 = float(np.percentile(residuals, 5))
        self._residual_p95 = float(np.percentile(residuals, 95))
        self._is_fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        return self._iso.predict(raw_probs)

    def prediction_interval(
        self, calibrated_prob: float
    ) -> tuple[float, float]:
        """Return (ci_lower, ci_upper) for a single calibrated probability."""
        lower = float(np.clip(calibrated_prob - self._residual_p95, 0.0, 1.0))
        upper = float(np.clip(calibrated_prob + self._residual_p95, 0.0, 1.0))
        return lower, upper
