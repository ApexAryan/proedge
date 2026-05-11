"""Isotonic regression calibrator with conformal prediction intervals."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Wraps sklearn IsotonicRegression for probability calibration."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._is_fitted: bool = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(raw_probs, labels)
        self._is_fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        return self._iso.predict(raw_probs)

