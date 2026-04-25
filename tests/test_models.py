import numpy as np
import pandas as pd
import pytest

from proedge.pipeline.models.calibration import IsotonicCalibrator
from proedge.pipeline.models.drift import DriftDetector, compute_psi
from proedge.pipeline.models.ensemble import OverUnderEnsemble
from proedge.pipeline.models.registry import ModelRegistry


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibrator_fit_transform():
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, 500)
    labels = (raw + rng.normal(0, 0.2, 500) > 0.5).astype(int)

    cal = IsotonicCalibrator()
    cal.fit(raw, labels)
    probs = cal.transform(raw)

    assert probs.shape == raw.shape
    assert (probs >= 0).all() and (probs <= 1).all()


def test_calibrator_prediction_interval():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 200)
    labels = (raw > 0.5).astype(int)

    cal = IsotonicCalibrator()
    cal.fit(raw, labels)

    lo, hi = cal.prediction_interval(0.6)
    assert lo <= 0.6 <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_calibrator_not_fitted_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="fitted"):
        cal.transform(np.array([0.5]))


# ── Drift Detection ───────────────────────────────────────────────────────────

def test_psi_identical_distributions():
    arr = np.random.default_rng(42).normal(0, 1, 1000)
    psi = compute_psi(arr, arr)
    assert psi < 0.01  # identical → near-zero PSI


def test_psi_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(2, 1, 1000)  # major shift
    psi = compute_psi(ref, cur)
    assert psi > 0.25  # should trigger drift


def test_drift_detector_no_drift():
    rng = np.random.default_rng(0)
    X_ref = pd.DataFrame({"feat_a": rng.normal(0, 1, 500), "feat_b": rng.normal(5, 2, 500)})
    X_cur = pd.DataFrame({"feat_a": rng.normal(0, 1, 100), "feat_b": rng.normal(5, 2, 100)})

    detector = DriftDetector(psi_threshold=0.25)
    detector.fit_reference(X_ref)
    report = detector.detect(X_cur)

    assert not report["retrain_triggered"]
    assert report["features_checked"] == 2


def test_drift_detector_triggers_on_drift():
    rng = np.random.default_rng(0)
    X_ref = pd.DataFrame({"feat_a": rng.normal(0, 1, 500)})
    X_cur = pd.DataFrame({"feat_a": rng.normal(10, 1, 100)})  # massive drift

    detector = DriftDetector(psi_threshold=0.25)
    detector.fit_reference(X_ref)
    report = detector.detect(X_cur)

    assert report["retrain_triggered"]
    assert report["features_drifted"] >= 1


# ── Ensemble ──────────────────────────────────────────────────────────────────

def test_ensemble_fit_predict(trained_model, sample_feature_matrix):
    model, feature_cols = trained_model
    X = sample_feature_matrix[feature_cols].fillna(0)

    probs = model.predict_proba(X.iloc[:10])
    assert probs.shape == (10,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_ensemble_predict_with_intervals(trained_model, sample_feature_matrix):
    model, feature_cols = trained_model
    X = sample_feature_matrix[feature_cols].fillna(0)

    results = model.predict_with_intervals(X.iloc[:5])
    assert len(results) == 5
    for r in results:
        assert "prob_over" in r and "prob_under" in r
        assert "ci_lower" in r and "ci_upper" in r
        assert abs(r["prob_over"] + r["prob_under"] - 1.0) < 1e-6
        assert r["ci_lower"] <= r["prob_over"] <= r["ci_upper"]


def test_ensemble_evaluate_returns_metrics(trained_model, sample_feature_matrix):
    model, feature_cols = trained_model
    X = sample_feature_matrix[feature_cols].fillna(0)
    y = sample_feature_matrix["result_over"].astype(int)

    metrics = model.evaluate(X, y)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0
    assert metrics["brier_score"] > 0


def test_ensemble_feature_importance(trained_model):
    model, _ = trained_model
    imp = model.feature_importance()
    assert "xgb" in imp.columns
    assert "lgb" in imp.columns
    assert "ensemble" in imp.columns
    assert (imp["ensemble"] >= 0).all()


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_save_load(trained_model, tmp_path):
    model, feature_cols = trained_model
    registry = ModelRegistry(registry_path=str(tmp_path))

    version = "test_v1"
    registry.save(
        model=model,
        sport="nba",
        version=version,
        metrics={"accuracy": 0.71},
        feature_names=list(feature_cols),
    )

    loaded = registry.load("nba", version)
    assert loaded is not None
    assert loaded.feature_names == model.feature_names


def test_registry_list_versions(trained_model, tmp_path):
    model, feature_cols = trained_model
    registry = ModelRegistry(registry_path=str(tmp_path))

    for v in ["v1", "v2"]:
        registry.save(model=model, sport="nba", version=v, feature_names=list(feature_cols))

    versions = registry.list_versions("nba")
    assert len(versions) == 2


def test_registry_latest_symlink(trained_model, tmp_path):
    model, feature_cols = trained_model
    registry = ModelRegistry(registry_path=str(tmp_path))

    registry.save(model=model, sport="nba", version="v1", feature_names=list(feature_cols))
    registry.save(model=model, sport="nba", version="v2", feature_names=list(feature_cols))

    loaded = registry.load("nba", "latest")
    assert loaded is not None
