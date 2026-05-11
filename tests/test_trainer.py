"""Unit tests for proedge.pipeline.training.trainer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from proedge.pipeline.ingestion.stats import STAT_KEYS
from proedge.pipeline.training.trainer import (
    MIN_TRAIN_GAMES,
    _persist_model_run,
    check_and_retrain,
    train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game_df(n: int = 600, sport: str = "nba") -> pd.DataFrame:
    """Synthetic game DataFrame large enough to satisfy MIN_TRAIN_GAMES."""
    rng = np.random.default_rng(0)
    teams = ["BOS", "LAL", "GSW", "MIA", "CHI", "PHX", "DEN", "MIL"]
    stat_cols = STAT_KEYS.get(sport, [])

    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 2) % len(teams)]
        total = float(rng.normal(224, 18))
        line = total + rng.normal(0, 2)
        rows.append({
            "game_id": f"test_{i:05d}",
            "sport": sport,
            "season": 2023 + (i // 82),
            "game_date": datetime(2022, 10, 1) + timedelta(days=i),
            "home_team": home,
            "away_team": away,
            "home_score": int(total / 2 + rng.normal(0, 5)),
            "away_score": int(total / 2 + rng.normal(0, 5)),
            "total": total,
            "total_line": round(line, 1),
            "result_over": int(total > line),
            "venue": f"{home}_arena",
            **{f"home_{s}": float(rng.uniform(90, 130)) for s in stat_cols},
            **{f"away_{s}": float(rng.uniform(90, 130)) for s in stat_cols},
        })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ---------------------------------------------------------------------------
# train() — success path
# ---------------------------------------------------------------------------

class TestTrain:
    def _patched_train(self, df: pd.DataFrame, sport: str = "nba"):
        """Run train() with mocked I/O but real feature engineering + model."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = df

        mock_registry = MagicMock()
        mock_registry.save.return_value = "/tmp/proedge_test/model.joblib"

        with (
            patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader),
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer._persist_model_run"),
        ):
            return train(sport)

    def test_returns_dict(self):
        result = self._patched_train(_make_game_df(600))
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = self._patched_train(_make_game_df(600))
        expected = {
            "sport", "version", "model_path", "feature_count",
            "training_games", "holdout_games",
            "accuracy", "auc", "log_loss", "brier_score", "lift_pct",
        }
        missing = expected - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_sport_matches_input(self):
        result = self._patched_train(_make_game_df(600), sport="nba")
        assert result["sport"] == "nba"

    def test_version_format(self):
        result = self._patched_train(_make_game_df(600))
        # version is "%Y%m%d_%H%M%S" — 15 chars
        assert len(result["version"]) == 15
        assert result["version"][8] == "_"

    def test_training_games_reasonable(self):
        df = _make_game_df(600)
        result = self._patched_train(df)
        # training set is ~70% of total after holdout carve-out
        assert result["training_games"] > 300

    def test_holdout_games_reasonable(self):
        df = _make_game_df(600)
        result = self._patched_train(df)
        assert result["holdout_games"] > 50

    def test_feature_count_positive(self):
        result = self._patched_train(_make_game_df(600))
        assert result["feature_count"] > 0

    def test_accuracy_in_range(self):
        result = self._patched_train(_make_game_df(600))
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_auc_in_range(self):
        result = self._patched_train(_make_game_df(600))
        assert 0.0 <= result["auc"] <= 1.0

    def test_log_loss_positive(self):
        result = self._patched_train(_make_game_df(600))
        assert result["log_loss"] > 0

    def test_model_path_forwarded_from_registry(self):
        result = self._patched_train(_make_game_df(600))
        assert result["model_path"] == "/tmp/proedge_test/model.joblib"

    def test_registry_save_called_once(self):
        df = _make_game_df(600)
        mock_loader = MagicMock()
        mock_loader.load.return_value = df
        mock_registry = MagicMock()
        mock_registry.save.return_value = "/tmp/model.joblib"

        with (
            patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader),
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer._persist_model_run"),
        ):
            train("nba")

        mock_registry.save.assert_called_once()
        _, kwargs = mock_registry.save.call_args
        assert kwargs["sport"] == "nba"
        assert "feature_names" in kwargs
        assert "feature_medians" in kwargs

    # ------------------------------------------------------------------
    # Insufficient data
    # ------------------------------------------------------------------

    def test_insufficient_data_raises_value_error(self):
        tiny_df = _make_game_df(n=MIN_TRAIN_GAMES - 1)
        mock_loader = MagicMock()
        mock_loader.load.return_value = tiny_df

        with patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader):
            with pytest.raises(ValueError, match="Not enough data"):
                train("nba")

    def test_exactly_min_games_does_not_raise(self):
        df = _make_game_df(n=MIN_TRAIN_GAMES)
        mock_loader = MagicMock()
        mock_loader.load.return_value = df
        mock_registry = MagicMock()
        mock_registry.save.return_value = "/tmp/model.joblib"

        with (
            patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader),
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer._persist_model_run"),
        ):
            result = train("nba")

        assert "sport" in result


# ---------------------------------------------------------------------------
# _persist_model_run() — DB failure is swallowed
# ---------------------------------------------------------------------------

class TestPersistModelRun:
    def _call(self, session_raises: Exception | None = None):
        mock_session = MagicMock()
        if session_raises:
            mock_session.__enter__ = MagicMock(side_effect=session_raises)
        else:
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

        with patch("proedge.db.session.SyncSessionLocal", return_value=mock_session):
            _persist_model_run(
                sport="nba",
                version="20240101_120000",
                model_path="/tmp/model.joblib",
                metrics={"accuracy": 0.56, "log_loss": 0.68, "brier_score": 0.24},
                feature_count=120,
                xgb_weight=0.5,
            )

    def test_db_error_does_not_raise(self):
        self._call(session_raises=RuntimeError("DB unavailable"))

    def test_operational_error_does_not_raise(self):
        self._call(session_raises=Exception("connection refused"))

    def test_success_path_does_not_raise(self):
        self._call(session_raises=None)


# ---------------------------------------------------------------------------
# check_and_retrain()
# ---------------------------------------------------------------------------

class TestCheckAndRetrain:
    def _dummy_X(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"f{i}" for i in range(5)])

    def test_no_existing_model_calls_train_and_returns_true(self):
        mock_registry = MagicMock()
        mock_registry.load_meta.return_value = {}

        with (
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer.train") as mock_train,
        ):
            result = check_and_retrain("nba", self._dummy_X())

        assert result is True
        mock_train.assert_called_once_with("nba")

    def _mock_feature_store(self, feature_cols: list[str]) -> MagicMock:
        rng = np.random.default_rng(99)
        mock_store = MagicMock()
        mock_store.compute.return_value = pd.DataFrame(
            rng.standard_normal((50, len(feature_cols))), columns=feature_cols
        )
        return mock_store

    def test_drift_detected_calls_train_with_reason_and_returns_true(self):
        feature_cols = [f"f{i}" for i in range(5)]
        mock_registry = MagicMock()
        mock_registry.load_meta.return_value = {"feature_names": feature_cols}
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = {
            "ensemble": {c: 0.2 for c in feature_cols}
        }
        mock_registry.load.return_value = mock_model

        mock_loader = MagicMock()
        mock_loader.load.return_value = _make_game_df(600)

        mock_drift = MagicMock()
        mock_drift.detect.return_value = {
            "retrain_triggered": True,
            "features_drifted": 3,
            "features_checked": 5,
            "feature_details": {},
        }

        with (
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader),
            patch("proedge.pipeline.training.trainer.FeatureStore",
                  return_value=self._mock_feature_store(feature_cols)),
            patch("proedge.pipeline.training.trainer.DriftDetector", return_value=mock_drift),
            patch("proedge.pipeline.training.trainer.train") as mock_train,
        ):
            result = check_and_retrain("nba", self._dummy_X())

        assert result is True
        mock_train.assert_called_once_with("nba", trigger_reason="drift_psi")

    def test_no_drift_returns_false_without_training(self):
        feature_cols = [f"f{i}" for i in range(5)]
        mock_registry = MagicMock()
        mock_registry.load_meta.return_value = {"feature_names": feature_cols}
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = {
            "ensemble": {c: 0.2 for c in feature_cols}
        }
        mock_registry.load.return_value = mock_model

        mock_loader = MagicMock()
        mock_loader.load.return_value = _make_game_df(600)

        mock_drift = MagicMock()
        mock_drift.detect.return_value = {
            "retrain_triggered": False,
            "features_drifted": 0,
            "features_checked": 5,
            "feature_details": {},
        }

        with (
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader),
            patch("proedge.pipeline.training.trainer.FeatureStore",
                  return_value=self._mock_feature_store(feature_cols)),
            patch("proedge.pipeline.training.trainer.DriftDetector", return_value=mock_drift),
            patch("proedge.pipeline.training.trainer.train") as mock_train,
        ):
            result = check_and_retrain("nba", self._dummy_X())

        assert result is False
        mock_train.assert_not_called()

    def test_drift_psi_labels_emitted(self):
        """Verify DRIFT_PSI metric is called for each feature in the report."""
        feature_cols = ["pts", "reb"]
        X_current = pd.DataFrame(
            np.random.default_rng(2).standard_normal((50, 2)), columns=feature_cols
        )
        mock_registry = MagicMock()
        mock_registry.load_meta.return_value = {"feature_names": feature_cols}
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = {
            "ensemble": {"pts": 0.6, "reb": 0.4}
        }
        mock_registry.load.return_value = mock_model

        mock_loader = MagicMock()
        mock_loader.load.return_value = _make_game_df(600)

        mock_drift = MagicMock()
        mock_drift.detect.return_value = {
            "retrain_triggered": False,
            "features_drifted": 0,
            "features_checked": 2,
            "feature_details": {
                "pts": {"psi": 0.05},
                "reb": {"psi": 0.03},
            },
        }

        psi_calls = []
        mock_psi_gauge = MagicMock()
        mock_psi_gauge.labels.side_effect = lambda **kw: (
            psi_calls.append(kw) or MagicMock()
        )

        with (
            patch("proedge.pipeline.training.trainer.ModelRegistry", return_value=mock_registry),
            patch("proedge.pipeline.training.trainer.HistoricalLoader", return_value=mock_loader),
            patch("proedge.pipeline.training.trainer.FeatureStore",
                  return_value=self._mock_feature_store(feature_cols)),
            patch("proedge.pipeline.training.trainer.DriftDetector", return_value=mock_drift),
            patch("proedge.pipeline.training.trainer.DRIFT_PSI", mock_psi_gauge),
            patch("proedge.pipeline.training.trainer.train"),
        ):
            check_and_retrain("nba", X_current)

        features_reported = {c["feature"] for c in psi_calls}
        assert features_reported == {"pts", "reb"}
