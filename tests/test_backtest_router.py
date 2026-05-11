"""Tests for POST /backtest/{sport} endpoint."""
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from proedge.api.main import app
from proedge.pipeline.backtesting.backtester import BacktestResult, FoldResult

client = TestClient(app)


def _fold(high_conf_accuracy=float("nan")) -> FoldResult:
    return FoldResult(
        fold=1,
        start_date="2023-04-01",
        end_date="2023-08-01",
        n_games=50,
        accuracy=0.54,
        auc=0.57,
        log_loss=0.68,
        brier_score=0.24,
        roi_flat=-0.02,
        roi_kelly=0.01,
        edge_mean=0.08,
        high_conf_accuracy=high_conf_accuracy,
    )


def _result(sport: str = "nba", **overrides) -> BacktestResult:
    defaults = dict(
        sport=sport,
        n_folds=3,
        min_confidence=0.0,
        total_games=100,
        total_bets=90,
        overall_accuracy=0.54,
        overall_auc=0.57,
        overall_roi_flat=-0.02,
        overall_roi_kelly=0.01,
        sharpe_ratio=0.12,
        max_drawdown=0.15,
        folds=[_fold()],
        calibration={"bin_midpoint": [0.5], "actual_freq": [0.54], "predicted_prob": [0.52]},
    )
    return BacktestResult(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Sport validation
# ---------------------------------------------------------------------------

def test_backtest_invalid_sport():
    resp = client.post("/backtest/hockey")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Query-parameter validation
# ---------------------------------------------------------------------------

def test_backtest_n_folds_too_low():
    resp = client.post("/backtest/nba?n_folds=1")
    assert resp.status_code == 422


def test_backtest_n_folds_too_high():
    resp = client.post("/backtest/nba?n_folds=11")
    assert resp.status_code == 422


def test_backtest_min_confidence_out_of_range():
    resp = client.post("/backtest/nba?min_confidence=1.5")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

def test_backtest_no_data_returns_404():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.side_effect = FileNotFoundError("no parquet")
        resp = client.post("/backtest/nba")
    assert resp.status_code == 404


def test_backtest_internal_error_returns_500():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.side_effect = RuntimeError("unexpected")
        resp = client.post("/backtest/nba")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Success path — response structure
# ---------------------------------------------------------------------------

def test_backtest_returns_200_and_top_level_fields():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.return_value = _result()
        resp = client.post("/backtest/nba")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("sport", "n_folds", "min_confidence", "total_games", "total_bets",
                "overall_accuracy", "overall_auc", "overall_roi_flat", "overall_roi_kelly",
                "sharpe_ratio", "max_drawdown", "folds", "calibration", "message"):
        assert key in data, f"Missing top-level key: {key}"


def test_backtest_fold_fields():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.return_value = _result()
        resp = client.post("/backtest/nba")
    folds = resp.json()["folds"]
    assert len(folds) == 1
    for key in ("fold", "start_date", "end_date", "n_games", "accuracy", "auc",
                "log_loss", "brier_score", "roi_flat", "roi_kelly", "edge_mean"):
        assert key in folds[0], f"Missing fold key: {key}"


def test_backtest_high_conf_accuracy_nan_serialises_as_null():
    """float('nan') from the backtester must become JSON null."""
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.return_value = _result(folds=[_fold(float("nan"))])
        resp = client.post("/backtest/nba")
    assert resp.json()["folds"][0]["high_conf_accuracy"] is None


def test_backtest_high_conf_accuracy_real_value_preserved():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.return_value = _result(folds=[_fold(0.61)])
        resp = client.post("/backtest/nba")
    assert resp.json()["folds"][0]["high_conf_accuracy"] == pytest.approx(0.61)


def test_backtest_query_params_forwarded_to_backtester():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.return_value = _result()
        client.post("/backtest/nba?n_folds=4&min_train_games=200&min_confidence=0.3")
    MockBT.return_value.run.assert_called_once_with(
        n_folds=4, min_train_games=200, min_confidence=0.3
    )


def test_backtest_message_contains_key_stats():
    with patch("proedge.pipeline.backtesting.backtester.Backtester") as MockBT:
        MockBT.return_value.run.return_value = _result()
        resp = client.post("/backtest/nba")
    msg = resp.json()["message"]
    assert "Acc=" in msg
    assert "AUC=" in msg
    assert "ROI_flat=" in msg
