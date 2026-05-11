"""Tests for POST /training/update and POST /training/retrain endpoints."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from proedge.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# POST /training/update/{sport}
# ---------------------------------------------------------------------------

def test_training_update_invalid_sport():
    resp = client.post("/training/update/hockey")
    assert resp.status_code == 422


def test_training_update_invalid_date_format():
    resp = client.post("/training/update/nba?date=not-a-date")
    assert resp.status_code == 422


def test_training_update_success():
    from proedge.pipeline.ingestion.daily_updater import UpdateResult

    mock_result = UpdateResult(
        sport="nba",
        date="2026-05-09",
        games_found=3,
        games_added=3,
        games_skipped=0,
        retrain_triggered=False,
        retrain_metrics={},
        error=None,
    )
    with (
        patch("proedge.pipeline.ingestion.daily_updater.DailyUpdater") as MockDU,
        patch("proedge.api.routers.training._save_update_status"),
    ):
        MockDU.return_value.run.return_value = mock_result
        resp = client.post("/training/update/nba?date=2026-05-09")

    assert resp.status_code == 200
    data = resp.json()
    assert data["sport"] == "nba"
    assert data["games_added"] == 3
    assert data["games_skipped"] == 0
    assert data["retrain_triggered"] is False
    assert "message" in data
    assert "Added 3" in data["message"]


def test_training_update_no_new_games_message():
    from proedge.pipeline.ingestion.daily_updater import UpdateResult

    mock_result = UpdateResult(sport="nba", date="2026-05-09", games_found=0)
    with (
        patch("proedge.pipeline.ingestion.daily_updater.DailyUpdater") as MockDU,
        patch("proedge.api.routers.training._save_update_status"),
    ):
        MockDU.return_value.run.return_value = mock_result
        resp = client.post("/training/update/nba")

    assert resp.status_code == 200
    assert resp.json()["message"] == "No new games found."


def test_training_update_with_retrain():
    from proedge.pipeline.ingestion.daily_updater import UpdateResult

    mock_result = UpdateResult(
        sport="nba",
        date="2026-05-09",
        games_found=35,
        games_added=35,
        games_skipped=0,
        retrain_triggered=True,
        retrain_metrics={"accuracy": 0.56},
        error=None,
    )
    with (
        patch("proedge.pipeline.ingestion.daily_updater.DailyUpdater") as MockDU,
        patch("proedge.api.routers.training._save_update_status"),
    ):
        MockDU.return_value.run.return_value = mock_result
        resp = client.post("/training/update/nba?auto_retrain=true")

    assert resp.status_code == 200
    data = resp.json()
    assert data["retrain_triggered"] is True
    assert data["retrain_metrics"] == {"accuracy": 0.56}


def test_training_update_response_fields():
    from proedge.pipeline.ingestion.daily_updater import UpdateResult

    mock_result = UpdateResult(sport="nba", date="2026-05-09")
    with (
        patch("proedge.pipeline.ingestion.daily_updater.DailyUpdater") as MockDU,
        patch("proedge.api.routers.training._save_update_status"),
    ):
        MockDU.return_value.run.return_value = mock_result
        resp = client.post("/training/update/nba")

    for key in ("sport", "date", "games_found", "games_added", "games_skipped",
                "retrain_triggered", "retrain_metrics", "message"):
        assert key in resp.json(), f"Missing key: {key}"


# ---------------------------------------------------------------------------
# POST /training/retrain/{sport}
# ---------------------------------------------------------------------------

def test_training_retrain_invalid_sport():
    resp = client.post("/training/retrain/hockey")
    assert resp.status_code == 422


def test_training_retrain_success():
    mock_metrics = {
        "version": "nba_20260510_001",
        "accuracy": 0.56,
        "auc": 0.59,
        "log_loss": 0.67,
        "brier_score": 0.23,
        "training_games": 5800,
        "feature_count": 210,
    }
    with patch("proedge.pipeline.training.trainer.train", return_value=mock_metrics):
        resp = client.post("/training/retrain/nba")

    assert resp.status_code == 200
    data = resp.json()
    assert data["sport"] == "nba"
    assert data["version"] == "nba_20260510_001"
    assert data["accuracy"] == 0.56
    assert data["training_games"] == 5800
    assert "Retrain complete" in data["message"]


def test_training_retrain_response_fields():
    mock_metrics = {"version": "v1", "accuracy": 0.54}
    with patch("proedge.pipeline.training.trainer.train", return_value=mock_metrics):
        resp = client.post("/training/retrain/nba")

    for key in ("sport", "version", "accuracy", "auc", "log_loss",
                "brier_score", "training_games", "feature_count", "message"):
        assert key in resp.json(), f"Missing key: {key}"


def test_training_retrain_concurrent_returns_409():
    """Second concurrent retrain must receive 409 Conflict."""
    import proedge.api.routers.training as training_module

    mock_lock = MagicMock()
    mock_lock.locked.return_value = True

    with patch.dict(training_module._retrain_locks, {"nba": mock_lock}):
        resp = client.post("/training/retrain/nba")

    assert resp.status_code == 409
    assert "progress" in resp.json()["detail"].lower()


def test_training_retrain_failure_returns_500():
    with patch("proedge.pipeline.training.trainer.train", side_effect=RuntimeError("OOM")):
        resp = client.post("/training/retrain/nba")
    assert resp.status_code == 500
