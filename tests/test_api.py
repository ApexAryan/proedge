"""FastAPI endpoint tests — uses httpx AsyncClient (no real DB required)."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from proedge.api.main import app

client = TestClient(app)


def test_health_endpoint():
    with patch("proedge.api.routers.health.get_db") as mock_db:
        session = AsyncMock()
        session.execute = AsyncMock(return_value=None)
        mock_db.return_value.__aiter__ = AsyncMock(return_value=iter([session]))

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data


def test_readiness_endpoint():
    from fastapi import HTTPException
    from proedge.db.session import get_db as real_get_db

    async def mock_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[real_get_db] = mock_db
    try:
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)
    finally:
        app.dependency_overrides.clear()


def test_predict_no_model_returns_503():
    """When no model is trained for the sport, API should return 503."""
    payload = {
        "sport": "nfl",
        "home_team": "KC",
        "away_team": "SF",
        "game_date": (datetime.utcnow() + timedelta(days=3)).isoformat(),
        "total_line": 47.5,
    }
    with patch("proedge.api.routers.predictions._get_model", return_value=None):
        resp = client.post("/predictions", json=payload)
        assert resp.status_code == 503


def test_predict_returns_prediction_structure(trained_model, sample_feature_matrix):
    """With a mocked model and DB, prediction endpoint should return correct schema."""
    model, feature_cols = trained_model

    mock_game = MagicMock()
    mock_game.id = uuid4()

    mock_pred = MagicMock()
    mock_pred.id = uuid4()
    mock_pred.prob_over = 0.62
    mock_pred.prob_under = 0.38
    mock_pred.ci_lower = 0.55
    mock_pred.ci_upper = 0.70
    mock_pred.predicted_direction = "over"
    mock_pred.confidence = 0.24
    mock_pred.latency_ms = 45.0

    payload = {
        "sport": "nba",
        "home_team": "BOS",
        "away_team": "LAL",
        "game_date": (datetime.utcnow() + timedelta(days=1)).isoformat(),
        "total_line": 228.5,
        "home_rest_days": 2,
        "away_rest_days": 1,
    }

    meta = {"version": "test_v1", "feature_names": list(feature_cols)}

    with (
        patch("proedge.api.routers.predictions._get_model", return_value=model),
        patch("proedge.api.routers.predictions._registry") as mock_registry,
        patch("proedge.api.routers.predictions.GameRepository") as MockGameRepo,
        patch("proedge.api.routers.predictions.PredictionRepository") as MockPredRepo,
    ):
        mock_registry.load_meta.return_value = meta

        game_repo_instance = AsyncMock()
        game_repo_instance.create = AsyncMock(return_value=mock_game)
        MockGameRepo.return_value = game_repo_instance

        pred_repo_instance = AsyncMock()
        pred_repo_instance.create = AsyncMock(return_value=mock_pred)
        MockPredRepo.return_value = pred_repo_instance

        resp = client.post("/predictions", json=payload)

    if resp.status_code != 201:
        print("Response:", resp.json())
    assert resp.status_code == 201
    data = resp.json()
    assert "prob_over" in data
    assert "prob_under" in data
    assert "ci_lower" in data
    assert "ci_upper" in data
    assert "predicted_direction" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert data["sport"] == "nba"


def test_get_predictions_invalid_uuid():
    resp = client.get("/predictions/not-a-uuid")
    assert resp.status_code == 400


def test_get_predictions_not_found():
    fake_id = str(uuid4())
    with patch("proedge.api.routers.predictions.GameRepository") as MockRepo:
        instance = AsyncMock()
        instance.get_by_id = AsyncMock(return_value=None)
        MockRepo.return_value = instance

        resp = client.get(f"/predictions/{fake_id}")
        assert resp.status_code == 404


def test_model_performance_endpoint():
    with patch("proedge.api.routers.performance._registry") as mock_registry:
        mock_registry.list_versions.return_value = []
        resp = client.get("/models/performance")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_feature_importance_no_model():
    with patch("proedge.api.routers.performance._registry") as mock_registry:
        mock_registry.load.side_effect = FileNotFoundError
        resp = client.get("/models/feature-importance/nba")
        assert resp.status_code == 404


def test_invalid_sport_returns_400():
    resp = client.get("/models/performance/hockey")
    assert resp.status_code == 400
