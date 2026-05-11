"""FastAPI endpoint tests — uses httpx AsyncClient (no real DB required)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from proedge.api.main import app

client = TestClient(app)


_VALID_HEALTH_STATUSES = {"ok", "degraded", "no_models"}


def test_health_endpoint():
    with patch("proedge.api.routers.health.get_db") as mock_db:
        session = AsyncMock()
        session.execute = AsyncMock(return_value=None)
        mock_db.return_value.__aiter__ = AsyncMock(return_value=iter([session]))

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in _VALID_HEALTH_STATUSES, (
            f"Unexpected health status: {data['status']!r}"
        )
        assert "version" in data
        assert "db_connected" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data


def test_health_no_models_returns_no_models_status():
    """When no models are trained the health status must be 'no_models', not 'ok'."""
    from proedge.db.session import get_db as real_get_db

    async def mock_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[real_get_db] = mock_db
    try:
        with patch("proedge.pipeline.models.registry.ModelRegistry") as MockReg:
            # Simulate no versions for any sport
            MockReg.return_value.latest_version.return_value = None
            MockReg.return_value.load_meta.return_value = {}
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_models"
        assert all(v is None for v in data["models_loaded"].values())
    finally:
        app.dependency_overrides.clear()


def test_readiness_endpoint():
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
        "game_date": (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
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
        "game_date": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
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


# ---------------------------------------------------------------------------
# GET /predictions/recent
# ---------------------------------------------------------------------------

def test_get_recent_predictions_empty_list():
    with patch("proedge.api.routers.predictions.PredictionRepository") as MockPredRepo:
        instance = AsyncMock()
        instance.get_recent = AsyncMock(return_value=[])
        MockPredRepo.return_value = instance

        resp = client.get("/predictions/recent")
        assert resp.status_code == 200
        assert resp.json() == []


def test_get_recent_predictions_returns_correct_keys():
    mock_pred = MagicMock()
    mock_pred.id = uuid4()
    mock_pred.game_id = uuid4()
    mock_pred.sport = "nba"
    mock_pred.model_version = "v1"
    mock_pred.prob_over = 0.65
    mock_pred.prob_under = 0.35
    mock_pred.predicted_direction = "over"
    mock_pred.confidence = 0.30
    mock_pred.predicted_at = None
    mock_pred.is_correct = True
    mock_pred.clv = 1.5
    mock_pred.actual_total = 230.0
    mock_pred.closing_line = 228.5
    mock_pred.settled_at = None

    with patch("proedge.api.routers.predictions.PredictionRepository") as MockPredRepo:
        instance = AsyncMock()
        instance.get_recent = AsyncMock(return_value=[mock_pred])
        MockPredRepo.return_value = instance

        resp = client.get("/predictions/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        item = data[0]
        for key in (
            "prediction_id", "sport", "prob_over", "prob_under",
            "predicted_direction", "confidence", "predicted_at",
            "is_correct", "clv", "actual_total", "closing_line",
            "settled_at", "model_version", "game_id",
        ):
            assert key in item, f"Missing key: {key}"


def test_get_recent_predictions_sport_filter():
    with patch("proedge.api.routers.predictions.PredictionRepository") as MockPredRepo:
        instance = AsyncMock()
        instance.get_recent = AsyncMock(return_value=[])
        MockPredRepo.return_value = instance

        resp = client.get("/predictions/recent?sport=nba&limit=10")
        assert resp.status_code == 200
        instance.get_recent.assert_called_once_with(sport="nba", limit=10)


# ---------------------------------------------------------------------------
# GET /predictions/alerts/recent
# ---------------------------------------------------------------------------

def test_get_recent_alerts_returns_correct_structure():
    mock_alert = MagicMock()
    mock_alert.alert_id = "alert-001"
    mock_alert.sport = "nba"
    mock_alert.home_team = "BOS"
    mock_alert.away_team = "LAL"
    mock_alert.game_date = "2026-04-25"
    mock_alert.direction = "over"
    mock_alert.confidence = 0.28
    mock_alert.prob_over = 0.64
    mock_alert.edge = 0.14
    mock_alert.total_line = 228.5
    mock_alert.fired = True
    mock_alert.created_at = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)

    mock_mgr = MagicMock()
    mock_mgr.recent.return_value = [mock_alert]

    with patch("proedge.api.routers.predictions.get_alert_manager", return_value=mock_mgr):
        resp = client.get("/predictions/alerts/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        item = data[0]
        for key in (
            "alert_id", "sport", "home_team", "away_team", "game_date",
            "direction", "confidence", "prob_over", "edge", "total_line",
            "fired", "created_at",
        ):
            assert key in item, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# POST /predictions/{prediction_id}/settle
# ---------------------------------------------------------------------------

def test_settle_prediction_invalid_uuid():
    resp = client.post(
        "/predictions/not-a-uuid/settle",
        json={"actual_total": 231.0, "closing_line": 228.5},
    )
    assert resp.status_code == 400


def test_settle_prediction_not_found():
    fake_id = str(uuid4())

    with patch("proedge.api.routers.predictions.PredictionRepository") as MockPredRepo:
        instance = AsyncMock()
        instance.get_by_id = AsyncMock(return_value=None)
        MockPredRepo.return_value = instance

        resp = client.post(
            f"/predictions/{fake_id}/settle",
            json={"actual_total": 231.0, "closing_line": 228.5},
        )
        assert resp.status_code == 404


def test_settle_prediction_valid():
    pred_id = uuid4()
    game_id = uuid4()

    mock_pred = MagicMock()
    mock_pred.id = pred_id
    mock_pred.game_id = game_id
    mock_pred.predicted_direction = "over"

    mock_game = MagicMock()
    mock_game.id = game_id
    mock_game.total_line = 225.0

    with (
        patch("proedge.api.routers.predictions.PredictionRepository") as MockPredRepo,
        patch("proedge.api.routers.predictions.GameRepository") as MockGameRepo,
    ):
        pred_instance = AsyncMock()
        pred_instance.get_by_id = AsyncMock(return_value=mock_pred)
        pred_instance.settle = AsyncMock()
        MockPredRepo.return_value = pred_instance

        game_instance = AsyncMock()
        game_instance.get_by_id = AsyncMock(return_value=mock_game)
        MockGameRepo.return_value = game_instance

        resp = client.post(
            f"/predictions/{pred_id}/settle",
            json={"actual_total": 231.0, "closing_line": 228.5},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["clv"] == 3.5
    assert data["is_correct"] is True
    assert data["predicted_direction"] == "over"
    assert "prediction_id" in data
    assert "actual_total" in data
    assert "closing_line" in data
    assert "message" in data


# ---------------------------------------------------------------------------
# GET /models/accuracy/live
# ---------------------------------------------------------------------------

def test_live_accuracy_returns_dict_with_correct_structure():
    with (
        patch("proedge.api.routers.performance._registry") as mock_registry,
        patch("proedge.api.routers.performance.PredictionRepository") as MockPredRepo,
    ):
        mock_registry.load_meta.return_value = {"version": "v1"}

        pred_instance = AsyncMock()
        pred_instance.accuracy_by_version = AsyncMock(
            return_value={"accuracy": 0.72, "total": 50}
        )
        MockPredRepo.return_value = pred_instance

        resp = client.get("/models/accuracy/live")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        for sport_data in data.values():
            assert "accuracy" in sport_data
            assert "total" in sport_data


def test_live_accuracy_file_not_found_returns_null_accuracy():
    with (
        patch("proedge.api.routers.performance._registry") as mock_registry,
        patch("proedge.api.routers.performance.PredictionRepository") as MockPredRepo,
    ):
        mock_registry.load_meta.side_effect = FileNotFoundError

        pred_instance = AsyncMock()
        MockPredRepo.return_value = pred_instance

        resp = client.get("/models/accuracy/live")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        for sport_data in data.values():
            assert sport_data["accuracy"] is None
            assert sport_data["total"] == 0


# ---------------------------------------------------------------------------
# POST /models/drift-check/{sport}
# ---------------------------------------------------------------------------

def test_drift_check_invalid_sport():
    resp = client.post("/models/drift-check/hockey")
    assert resp.status_code == 400


def test_drift_check_no_model_returns_404():
    with patch("proedge.api.routers.performance._registry") as mock_registry:
        mock_registry.load_meta.side_effect = FileNotFoundError("no model")
        resp = client.post("/models/drift-check/nba")
        assert resp.status_code == 404


def test_drift_check_returns_report():
    fake_report = {
        "retrain_triggered": False,
        "features_checked": 5,
        "features_drifted": 0,
        "feature_details": {},
    }
    with patch("proedge.api.routers.performance._run_drift", return_value=fake_report):
        resp = client.post("/models/drift-check/nba")
    assert resp.status_code == 200
    data = resp.json()
    assert "retrain_triggered" in data
    assert "features_checked" in data


# ---------------------------------------------------------------------------
# GET /training/status/{sport} and GET /training/status
# ---------------------------------------------------------------------------

def test_training_status_invalid_sport():
    resp = client.get("/training/status/hockey")
    assert resp.status_code == 422


def test_training_status_valid_sport():
    with patch("proedge.pipeline.models.registry.ModelRegistry") as MockReg:
        MockReg.return_value.load_meta.return_value = {"version": "v1", "trained_at": "2026-01-01T00:00:00"}
        resp = client.get("/training/status/nba")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sport"] == "nba"
    assert "last_retrain_version" in data
    assert "total_historical_games" in data


def test_training_status_all_sports():
    with patch("proedge.pipeline.models.registry.ModelRegistry") as MockReg:
        MockReg.return_value.load_meta.side_effect = FileNotFoundError
        resp = client.get("/training/status")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    for sport in ("nba", "nfl", "mlb"):
        assert sport in data


def test_training_update_invalid_sport():
    resp = client.post("/training/update/hockey")
    assert resp.status_code == 422


def test_training_retrain_invalid_sport():
    resp = client.post("/training/retrain/hockey")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /lines/prizepicks/{sport}
# ---------------------------------------------------------------------------

def test_lines_invalid_sport():
    resp = client.get("/lines/prizepicks/hockey")
    assert resp.status_code == 422


def test_lines_prizepicks_returns_board():
    from unittest.mock import AsyncMock
    from proedge.pipeline.ingestion.prizepicks_fetcher import PrizePicksBoard

    fake_board = PrizePicksBoard(
        sport="nba",
        fetched_at="2026-04-25T12:00:00",
        player_projections=[],
        game_lines=[],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=fake_board)):
        resp = client.get("/lines/prizepicks/nba")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sport"] == "nba"
    assert "game_count" in data
    assert "games" in data
