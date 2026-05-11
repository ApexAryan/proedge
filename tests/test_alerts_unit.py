"""Unit tests for AlertManager — evaluate, fire, deque management."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from proedge.monitoring.alerts import Alert, AlertManager


def _alert(**overrides) -> Alert:
    defaults = dict(
        alert_id="test-alert-id",
        sport="NBA",
        home_team="BOS",
        away_team="LAL",
        game_date="2026-05-10",
        direction="over",
        prob_over=0.675,
        confidence=0.35,
        edge=0.175,
        total_line=228.5,
        created_at=datetime.now(timezone.utc),
    )
    return Alert(**{**defaults, **overrides})


def _pred(confidence: float = 0.35, direction: str = "over") -> dict:
    return {
        "sport": "nba",
        "home_team": "BOS",
        "away_team": "LAL",
        "game_date": "2026-05-10",
        "prob_over": 0.675 if direction == "over" else 0.325,
        "prob_under": 0.325 if direction == "over" else 0.675,
        "confidence": confidence,
        "total_line": 228.5,
        "predicted_direction": direction,
    }


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_below_threshold_returns_none(self):
        mgr = AlertManager(confidence_threshold=0.65)
        assert mgr.evaluate(_pred(confidence=0.20)) is None

    def test_exactly_at_threshold_fires(self):
        mgr = AlertManager(confidence_threshold=0.65)
        result = mgr.evaluate(_pred(confidence=0.65))
        assert result is not None

    def test_above_threshold_fires(self):
        mgr = AlertManager(confidence_threshold=0.30)
        result = mgr.evaluate(_pred(confidence=0.80))
        assert isinstance(result, Alert)

    def test_alert_sport_uppercased(self):
        mgr = AlertManager(confidence_threshold=0.0)
        alert = mgr.evaluate(_pred(confidence=0.30))
        assert alert.sport == "NBA"

    def test_alert_fields_match_prediction(self):
        mgr = AlertManager(confidence_threshold=0.0)
        alert = mgr.evaluate(_pred(confidence=0.30, direction="over"))
        assert alert.home_team == "BOS"
        assert alert.away_team == "LAL"
        assert alert.direction == "over"
        assert alert.total_line == 228.5
        assert alert.confidence == pytest.approx(0.30, abs=1e-4)

    def test_edge_is_abs_prob_minus_half(self):
        mgr = AlertManager(confidence_threshold=0.0)
        alert = mgr.evaluate(_pred(confidence=0.30))
        assert alert.edge == pytest.approx(abs(0.675 - 0.5), abs=1e-4)

    def test_under_direction(self):
        mgr = AlertManager(confidence_threshold=0.0)
        alert = mgr.evaluate(_pred(confidence=0.30, direction="under"))
        assert alert.direction == "under"
        assert alert.prob_over == pytest.approx(0.325, abs=1e-4)

    def test_alert_added_to_deque(self):
        mgr = AlertManager(confidence_threshold=0.0)
        mgr.evaluate(_pred(confidence=0.30))
        assert len(mgr.recent(10)) == 1

    def test_multiple_alerts_accumulate(self):
        mgr = AlertManager(confidence_threshold=0.0)
        for _ in range(5):
            mgr.evaluate(_pred(confidence=0.30))
        assert len(mgr.recent(10)) == 5

    def test_deque_evicts_at_maxlen(self):
        mgr = AlertManager(confidence_threshold=0.0)
        for _ in range(503):
            mgr.evaluate(_pred(confidence=0.30))
        assert len(mgr.recent(1000)) == 500


# ---------------------------------------------------------------------------
# recent() and pending()
# ---------------------------------------------------------------------------

class TestRecentAndPending:
    def test_recent_respects_limit(self):
        mgr = AlertManager(confidence_threshold=0.0)
        for _ in range(10):
            mgr.evaluate(_pred(confidence=0.30))
        assert len(mgr.recent(3)) == 3

    def test_recent_newest_first(self):
        mgr = AlertManager(confidence_threshold=0.0)
        for _ in range(3):
            mgr.evaluate(_pred(confidence=0.30))
        alerts = mgr.recent(3)
        times = [a.created_at for a in alerts]
        # Either all the same second, or strictly descending
        assert times == sorted(times, reverse=True) or len(set(times)) == 1

    def test_recent_empty_when_all_below_threshold(self):
        mgr = AlertManager(confidence_threshold=0.99)
        mgr.evaluate(_pred(confidence=0.10))
        assert mgr.recent(10) == []

    def test_pending_returns_unfired(self):
        mgr = AlertManager(confidence_threshold=0.0, webhook_url=None)
        mgr.evaluate(_pred(confidence=0.30))
        pending = mgr.pending()
        assert len(pending) == 1
        assert pending[0].fired is False

    def test_pending_excludes_fired(self):
        mgr = AlertManager(confidence_threshold=0.0, webhook_url=None)
        mgr.evaluate(_pred(confidence=0.30))
        mgr.recent(1)[0].fired = True
        assert mgr.pending() == []


# ---------------------------------------------------------------------------
# fire()
# ---------------------------------------------------------------------------

class TestFire:
    def test_no_webhook_returns_false_not_fired(self):
        mgr = AlertManager(webhook_url=None)
        a = _alert()
        assert mgr.fire(a) is False
        assert a.fired is False

    def test_webhook_success_returns_true_and_marks_fired(self):
        mgr = AlertManager(webhook_url="https://hooks.example.com/test")
        a = _alert()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        with patch("httpx.post", return_value=mock_resp):
            result = mgr.fire(a)
        assert result is True
        assert a.fired is True
        assert a.webhook_response == "200"

    def test_webhook_4xx_returns_false(self):
        mgr = AlertManager(webhook_url="https://hooks.example.com/test")
        a = _alert()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.is_success = False
        with patch("httpx.post", return_value=mock_resp):
            result = mgr.fire(a)
        assert result is False
        assert a.webhook_response == "403"

    def test_webhook_5xx_returns_false(self):
        mgr = AlertManager(webhook_url="https://hooks.example.com/test")
        a = _alert()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.is_success = False
        with patch("httpx.post", return_value=mock_resp):
            result = mgr.fire(a)
        assert result is False

    def test_network_error_returns_false_records_message(self):
        mgr = AlertManager(webhook_url="https://hooks.example.com/test")
        a = _alert()
        with patch("httpx.post", side_effect=httpx.ConnectError("timeout")):
            result = mgr.fire(a)
        assert result is False
        assert "error:" in a.webhook_response
