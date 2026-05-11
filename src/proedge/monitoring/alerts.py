"""Alert system: fires when model finds high-confidence edges."""
from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    alert_id: str           # uuid
    sport: str
    home_team: str
    away_team: str
    game_date: str
    direction: str          # "over" | "under"
    prob_over: float
    confidence: float
    edge: float             # abs(prob_over - 0.5)
    total_line: float
    created_at: datetime
    fired: bool = False
    webhook_response: str | None = None


class AlertManager:
    """
    Evaluates model predictions against a confidence threshold and fires
    webhook alerts when an edge is detected.

    Keeps the last 500 alerts in memory as a write-through cache.
    The predictions router persists each alert to the ``alert_records`` DB
    table so alerts survive process restarts.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        webhook_url: str | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.webhook_url = webhook_url
        self._alerts: deque[Alert] = deque(maxlen=500)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, prediction: dict) -> Alert | None:
        """Check if a prediction crosses the confidence threshold.

        ``prediction`` must contain the keys:
            sport, home_team, away_team, game_date, prob_over, prob_under,
            confidence, total_line, predicted_direction

        Creates and fires an Alert if confidence >= threshold; returns None
        otherwise.
        """
        confidence: float = float(prediction.get("confidence", 0.0))
        if confidence < self.confidence_threshold:
            return None

        prob_over: float = float(prediction.get("prob_over", 0.5))
        direction: str = str(prediction.get("predicted_direction", "over")).lower()
        edge: float = abs(prob_over - 0.5)
        total_line: float = float(prediction.get("total_line", 0.0))

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            sport=str(prediction.get("sport", "")).upper(),
            home_team=str(prediction.get("home_team", "")),
            away_team=str(prediction.get("away_team", "")),
            game_date=str(prediction.get("game_date", "")),
            direction=direction,
            prob_over=round(prob_over, 4),
            confidence=round(confidence, 4),
            edge=round(edge, 4),
            total_line=total_line,
            created_at=datetime.now(timezone.utc),
        )

        self._alerts.append(alert)
        self.fire(alert)
        return alert

    def fire(self, alert: Alert) -> bool:
        """POST the alert to webhook_url if configured.

        Returns True if the webhook was reached successfully (2xx), False
        otherwise.  Never raises — logs errors and continues.
        """
        if not self.webhook_url:
            logger.debug(
                "Alert %s created (no webhook configured): %s %s %s",
                alert.alert_id,
                alert.sport,
                alert.direction.upper(),
                alert.total_line,
            )
            return False

        direction_label = alert.direction.upper()
        prob_pct = round(alert.prob_over * 100) if alert.direction == "over" else round((1 - alert.prob_over) * 100)
        matchup = f"{alert.home_team} vs {alert.away_team}"
        pick_label = f"{direction_label} {alert.total_line}"
        conf_pct = round(alert.confidence * 100)

        message = (
            f"High-confidence {alert.sport} {direction_label}: "
            f"{matchup} | Line {alert.total_line} | "
            f"{prob_pct}% {direction_label.lower()} | Conf {conf_pct}%"
        )

        payload = {
            "alert_id": alert.alert_id,
            "sport": alert.sport,
            "matchup": matchup,
            "pick": pick_label,
            "confidence": alert.confidence,
            "prob_over": alert.prob_over,
            "game_date": alert.game_date,
            "message": message,
        }

        try:
            resp = httpx.post(self.webhook_url, json=payload, timeout=5.0)
            alert.fired = True
            alert.webhook_response = f"{resp.status_code}"
            if resp.is_success:
                logger.info(
                    "Alert fired — %s | %s | conf=%.2f | HTTP %d",
                    alert.sport,
                    pick_label,
                    alert.confidence,
                    resp.status_code,
                )
                return True
            else:
                logger.warning(
                    "Alert webhook returned non-2xx %d for alert %s",
                    resp.status_code,
                    alert.alert_id,
                )
                return False
        except httpx.RequestError as exc:
            alert.webhook_response = f"error: {exc}"
            logger.warning(
                "Alert webhook request failed for %s: %s", alert.alert_id, exc
            )
            return False

    def recent(self, n: int = 50) -> list[Alert]:
        """Return up to n most recent alerts (newest first)."""
        alerts = list(self._alerts)
        return list(reversed(alerts))[:n]

    def pending(self) -> list[Alert]:
        """Return alerts that have not been successfully fired (fired=False)."""
        return [a for a in self._alerts if not a.fired]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """Return the process-wide AlertManager singleton.

    Initialised lazily from application settings on first call.
    """
    global _manager
    if _manager is None:
        from proedge.config import get_settings  # local import to avoid circular deps

        s = get_settings()
        _manager = AlertManager(
            confidence_threshold=getattr(s, "alert_confidence_threshold", 0.65),
            webhook_url=getattr(s, "alert_webhook_url", None),
        )
    return _manager
