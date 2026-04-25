"""Injury report ingestion and impact scoring."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from proedge.config import get_settings
from proedge.pipeline.ingestion.client import SportsDataClient

logger = logging.getLogger(__name__)
settings = get_settings()

STATUS_IMPACT: dict[str, float] = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.40,
    "probable": 0.10,
    "active": 0.0,
}

# Approximate WAR/VORP proxy per position across sports
POSITION_IMPACT: dict[str, dict[str, float]] = {
    "nfl": {
        "QB": 0.90, "WR": 0.45, "RB": 0.40, "TE": 0.35,
        "OL": 0.30, "CB": 0.35, "DE": 0.40, "LB": 0.30, "S": 0.25,
    },
    "nba": {
        "PG": 0.75, "SG": 0.65, "SF": 0.70, "PF": 0.65, "C": 0.60,
    },
    "mlb": {
        "SP": 0.80, "RP": 0.30, "C": 0.50, "1B": 0.45, "2B": 0.50,
        "3B": 0.55, "SS": 0.60, "OF": 0.45,
    },
}


class InjuryIngester:
    def __init__(self):
        self.base_url = settings.espn_api_base_url
        self.api_key = settings.sportradar_api_key

    async def fetch_injury_report(self, sport: str) -> list[dict[str, Any]]:
        path = f"/football/nfl/injuries" if sport == "nfl" else f"/{sport}/injuries"
        async with SportsDataClient(self.base_url, self.api_key) as client:
            try:
                data = await client.get(path)
                return self._parse_injuries(sport, data)
            except Exception as e:
                logger.warning("Injury fetch failed for %s: %s", sport, e)
                return []

    def _parse_injuries(self, sport: str, data: dict) -> list[dict[str, Any]]:
        reports = []
        items = data.get("injuries", data.get("items", []))
        for item in items:
            athlete = item.get("athlete", {})
            position = athlete.get("position", {}).get("abbreviation", "")
            status = item.get("status", "active").lower()
            impact = STATUS_IMPACT.get(status, 0.0) * POSITION_IMPACT.get(sport, {}).get(
                position, 0.3
            )
            reports.append({
                "player_id": str(athlete.get("id", "")),
                "player_name": athlete.get("displayName", ""),
                "team_id": str(item.get("team", {}).get("id", "")),
                "sport": sport,
                "status": status,
                "injury_type": item.get("type", {}).get("description", ""),
                "impact_score": round(impact, 3),
                "reported_at": datetime.utcnow(),
            })
        return reports

    def compute_team_injury_impact(
        self, team_id: str, injury_reports: list[dict]
    ) -> float:
        """Sum of impact scores for all injured players on a team, capped at 1."""
        total = sum(
            r["impact_score"]
            for r in injury_reports
            if r["team_id"] == team_id and r["status"] in STATUS_IMPACT
        )
        return min(total, 1.0)
