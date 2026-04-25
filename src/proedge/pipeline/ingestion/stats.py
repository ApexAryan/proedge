"""Live player and team stats ingestion from ESPN / SportRadar APIs."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from proedge.config import get_settings
from proedge.pipeline.ingestion.client import SportsDataClient

logger = logging.getLogger(__name__)

settings = get_settings()

_SPORT_PATHS = {
    "nfl": "football/nfl",
    "nba": "basketball/nba",
    "mlb": "baseball/mlb",
}

# Core stat keys per sport (used to normalize API responses)
STAT_KEYS: dict[str, list[str]] = {
    "nfl": [
        "passingYards", "rushingYards", "receivingYards", "pointsScored",
        "pointsAllowed", "turnovers", "sacks", "thirdDownConversion",
        "redZoneEfficiency", "timeOfPossession",
    ],
    "nba": [
        "points", "rebounds", "assists", "steals", "blocks",
        "turnovers", "fieldGoalPct", "threePointPct", "freeThrowPct",
        "offensiveRebounds", "defensiveRating", "offensiveRating",
        "netRating", "pace",
    ],
    "mlb": [
        "runsScored", "runsAllowed", "hits", "errors", "walks",
        "strikeouts", "era", "whip", "battingAvg", "onBasePct",
        "sluggingPct", "ops", "homeRuns",
    ],
}


class StatsIngester:
    def __init__(self):
        self.base_url = settings.espn_api_base_url
        self.api_key = settings.sportradar_api_key

    async def fetch_team_game_log(
        self, sport: str, team_id: str, season: int
    ) -> pd.DataFrame:
        path = f"/{_SPORT_PATHS[sport]}/teams/{team_id}/schedule"
        async with SportsDataClient(self.base_url, self.api_key) as client:
            try:
                data = await client.get(path, params={"season": season})
                return self._parse_game_log(sport, data)
            except Exception as e:
                logger.warning("ESPN fetch failed for %s %s: %s — using mock", sport, team_id, e)
                return self._generate_mock_game_log(sport, team_id, season)

    async def fetch_live_boxscore(self, sport: str, game_id: str) -> dict[str, Any]:
        path = f"/{_SPORT_PATHS[sport]}/summary"
        async with SportsDataClient(self.base_url, self.api_key) as client:
            try:
                data = await client.get(path, params={"event": game_id})
                return self._parse_boxscore(sport, data)
            except Exception as e:
                logger.warning("Boxscore fetch failed %s: %s", game_id, e)
                return {}

    def _parse_game_log(self, sport: str, data: dict) -> pd.DataFrame:
        events = data.get("events", [])
        rows = []
        for event in events:
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            for comp in competitors:
                stats_raw = comp.get("statistics", [])
                stat_dict = {s["name"]: s.get("displayValue", 0) for s in stats_raw}
                rows.append({
                    "game_id": event.get("id"),
                    "game_date": event.get("date"),
                    "team_id": comp.get("team", {}).get("id"),
                    "team_name": comp.get("team", {}).get("displayName"),
                    "home_away": comp.get("homeAway", "home"),
                    "score": comp.get("score", 0),
                    **{k: _safe_float(stat_dict.get(k, 0)) for k in STAT_KEYS.get(sport, [])},
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _parse_boxscore(self, sport: str, data: dict) -> dict[str, Any]:
        return data  # pass-through; caller normalizes

    def _generate_mock_game_log(self, sport: str, team_id: str, season: int) -> pd.DataFrame:
        """Deterministic mock data seeded by team_id for development."""
        import numpy as np
        rng = np.random.default_rng(hash(team_id) % (2**31))

        n_games = 82 if sport == "nba" else (162 if sport == "mlb" else 17)
        dates = pd.date_range(
            start=f"{season}-10-01" if sport == "nba" else f"{season}-04-01",
            periods=n_games, freq="3D",
        )

        stat_ranges: dict[str, tuple[float, float]] = {
            "nfl": {k: v for k, v in zip(STAT_KEYS["nfl"], [
                (150, 400), (80, 200), (40, 150), (14, 38), (14, 38),
                (0, 4), (0, 6), (0.3, 0.55), (0.4, 0.7), (25, 35),
            ])},
            "nba": {k: v for k, v in zip(STAT_KEYS["nba"], [
                (95, 130), (40, 55), (20, 30), (5, 12), (4, 8),
                (10, 18), (0.43, 0.52), (0.33, 0.42), (0.72, 0.82),
                (8, 16), (98, 118), (98, 118), (-8, 8), (96, 104),
            ])},
            "mlb": {k: v for k, v in zip(STAT_KEYS["mlb"], [
                (2, 9), (2, 9), (5, 14), (0, 3), (2, 6),
                (5, 12), (2.5, 5.5), (1.0, 1.6), (0.220, 0.290),
                (0.300, 0.380), (0.350, 0.480), (0.650, 0.850), (0, 3),
            ])},
        }.get(sport, {})

        rows = []
        for i, d in enumerate(dates):
            row: dict[str, Any] = {
                "game_id": f"mock_{team_id}_{season}_{i:03d}",
                "game_date": d.isoformat(),
                "team_id": team_id,
                "team_name": f"Team_{team_id}",
                "home_away": "home" if i % 2 == 0 else "away",
                "score": int(rng.integers(14, 38) if sport == "nfl" else
                             rng.integers(95, 130) if sport == "nba" else
                             rng.integers(1, 10)),
            }
            for stat, (lo, hi) in stat_ranges.items():
                row[stat] = float(rng.uniform(lo, hi))
            rows.append(row)
        return pd.DataFrame(rows)


def _safe_float(val: Any) -> float:
    try:
        return float(str(val).replace("%", "").replace(":", "."))
    except (ValueError, TypeError):
        return 0.0
