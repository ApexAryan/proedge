"""Live player and team stats ingestion from ESPN / SportRadar APIs."""
from __future__ import annotations

import logging
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

# Core stat keys per sport — drives both feature rolling and synthetic generation
STAT_KEYS: dict[str, list[str]] = {
    "nfl": [
        # Basic
        "passingYards", "rushingYards", "receivingYards", "pointsScored",
        "pointsAllowed", "turnovers", "sacks", "thirdDownConversion",
        "redZoneEfficiency", "timeOfPossession",
        # Advanced offense
        "yardsPerPlay", "redZoneConvRate",
        "fourthDownConversion", "penaltyYards",
        # Defensive / tempo
        "pressureRate", "secondsPerPlay",
    ],
    "nba": [
        # Basic box score
        "points", "rebounds", "assists", "steals", "blocks",
        "turnovers", "personalFouls",
        # Shooting volume
        "fieldGoalsMade", "fieldGoalAttempts",
        "threesMade", "threePointAttempts",
        "freeThrowsMade", "freeThrowAttempts",
        # Shooting efficiency
        "fieldGoalPct", "threePointPct", "freeThrowPct",
        "trueShooting", "ftRate", "threePointRate",
        # Rebounding
        "offensiveRebounds", "defensiveRebounds", "drebRate",
        # Pace / possession
        "possessions", "pointsPerPossession", "pace", "assistRate",
        # Ratings
        "offensiveRating", "defensiveRating", "netRating",
    ],
    "mlb": [
        # Basic
        "runsScored", "runsAllowed", "hits", "errors", "walks",
        "strikeouts", "era", "whip", "battingAvg", "onBasePct",
        "sluggingPct", "ops", "homeRuns",
        # Advanced — only fields available from MLB Stats API boxscore
        "kBbRatio", "groundBallRate", "flyBallRate",
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
        return data

    def _generate_mock_game_log(self, sport: str, team_id: str, season: int) -> pd.DataFrame:
        import numpy as np
        rng = np.random.default_rng(hash(team_id) % (2**31))

        n_games = 82 if sport == "nba" else (162 if sport == "mlb" else 17)
        dates = pd.date_range(
            start=f"{season}-10-01" if sport == "nba" else f"{season}-04-01",
            periods=n_games, freq="3D",
        )

        stat_ranges: dict[str, tuple[float, float]] = _MOCK_RANGES.get(sport, {})
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


_MOCK_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "nfl": {k: v for k, v in zip(STAT_KEYS["nfl"], [
        (150, 400), (80, 200), (40, 150), (14, 38), (14, 38),
        (0, 4), (0, 6), (0.3, 0.55), (0.4, 0.7), (25, 35),
        (4.5, 6.5), (0.45, 0.70),
        (0.35, 0.60), (30, 90),
        (0.18, 0.35), (25, 32),
    ])},
    "nba": {k: v for k, v in zip(STAT_KEYS["nba"], [
        (95, 130), (40, 55), (20, 30), (5, 12), (4, 8),
        (10, 18), (16, 24),
        (38, 48), (80, 92),
        (10, 16), (28, 40),
        (16, 26), (20, 28),
        (0.43, 0.52), (0.33, 0.42), (0.72, 0.82),
        (0.54, 0.60), (0.22, 0.32), (0.38, 0.46),
        (8, 16), (30, 40), (0.65, 0.78),
        (88, 105), (1.05, 1.22), (88, 105), (0.22, 0.28),
        (98, 118), (98, 118), (-8, 8),
    ])},
    "mlb": {k: v for k, v in zip(STAT_KEYS["mlb"], [
        (2, 9), (2, 9), (5, 14), (0, 3), (2, 6),
        (5, 12), (2.5, 5.5), (1.0, 1.6), (0.220, 0.290),
        (0.300, 0.380), (0.350, 0.480), (0.650, 0.850), (0, 3),
        (2.0, 4.5), (0.40, 0.55), (0.30, 0.45),
    ])},
}


def _safe_float(val: Any) -> float:
    try:
        return float(str(val).replace("%", "").replace(":", "."))
    except (ValueError, TypeError):
        return 0.0
