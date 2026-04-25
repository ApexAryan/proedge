"""Historical game data loader — 5+ years of results for model training."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proedge.config import get_settings
from proedge.pipeline.ingestion.stats import StatsIngester, STAT_KEYS

logger = logging.getLogger(__name__)
settings = get_settings()

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
    "TEN", "WAS",
]
NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]
MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
]

TEAMS: dict[str, list[str]] = {"nfl": NFL_TEAMS, "nba": NBA_TEAMS, "mlb": MLB_TEAMS}
SEASONS_BACK = 5


class HistoricalLoader:
    """
    Loads historical game data.  In production this fetches from the DB or
    a data lake; here we build a realistic synthetic dataset for development
    and model training when no real data is available.
    """

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ingester = StatsIngester()

    def load(self, sport: str, seasons: int = SEASONS_BACK) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{sport}_historical.parquet"
        if cache_path.exists():
            logger.info("Loading %s historical data from cache", sport)
            return pd.read_parquet(cache_path)

        logger.info("Generating synthetic historical data for %s (%d seasons)", sport, seasons)
        df = self._build_synthetic_dataset(sport, seasons)
        df.to_parquet(cache_path, index=False)
        return df

    def _build_synthetic_dataset(self, sport: str, seasons: int) -> pd.DataFrame:
        current_year = datetime.now().year
        all_rows: list[dict] = []

        teams = TEAMS[sport]
        rng = np.random.default_rng(42)

        # Assign a latent "strength" per team that persists across seasons
        team_strength = {t: float(rng.normal(0, 1)) for t in teams}

        season_start_month = {"nfl": 9, "nba": 10, "mlb": 4}[sport]
        games_per_season = {"nfl": 17, "nba": 82, "mlb": 162}[sport]

        for season_offset in range(seasons):
            season = current_year - seasons + season_offset
            season_start = datetime(season, season_start_month, 1)

            matchups = self._generate_schedule(teams, games_per_season, rng)
            for i, (home, away) in enumerate(matchups):
                game_date = season_start + timedelta(days=int(i * (180 / games_per_season)))
                h_strength = team_strength[home] + rng.normal(0, 0.3)
                a_strength = team_strength[away] + rng.normal(0, 0.3)

                home_score, away_score = self._simulate_scores(
                    sport, h_strength, a_strength, rng
                )
                total = home_score + away_score
                # Realistic line = true total ± small noise
                line = total + rng.normal(0, 2)
                result_over = total > line

                row: dict[str, Any] = {
                    "game_id": f"{sport}_{season}_{i:04d}",
                    "sport": sport,
                    "season": season,
                    "game_date": game_date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "total": total,
                    "total_line": round(line, 1),
                    "result_over": int(result_over),
                    "venue": f"{home}_arena",
                }

                # Generate per-team stat columns
                for side, team, strength in [("home", home, h_strength), ("away", away, a_strength)]:
                    for stat in STAT_KEYS.get(sport, []):
                        base, std = _STAT_DISTRIBUTIONS[sport].get(stat, (50, 10))
                        row[f"{side}_{stat}"] = max(
                            0.0, float(rng.normal(base + strength * std * 0.3, std))
                        )

                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df.sort_values("game_date").reset_index(drop=True)

    def _generate_schedule(
        self, teams: list[str], n_games: int, rng: np.random.Generator
    ) -> list[tuple[str, str]]:
        matchups: list[tuple[str, str]] = []
        game_counts = {t: 0 for t in teams}
        target = n_games
        while min(game_counts.values()) < target:
            home, away = rng.choice(teams, 2, replace=False)
            matchups.append((home, away))
            game_counts[home] += 1
            game_counts[away] += 1
        return matchups[:n_games * len(teams) // 2]

    def _simulate_scores(
        self, sport: str, h_strength: float, a_strength: float, rng: np.random.Generator
    ) -> tuple[int, int]:
        if sport == "nfl":
            home_base, away_base, std = 23, 21, 10
        elif sport == "nba":
            home_base, away_base, std = 113, 110, 12
        else:
            home_base, away_base, std = 4.5, 4.2, 2.5

        home_score = max(0, int(rng.normal(home_base + h_strength * std * 0.2, std)))
        away_score = max(0, int(rng.normal(away_base + a_strength * std * 0.2, std)))
        return home_score, away_score


_STAT_DISTRIBUTIONS: dict[str, dict[str, tuple[float, float]]] = {
    "nfl": {
        "passingYards": (240, 60), "rushingYards": (120, 40),
        "receivingYards": (240, 60), "pointsScored": (23, 10),
        "pointsAllowed": (23, 10), "turnovers": (1.5, 1.2),
        "sacks": (2.5, 1.8), "thirdDownConversion": (0.42, 0.08),
        "redZoneEfficiency": (0.58, 0.12), "timeOfPossession": (30, 4),
    },
    "nba": {
        "points": (113, 12), "rebounds": (44, 5),
        "assists": (25, 5), "steals": (7.5, 2),
        "blocks": (5, 2), "turnovers": (14, 3),
        "fieldGoalPct": (0.47, 0.04), "threePointPct": (0.36, 0.05),
        "freeThrowPct": (0.78, 0.06), "offensiveRebounds": (10, 3),
        "defensiveRating": (112, 6), "offensiveRating": (112, 6),
        "netRating": (0, 6), "pace": (100, 3),
    },
    "mlb": {
        "runsScored": (4.5, 2.5), "runsAllowed": (4.5, 2.5),
        "hits": (9, 3), "errors": (0.6, 0.8),
        "walks": (3.5, 1.8), "strikeouts": (9, 3),
        "era": (4.2, 1.5), "whip": (1.30, 0.25),
        "battingAvg": (0.255, 0.025), "onBasePct": (0.325, 0.030),
        "sluggingPct": (0.420, 0.045), "ops": (0.745, 0.070),
        "homeRuns": (1.2, 1.0),
    },
}
