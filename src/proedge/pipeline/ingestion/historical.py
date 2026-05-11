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
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LV",
    "LAC",
    "LAR",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SF",
    "SEA",
    "TB",
    "TEN",
    "WAS",
]
NBA_TEAMS = [
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]
MLB_TEAMS = [
    "ARI",
    "ATL",
    "BAL",
    "BOS",
    "CHC",
    "CWS",
    "CIN",
    "CLE",
    "COL",
    "DET",
    "HOU",
    "KC",
    "LAA",
    "LAD",
    "MIA",
    "MIL",
    "MIN",
    "NYM",
    "NYY",
    "OAK",
    "PHI",
    "PIT",
    "SD",
    "SF",
    "SEA",
    "STL",
    "TB",
    "TEX",
    "TOR",
    "WSH",
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

        if sport == "nba":
            try:
                from proedge.pipeline.ingestion.nba_fetcher import fetch_nba_games

                logger.info("Fetching real NBA game data from NBA stats API...")
                df = fetch_nba_games()
                df.to_parquet(cache_path, index=False)
                return df
            except Exception as exc:
                self._record_fetch_failure(sport, "nba_api", exc)

        if sport == "nfl":
            try:
                from proedge.pipeline.ingestion.espn_nfl_fetcher import fetch_nfl_games

                logger.info("Fetching real NFL game data from ESPN...")
                df = fetch_nfl_games()
                if not df.empty:
                    df.to_parquet(cache_path, index=False)
                    return df
                self._record_fetch_failure(sport, "espn", ValueError("empty DataFrame returned"))
            except Exception as exc:
                self._record_fetch_failure(sport, "espn", exc)

        if sport == "mlb":
            try:
                from proedge.pipeline.ingestion.mlb_stats_fetcher import fetch_mlb_games

                logger.info("Fetching real MLB game data from MLB Stats API...")
                df = fetch_mlb_games()
                if not df.empty:
                    df.to_parquet(cache_path, index=False)
                    return df
                self._record_fetch_failure(
                    sport, "mlb_statsapi", ValueError("empty DataFrame returned")
                )
            except Exception as exc:
                self._record_fetch_failure(sport, "mlb_statsapi", exc)

        logger.error(
            "SYNTHETIC DATA WARNING: Falling back to generated data for %s. "
            "All predictions trained on this data will have inflated metrics and poor real-world accuracy. "
            "Fix the data fetcher above before using this model in production.",
            sport,
        )
        try:
            from proedge.monitoring.metrics import SYNTHETIC_DATA_TOTAL

            SYNTHETIC_DATA_TOTAL.labels(sport=sport).inc()
        except Exception:
            pass

        df = self._build_synthetic_dataset(sport, seasons)
        df.to_parquet(cache_path, index=False)
        return df

    @staticmethod
    def _record_fetch_failure(sport: str, source: str, exc: Exception) -> None:
        logger.error(
            "Real %s data fetch failed (source=%s): %s — will attempt synthetic fallback",
            sport.upper(),
            source,
            exc,
        )
        try:
            from proedge.monitoring.metrics import DATA_FETCH_ERRORS

            DATA_FETCH_ERRORS.labels(sport=sport, source=source).inc()
        except Exception:
            pass

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

                home_score, away_score = self._simulate_scores(sport, h_strength, a_strength, rng)
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
                for side, team, strength in [
                    ("home", home, h_strength),
                    ("away", away, a_strength),
                ]:
                    for stat in STAT_KEYS.get(sport, []):
                        base, std = _STAT_DISTRIBUTIONS[sport].get(stat, (50, 10))
                        row[f"{side}_{stat}"] = max(
                            0.0, float(rng.normal(base + strength * std * 0.3, std))
                        )

                # GROUP C — situational context (sport-aware defaults)
                row["wind_speed_mph"] = 0.0 if sport == "nba" else float(rng.uniform(0, 15))
                row["temperature_f"] = 72.0 if sport == "nba" else float(rng.uniform(45, 85))
                row["is_dome"] = 1.0 if sport == "nba" else float(rng.choice([0, 1], p=[0.7, 0.3]))
                row["altitude_feet"] = 5280.0 if home in ("DEN", "COL") else 0.0
                row["is_playoff"] = 0.0

                # GROUP D — market signals (neutral at training time)
                row["line_movement"] = 0.0
                row["public_over_pct"] = 0.5
                row["sharp_over_pct"] = 0.5
                row["ref_foul_rate"] = 0.0
                row["ump_walk_rate"] = 0.0

                # GROUP E — realistic injury counts (Poisson λ≈0.7 per team per game,
                # capped at 3; teaches the model the injury signal vs always-zero)
                row["home_key_players_out"] = float(min(3, int(rng.poisson(0.7))))
                row["away_key_players_out"] = float(min(3, int(rng.poisson(0.7))))

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
        return matchups[: n_games * len(teams) // 2]

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
        # Basic
        "passingYards": (240, 60),
        "rushingYards": (120, 40),
        "receivingYards": (240, 60),
        "pointsScored": (23, 10),
        "pointsAllowed": (23, 10),
        "turnovers": (1.5, 1.2),
        "sacks": (2.5, 1.8),
        "thirdDownConversion": (0.42, 0.08),
        "redZoneEfficiency": (0.58, 0.12),
        "timeOfPossession": (30, 4),
        # Advanced offense
        "yardsPerPlay": (5.5, 0.7),
        "redZoneConvRate": (0.58, 0.12),
        "fourthDownConversion": (0.50, 0.15),
        "penaltyYards": (55, 25),
        # Defensive / tempo
        "pressureRate": (0.25, 0.07),
        "secondsPerPlay": (28.0, 3.0),
    },
    "nba": {
        # Basic
        "points": (113, 12),
        "rebounds": (44, 5),
        "assists": (25, 5),
        "steals": (7.5, 2),
        "blocks": (5, 2),
        "turnovers": (14, 3),
        "personalFouls": (20, 4),
        # Shooting volume
        "fieldGoalsMade": (42, 5),
        "fieldGoalAttempts": (87, 8),
        "threesMade": (13, 3),
        "threePointAttempts": (35, 5),
        "freeThrowsMade": (18, 4),
        "freeThrowAttempts": (22, 5),
        # Shooting efficiency
        "fieldGoalPct": (0.47, 0.04),
        "threePointPct": (0.36, 0.05),
        "freeThrowPct": (0.78, 0.06),
        "trueShooting": (0.565, 0.03),
        "ftRate": (0.26, 0.05),
        "threePointRate": (0.41, 0.05),
        # Rebounding
        "offensiveRebounds": (10, 3),
        "defensiveRebounds": (34, 4),
        "drebRate": (0.72, 0.06),
        # Pace / possession
        "possessions": (100, 5),
        "pointsPerPossession": (1.12, 0.08),
        "pace": (100, 3),
        "assistRate": (0.25, 0.04),
        # Ratings
        "offensiveRating": (112, 6),
        "defensiveRating": (112, 6),
        "netRating": (0, 6),
    },
    "mlb": {
        # Basic
        "runsScored": (4.5, 2.5),
        "runsAllowed": (4.5, 2.5),
        "hits": (9, 3),
        "errors": (0.6, 0.8),
        "walks": (3.5, 1.8),
        "strikeouts": (9, 3),
        "era": (4.2, 1.5),
        "whip": (1.30, 0.25),
        "battingAvg": (0.255, 0.025),
        "onBasePct": (0.325, 0.030),
        "sluggingPct": (0.420, 0.045),
        "ops": (0.745, 0.070),
        "homeRuns": (1.2, 1.0),
        # Advanced — only fields available from MLB Stats API boxscore
        "kBbRatio": (3.0, 0.8),
        "groundBallRate": (0.45, 0.06),
        "flyBallRate": (0.35, 0.06),
    },
}
