"""Fetch real NBA game data from the official NBA stats API (no key required)."""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, teamgamelogs

logger = logging.getLogger(__name__)

# nba_api column → our internal stat name
_STAT_MAP = {
    "PTS":      "points",
    "REB":      "rebounds",
    "AST":      "assists",
    "STL":      "steals",
    "BLK":      "blocks",
    "TOV":      "turnovers",
    "FG_PCT":   "fieldGoalPct",
    "FG3_PCT":  "threePointPct",
    "FT_PCT":   "freeThrowPct",
    "OREB":     "offensiveRebounds",
    "PLUS_MINUS": "netRating",
}

# Seasons to fetch: 2019-20 through 2023-24
_DEFAULT_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

# Request delay to avoid rate-limiting NBA stats site
_DELAY = 0.7


def fetch_nba_games(
    seasons: list[str] | None = None,
    delay: float = _DELAY,
) -> pd.DataFrame:
    """
    Pull real NBA regular-season game logs for the requested seasons and
    return a DataFrame that matches the HistoricalLoader schema.
    """
    seasons = seasons or _DEFAULT_SEASONS
    logger.info("Fetching real NBA data for seasons: %s", seasons)

    raw_frames: list[pd.DataFrame] = []
    for season in seasons:
        logger.info("  → season %s", season)
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                timeout=30,
            )
            df = log.get_data_frames()[0]
            raw_frames.append(df)
        except Exception as exc:
            logger.warning("Failed to fetch season %s: %s", season, exc)
        time.sleep(delay)

    if not raw_frames:
        raise RuntimeError("Could not fetch any NBA seasons — check internet connection")

    raw = pd.concat(raw_frames, ignore_index=True)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])

    # Each game has two rows (one per team). Split by home/away then join.
    raw["is_home"] = raw["MATCHUP"].str.contains(r" vs\. ", regex=True)
    home = raw[raw["is_home"]].copy()
    away = raw[~raw["is_home"]].copy()

    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))
    logger.info("Merged %d games across %d seasons", len(games), len(seasons))

    rows: list[dict] = []
    for _, g in games.iterrows():
        h_pts = int(g["PTS_home"])
        a_pts = int(g["PTS_away"])
        total = h_pts + a_pts

        row: dict = {
            "game_id": str(g["GAME_ID"]),
            "sport":   "nba",
            "season":  _season_year(g["GAME_DATE_home"]),
            "game_date": g["GAME_DATE_home"],
            "home_team": g["TEAM_ABBREVIATION_home"],
            "away_team": g["TEAM_ABBREVIATION_away"],
            "home_score": h_pts,
            "away_score": a_pts,
            "total":      total,
            "total_line": np.nan,   # filled by _compute_proxy_lines
            "result_over": np.nan,  # filled after proxy line
            "venue": f"{g['TEAM_ABBREVIATION_home']}_arena",
        }

        for api_col, stat_name in _STAT_MAP.items():
            row[f"home_{stat_name}"] = float(g.get(f"{api_col}_home", 0) or 0)
            row[f"away_{stat_name}"] = float(g.get(f"{api_col}_away", 0) or 0)

        # Approximate advanced stats from basic box score
        row["home_offensiveRating"] = float(h_pts)         # correlated proxy
        row["away_offensiveRating"] = float(a_pts)
        row["home_defensiveRating"] = float(a_pts)         # points allowed
        row["away_defensiveRating"] = float(h_pts)
        row["home_pace"] = 100.0                           # default; no per-game pace in basic log
        row["away_pace"] = 100.0

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
    df = _compute_proxy_lines(df)

    logger.info(
        "Real NBA dataset: %d games | over rate: %.1f%% | avg total: %.1f",
        len(df),
        df["result_over"].mean() * 100,
        df["total"].mean(),
    )
    return df


def _season_year(dt: pd.Timestamp) -> int:
    """NBA season year is the year the season *starts* (Oct → Apr)."""
    return int(dt.year) if dt.month >= 10 else int(dt.year) - 1


def _compute_proxy_lines(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Build a realistic proxy over/under line using a bookmaker-style formula:

        line ≈ home_off_avg + away_off_avg
                  weighted against each team's recent defensive pace

    We use separate offensive (points scored) and defensive (points allowed)
    rolling averages over a wider 20-game window.  This is much harder to
    reconstruct from the 10-game rolling features the model sees, giving a
    realistic 50-55 % prediction accuracy instead of a trivially learnable line.
    """
    df = df.copy()

    # Track each team's scored/allowed history separately
    team_scored:  dict[str, list[float]] = defaultdict(list)
    team_allowed: dict[str, list[float]] = defaultdict(list)

    lines: list[float] = []
    rng = np.random.default_rng(42)

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        h_pts, a_pts = float(row["home_score"]), float(row["away_score"])

        h_off = float(np.mean(team_scored[home][-window:]))  if team_scored[home]  else 113.0
        h_def = float(np.mean(team_allowed[home][-window:])) if team_allowed[home] else 113.0
        a_off = float(np.mean(team_scored[away][-window:]))  if team_scored[away]  else 111.0
        a_def = float(np.mean(team_allowed[away][-window:])) if team_allowed[away] else 111.0

        # Bookmaker blends offense vs opponent defense; home court adds ~2 pts
        expected_home = (h_off + a_def) / 2.0 + 1.5
        expected_away = (a_off + h_def) / 2.0
        line = expected_home + expected_away

        # Bookmakers also have their own private edge — model it as small noise
        line += float(rng.normal(0, 3.0))

        # Round to nearest 0.5 like real lines
        line = round(line * 2) / 2
        lines.append(max(180.0, min(280.0, line)))   # sanity clamp

        team_scored[home].append(h_pts)
        team_allowed[home].append(a_pts)
        team_scored[away].append(a_pts)
        team_allowed[away].append(h_pts)

    df["total_line"] = lines
    df["result_over"] = (df["total"] > df["total_line"]).astype(int)
    return df
