"""Fetch real NBA game data from the official NBA stats API (no key required)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# DNP comment substrings that indicate a genuine injury absence
_INJURY_KEYWORDS = frozenset(
    {
        "INJURY",
        "ILLNESS",
        "ILL",
        "SICK",
        "PERSONAL",
        "MEDICAL",
        "DND",
        "PROTOCOL",
        "CONCUSSION",
        "KNEE",
        "ANKLE",
        "HAMSTRING",
        "BACK",
        "WRIST",
        "SHOULDER",
        "HIP",
        "CALF",
        "QUAD",
        "GROIN",
        "FOOT",
        "ACHILLES",
        "ELBOW",
        "HAND",
        "FINGER",
        "REST",
    }
)
_NON_INJURY = frozenset({"COACH'S DECISION", "COACHES DECISION"})

logger = logging.getLogger(__name__)

# nba_api column → our internal stat name
_STAT_MAP = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "FGM": "fieldGoalsMade",
    "FGA": "fieldGoalAttempts",
    "FG_PCT": "fieldGoalPct",
    "FG3M": "threesMade",
    "FG3A": "threePointAttempts",
    "FG3_PCT": "threePointPct",
    "FTM": "freeThrowsMade",
    "FTA": "freeThrowAttempts",
    "FT_PCT": "freeThrowPct",
    "OREB": "offensiveRebounds",
    "DREB": "defensiveRebounds",
    "PF": "personalFouls",
    "PLUS_MINUS": "netRating",
}

# Denver Nuggets home arena altitude (feet)
_ALTITUDE_MAP = {"DEN": 5280.0}

_DEFAULT_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
_DELAY = 0.7


def fetch_nba_games(
    seasons: list[str] | None = None,
    delay: float = _DELAY,
) -> pd.DataFrame:
    """
    Pull real NBA regular-season game logs and return a DataFrame matching
    the HistoricalLoader schema, with real advanced stats computed from box scores.
    """
    seasons = seasons or _DEFAULT_SEASONS
    logger.info("Fetching real NBA data for seasons: %s", seasons)

    raw_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []

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
            logger.warning("Failed to fetch team log season %s: %s", season, exc)
        time.sleep(delay)

        # Player-level log has COMMENT column — one call per season, not per game
        try:
            plog = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
                timeout=30,
            )
            pdf = plog.get_data_frames()[0]
            if "COMMENT" in pdf.columns:
                player_frames.append(pdf[["GAME_ID", "TEAM_ID", "COMMENT"]])
        except Exception as exc:
            logger.warning("Failed to fetch player log season %s: %s", season, exc)
        time.sleep(delay)

    if not raw_frames:
        raise RuntimeError("Could not fetch any NBA seasons — check internet connection")

    # Build per-(game_id, team_id) injury count from player DNP comments
    injury_counts: dict[tuple[str, str], int] = defaultdict(int)
    if player_frames:
        player_raw = pd.concat(player_frames, ignore_index=True)
        for _, pr in player_raw.iterrows():
            comment = str(pr.get("COMMENT") or "").upper()
            if not comment:
                continue
            if any(kw in comment for kw in _NON_INJURY):
                continue
            if any(kw in comment for kw in _INJURY_KEYWORDS):
                injury_counts[(str(pr["GAME_ID"]), str(pr["TEAM_ID"]))] += 1

    raw = pd.concat(raw_frames, ignore_index=True)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw["is_home"] = raw["MATCHUP"].str.contains(r" vs\. ", regex=True)

    home = raw[raw["is_home"]].copy()
    away = raw[~raw["is_home"]].copy()
    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))
    logger.info("Merged %d games across %d seasons", len(games), len(seasons))

    rows: list[dict] = []
    for _, g in games.iterrows():
        h_pts = int(g["PTS_home"])
        a_pts = int(g["PTS_away"])
        h_fga = float(g.get("FGA_home", 85) or 85)
        h_fta = float(g.get("FTA_home", 22) or 22)
        h_fg3a = float(g.get("FG3A_home", 33) or 33)
        h_oreb = float(g.get("OREB_home", 10) or 10)
        h_dreb = float(g.get("DREB_home", 34) or 34)
        h_tov = float(g.get("TOV_home", 14) or 14)
        a_fga = float(g.get("FGA_away", 85) or 85)
        a_fta = float(g.get("FTA_away", 22) or 22)
        a_fg3a = float(g.get("FG3A_away", 33) or 33)
        a_oreb = float(g.get("OREB_away", 10) or 10)
        a_dreb = float(g.get("DREB_away", 34) or 34)
        a_tov = float(g.get("TOV_away", 14) or 14)

        # Hollinger possession estimate
        h_poss = max(1.0, h_fga - h_oreb + h_tov + 0.44 * h_fta)
        a_poss = max(1.0, a_fga - a_oreb + a_tov + 0.44 * a_fta)

        home_team = g["TEAM_ABBREVIATION_home"]
        away_team = g["TEAM_ABBREVIATION_away"]
        total = h_pts + a_pts

        row: dict = {
            "game_id": str(g["GAME_ID"]),
            "sport": "nba",
            "season": _season_year(g["GAME_DATE_home"]),
            "game_date": g["GAME_DATE_home"],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": h_pts,
            "away_score": a_pts,
            "total": total,
            "total_line": np.nan,
            "result_over": np.nan,
            "venue": f"{home_team}_arena",
        }

        # Raw box-score stats
        for api_col, stat_name in _STAT_MAP.items():
            row[f"home_{stat_name}"] = float(g.get(f"{api_col}_home", 0) or 0)
            row[f"away_{stat_name}"] = float(g.get(f"{api_col}_away", 0) or 0)

        # Real advanced stats derived from possession formula
        row["home_possessions"] = h_poss
        row["away_possessions"] = a_poss
        row["home_pointsPerPossession"] = h_pts / h_poss
        row["away_pointsPerPossession"] = a_pts / a_poss
        row["home_trueShooting"] = h_pts / max(1.0, 2 * (h_fga + 0.44 * h_fta))
        row["away_trueShooting"] = a_pts / max(1.0, 2 * (a_fga + 0.44 * a_fta))
        row["home_offensiveRating"] = 100.0 * h_pts / h_poss
        row["away_offensiveRating"] = 100.0 * a_pts / a_poss
        row["home_defensiveRating"] = 100.0 * a_pts / h_poss  # pts allowed per 100 home poss
        row["away_defensiveRating"] = 100.0 * h_pts / a_poss
        row["home_pace"] = h_poss
        row["away_pace"] = a_poss
        row["home_assistRate"] = float(g.get("AST_home", 25) or 25) / h_poss
        row["away_assistRate"] = float(g.get("AST_away", 25) or 25) / a_poss
        row["home_drebRate"] = h_dreb / max(1.0, h_dreb + a_oreb)
        row["away_drebRate"] = a_dreb / max(1.0, a_dreb + h_oreb)
        row["home_ftRate"] = h_fta / max(1.0, h_fga)
        row["away_ftRate"] = a_fta / max(1.0, a_fga)
        row["home_threePointRate"] = h_fg3a / max(1.0, h_fga)
        row["away_threePointRate"] = a_fg3a / max(1.0, a_fga)

        # GROUP C — situational context
        row["wind_speed_mph"] = 0.0  # indoor
        row["temperature_f"] = 72.0  # indoor
        row["is_dome"] = 1.0  # NBA is always indoor
        row["altitude_feet"] = _ALTITUDE_MAP.get(home_team, 0.0)
        row["is_playoff"] = 0.0  # regular season only

        # GROUP D — market signals (zero at training; overridden at inference)
        row["line_movement"] = 0.0
        row["public_over_pct"] = 0.5
        row["sharp_over_pct"] = 0.5
        row["ref_foul_rate"] = 0.0
        row["ump_walk_rate"] = 0.0

        # GROUP E — real injury counts from player DNP comments
        game_id_str = str(g["GAME_ID"])
        home_team_id = str(g.get("TEAM_ID_home", ""))
        away_team_id = str(g.get("TEAM_ID_away", ""))
        row["home_key_players_out"] = float(injury_counts.get((game_id_str, home_team_id), 0))
        row["away_key_players_out"] = float(injury_counts.get((game_id_str, away_team_id), 0))

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
    return int(dt.year) if dt.month >= 10 else int(dt.year) - 1


def _compute_proxy_lines(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Bookmaker-style proxy line: blends each team's offensive average against
    the opponent's recent defensive pace, with calibrated noise.
    Uses a 20-game window and separate scored/allowed histories so the line
    is harder to reconstruct from the 10-game rolling features.
    """
    df = df.copy()
    team_scored: dict[str, list[float]] = defaultdict(list)
    team_allowed: dict[str, list[float]] = defaultdict(list)
    rng = np.random.default_rng(42)

    lines: list[float] = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        h_pts, a_pts = float(row["home_score"]), float(row["away_score"])

        h_off = float(np.mean(team_scored[home][-window:])) if team_scored[home] else 113.0
        h_def = float(np.mean(team_allowed[home][-window:])) if team_allowed[home] else 113.0
        a_off = float(np.mean(team_scored[away][-window:])) if team_scored[away] else 111.0
        a_def = float(np.mean(team_allowed[away][-window:])) if team_allowed[away] else 111.0

        expected_home = (h_off + a_def) / 2.0 + 1.5  # home court
        expected_away = (a_off + h_def) / 2.0
        line = expected_home + expected_away + float(rng.normal(0, 3.0))
        line = round(line * 2) / 2
        lines.append(float(np.clip(line, 180.0, 280.0)))

        team_scored[home].append(h_pts)
        team_allowed[home].append(a_pts)
        team_scored[away].append(a_pts)
        team_allowed[away].append(h_pts)

    df["total_line"] = lines
    df["result_over"] = (df["total"] > df["total_line"]).astype(int)
    return df
