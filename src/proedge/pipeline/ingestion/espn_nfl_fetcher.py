"""Fetch real NFL regular-season game data via nfl-data-py (nflverse)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from proedge.pipeline.ingestion.stats import STAT_KEYS

logger = logging.getLogger(__name__)

_DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

# Teams that play in a dome or retractable-roof stadium
_DOME_HOME_TEAMS = frozenset({
    "ATL", "NO", "IND", "LV", "MIN", "HOU", "ARI", "DET", "DAL", "LA", "LAR", "LAC",
})

_ALTITUDE_FEET: dict[str, float] = {"DEN": 5280.0}

# nflverse uses different abbreviations for some teams — normalize to our schema
_ABBREV_MAP = {
    "LA": "LAR",   # Rams (nflverse uses "LA", we use "LAR")
    "JAC": "JAX",  # Jaguars
    "OAK": "LV",   # Raiders moved to Las Vegas in 2020
}


def fetch_nfl_games(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Fetch NFL regular-season game data from nflverse via nfl-data-py.
    Returns real betting lines (total_line) and real outcomes (result_over).
    """
    import nfl_data_py as nfl

    seasons = seasons or _DEFAULT_SEASONS
    logger.info("Fetching real NFL schedule data for seasons: %s", seasons)

    try:
        schedules = nfl.import_schedules(seasons)
    except Exception as exc:
        logger.error("nfl-data-py schedule fetch failed: %s", exc)
        return pd.DataFrame()

    # Keep only completed regular-season games with known score
    schedules = schedules[
        (schedules["game_type"] == "REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].copy()

    logger.info("Raw schedule rows after filtering: %d", len(schedules))

    rows = []
    for _, g in schedules.iterrows():
        home = _ABBREV_MAP.get(str(g.get("home_team", "")), str(g.get("home_team", "")))
        away = _ABBREV_MAP.get(str(g.get("away_team", "")), str(g.get("away_team", "")))
        home_score = int(g["home_score"])
        away_score = int(g["away_score"])
        total = home_score + away_score

        # Real total line from nflverse (market line, NaN if unavailable)
        total_line_raw = g.get("total_line", np.nan)
        try:
            total_line = float(total_line_raw) if pd.notna(total_line_raw) else np.nan
        except (TypeError, ValueError):
            total_line = np.nan

        # Derive result_over only when we have a real line
        if pd.notna(total_line):
            result_over = int(total > total_line)
        else:
            result_over = np.nan

        game_date = pd.Timestamp(g.get("gameday", g.get("game_date", pd.NaT)))

        # Situational context
        roof = str(g.get("roof", "")).lower()
        is_dome = float(roof in {"dome", "retractable"} or home in _DOME_HOME_TEAMS)
        temp = g.get("temp", None)
        wind = g.get("wind", None)
        try:
            temperature_f = float(temp) if pd.notna(temp) else (72.0 if is_dome else 55.0)
        except (TypeError, ValueError):
            temperature_f = 72.0 if is_dome else 55.0
        try:
            wind_speed_mph = float(wind) if pd.notna(wind) else 0.0
        except (TypeError, ValueError):
            wind_speed_mph = 0.0
        if is_dome:
            wind_speed_mph = 0.0

        row: dict = {
            "game_id": str(g.get("game_id", "")),
            "sport": "nfl",
            "season": int(g.get("season", 0)),
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "total": total,
            "total_line": total_line,
            "result_over": result_over,
            "venue": str(g.get("stadium", f"{home}_stadium")),
            # Situational
            "wind_speed_mph": wind_speed_mph,
            "temperature_f": temperature_f,
            "is_dome": is_dome,
            "altitude_feet": _ALTITUDE_FEET.get(home, 0.0),
            "is_playoff": 0.0,
            # Market signals
            "line_movement": 0.0,
            "public_over_pct": 0.5,
            "sharp_over_pct": 0.5,
            "ref_foul_rate": 0.0,
            "ump_walk_rate": 0.0,
            # Injuries
            "home_key_players_out": 0.0,
            "away_key_players_out": 0.0,
        }

        # Stat columns — nflverse schedule doesn't have play-level stats;
        # zero-fill so rolling features gracefully degrade to 0-mean.
        for stat in STAT_KEYS.get("nfl", []):
            row[f"home_{stat}"] = 0.0
            row[f"away_{stat}"] = 0.0

        # Fill pointsScored/pointsAllowed from actual scores
        row["home_pointsScored"] = float(home_score)
        row["home_pointsAllowed"] = float(away_score)
        row["away_pointsScored"] = float(away_score)
        row["away_pointsAllowed"] = float(home_score)

        rows.append(row)

    if not rows:
        logger.error("No NFL game rows collected — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)
    df = df.dropna(subset=["result_over"])  # drop games with no usable line
    df = df.drop_duplicates(subset=["game_id"]).sort_values("game_date").reset_index(drop=True)

    logger.info(
        "Real NFL dataset: %d games | over rate: %.1f%% | avg total: %.1f | avg line: %.1f",
        len(df),
        df["result_over"].mean() * 100,
        df["total"].mean(),
        df["total_line"].mean(),
    )
    return df
