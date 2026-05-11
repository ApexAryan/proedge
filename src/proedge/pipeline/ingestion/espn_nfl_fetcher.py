"""Fetch real NFL regular-season game data from ESPN's public (unofficial) API."""

from __future__ import annotations

import logging
import re
import time

import httpx
import numpy as np
import pandas as pd

from proedge.pipeline.ingestion.stats import STAT_KEYS
from proedge.pipeline.ingestion.utils import (
    safe_int as _safe_int,
    safe_float as _safe_float,
    compute_proxy_lines as _proxy,
)

logger = logging.getLogger(__name__)

_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

_DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

# Teams that play home games in a dome (full or retractable-roof)
_DOME_HOME_TEAMS = frozenset(
    {
        "ATL",  # Mercedes-Benz Stadium
        "NO",  # Caesars Superdome
        "IND",  # Lucas Oil Stadium
        "LV",  # Allegiant Stadium
        "MIN",  # U.S. Bank Stadium
        "HOU",  # NRG Stadium
        "ARI",  # State Farm Stadium (retractable)
        "DET",  # Ford Field
        "DAL",  # AT&T Stadium (retractable)
        "LAR",  # SoFi Stadium (covered)
        "LAC",  # SoFi Stadium (covered)
    }
)

# Home teams whose stadiums are at meaningful altitude (feet above sea level)
_ALTITUDE_FEET: dict[str, float] = {
    "DEN": 5280.0,  # Empower Field at Mile High
}

# Seasons that had a 17-week regular season schedule vs. the older 16-week/17-game distinction.
# ESPN week numbering: 2019–2020 had 16 games (17 weeks max with bye), 2021+ had 18 weeks.
_SEASON_WEEKS = {
    2019: range(1, 18),  # 16-game season, weeks 1-17
    2020: range(1, 18),  # 16-game season, weeks 1-17
    2021: range(1, 19),  # 17-game season, weeks 1-18
    2022: range(1, 19),
    2023: range(1, 19),
    2024: range(1, 19),
}


def _parse_stat(stat_list: list[dict], label: str) -> str | None:
    """Return the displayValue for a given label from a statistics list."""
    for item in stat_list:
        if item.get("label") == label or item.get("name") == label:
            return item.get("displayValue")
    return None


def _parse_fraction(value: str | None, default: float = 0.0) -> float:
    """Parse 'X-Y' efficiency strings to X/Y. Returns default if 0-0 or unparseable."""
    if value is None:
        return default
    try:
        parts = value.replace(" ", "").split("-")
        num, den = float(parts[0]), float(parts[1])
        return num / den if den > 0 else default
    except (IndexError, ValueError, ZeroDivisionError):
        return default


def _parse_first_number(value: str | None, default: float = 0.0) -> float:
    """Parse the first number from a 'X-Y' string (e.g. sacks)."""
    if value is None:
        return default
    try:
        return float(value.split("-")[0].strip())
    except (ValueError, IndexError):
        return default


def _parse_second_number(value: str | None, default: float = 0.0) -> float:
    """Parse the second number from a 'X-Y' string (e.g. penalty yards)."""
    if value is None:
        return default
    try:
        return float(value.split("-")[1].strip())
    except (ValueError, IndexError):
        return default


def _parse_time_of_possession(value: str | None) -> float:
    """Parse 'MM:SS' into total minutes as a float."""
    if value is None:
        return 30.0
    try:
        parts = value.strip().split(":")
        minutes = float(parts[0])
        seconds = float(parts[1]) if len(parts) > 1 else 0.0
        return minutes + seconds / 60.0
    except (ValueError, IndexError):
        return 30.0


def _parse_team_stats(stat_list: list[dict]) -> dict[str, float]:
    """
    Parse a boxscore statistics list into a flat dict of our internal stat names.
    """
    passing_yards = _safe_int(_parse_stat(stat_list, "Passing"))
    rushing_yards = _safe_int(_parse_stat(stat_list, "Rushing"))
    total_yards_raw = _parse_stat(stat_list, "Total Yards")
    total_yards = _safe_int(total_yards_raw)
    turnovers = _safe_int(_parse_stat(stat_list, "Turnovers"))
    sacks = _parse_first_number(_parse_stat(stat_list, "Sacks-Yards Lost"))
    third_down = _parse_fraction(_parse_stat(stat_list, "3rd down efficiency"))
    red_zone = _parse_fraction(_parse_stat(stat_list, "Red Zone (Made-Att)"))
    possession = _parse_time_of_possession(_parse_stat(stat_list, "Possession"))
    yards_per_play = _safe_float(_parse_stat(stat_list, "Yards per Play"))
    fourth_down = _parse_fraction(_parse_stat(stat_list, "4th down efficiency"), default=0.0)
    penalty_yards = _parse_second_number(_parse_stat(stat_list, "Penalties"))

    # Derived stats
    total_plays_raw = _parse_stat(stat_list, "Total Plays")
    total_plays = max(1, _safe_int(total_plays_raw, default=60))

    comp_att_raw = _parse_stat(stat_list, "Comp/Att")
    passing_attempts = 1
    if comp_att_raw:
        try:
            passing_attempts = max(1, int(comp_att_raw.split("/")[1].strip()))
        except (ValueError, IndexError):
            passing_attempts = 1

    seconds_per_play = possession * 60.0 / total_plays
    pressure_rate = sacks / passing_attempts

    return {
        "passingYards": float(passing_yards),
        "rushingYards": float(rushing_yards),
        "receivingYards": float(total_yards),  # proxy: totalYards stored here
        "turnovers": float(turnovers),
        "sacks": float(sacks),
        "thirdDownConversion": third_down,
        "redZoneEfficiency": red_zone,
        "timeOfPossession": possession,
        "yardsPerPlay": yards_per_play,
        "fourthDownConversion": fourth_down,
        "penaltyYards": penalty_yards,
        "pressureRate": pressure_rate,
        "secondsPerPlay": seconds_per_play,
        # pointsScored/pointsAllowed populated from score after
        "pointsScored": 0.0,
        "pointsAllowed": 0.0,
        "redZoneConvRate": red_zone,  # same as redZoneEfficiency
    }


def _count_injuries(competitor: dict) -> int:
    """Count players listed as Out or Doubtful from competitor injuries."""
    injuries = competitor.get("injuries", [])
    count = 0
    for inj in injuries:
        status = inj.get("status", "") or inj.get("type", {}).get("name", "")
        if isinstance(status, str) and status.upper() in {"OUT", "DOUBTFUL"}:
            count += 1
    return count


def _fetch_scoreboard(client: httpx.Client, season: int, week: int) -> list[dict]:
    """Return the list of event dicts for a given season/week."""
    try:
        resp = client.get(
            _SCOREBOARD_URL,
            params={"week": week, "seasontype": 2, "season": season},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])
    except Exception as exc:
        logger.warning("Scoreboard fetch failed season=%d week=%d: %s", season, week, exc)
        return []


def _fetch_summary(client: httpx.Client, game_id: str) -> dict:
    """Return the full summary JSON for a single game."""
    try:
        resp = client.get(_SUMMARY_URL, params={"event": game_id}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Summary fetch failed game_id=%s: %s", game_id, exc)
        return {}


def _build_game_row(
    event: dict,
    summary: dict,
    season: int,
) -> dict | None:
    """
    Parse a single event + its summary into a flat row dict.
    Returns None if the game cannot be parsed.
    """
    try:
        competition = event.get("competitions", [{}])[0]

        # Confirm final status
        status_name = competition.get("status", {}).get("type", {}).get("name", "")
        if status_name != "STATUS_FINAL":
            return None

        game_id = str(event.get("id", ""))
        game_date_raw = event.get("date", "")
        try:
            game_date = pd.Timestamp(game_date_raw).tz_localize(None)
        except Exception:
            game_date = pd.NaT

        competitors: list[dict] = competition.get("competitors", [])
        if len(competitors) < 2:
            return None

        # Identify home and away
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_team = home_comp.get("team", {}).get("abbreviation", "UNK")
        away_team = away_comp.get("team", {}).get("abbreviation", "UNK")
        home_score = _safe_int(home_comp.get("score", 0))
        away_score = _safe_int(away_comp.get("score", 0))
        total = home_score + away_score

        # Injury counts from competition competitors (event-level)
        home_injuries = _count_injuries(home_comp)
        away_injuries = _count_injuries(away_comp)

        # Parse boxscore statistics from summary
        boxscore_teams = summary.get("boxscore", {}).get("teams", [])

        home_stats: dict[str, float] = {}
        away_stats: dict[str, float] = {}

        for team_entry in boxscore_teams:
            team_info = team_entry.get("team", {})
            abbrev = team_info.get("abbreviation", "")
            stat_list = team_entry.get("statistics", [])
            parsed = _parse_team_stats(stat_list)
            if abbrev == home_team:
                home_stats = parsed
            elif abbrev == away_team:
                away_stats = parsed

        # Fall back: try matching by index if abbreviations don't match
        if not home_stats and len(boxscore_teams) >= 1:
            home_stats = _parse_team_stats(boxscore_teams[0].get("statistics", []))
        if not away_stats and len(boxscore_teams) >= 2:
            away_stats = _parse_team_stats(boxscore_teams[1].get("statistics", []))

        # Fill pointsScored / pointsAllowed
        home_stats["pointsScored"] = float(home_score)
        home_stats["pointsAllowed"] = float(away_score)
        away_stats["pointsScored"] = float(away_score)
        away_stats["pointsAllowed"] = float(home_score)

        # GROUP C — situational context
        # ESPN competition object includes a "weather" dict when available
        weather = competition.get("weather", {})
        temperature_f = _safe_float(weather.get("temperature", None))
        wind_speed_mph = _safe_float(weather.get("windSpeed", None))
        # displayValue e.g. "63° F, Wind 8 mph" — parse wind if windSpeed absent
        if wind_speed_mph == 0.0 and not weather.get("windSpeed"):
            disp = weather.get("displayValue", "")
            m = re.search(r"(\d+)\s*mph", disp, re.I)
            wind_speed_mph = float(m.group(1)) if m else 0.0
        # Use sensible fallback if ESPN returned no weather at all
        if temperature_f == 0.0 and not weather.get("temperature"):
            temperature_f = 55.0

        is_dome = float(home_team in _DOME_HOME_TEAMS)
        # Domes have no wind and controlled temperature
        if is_dome:
            wind_speed_mph = 0.0
            temperature_f = temperature_f if temperature_f != 55.0 else 72.0

        altitude_feet = _ALTITUDE_FEET.get(home_team, 0.0)

        venue_info = competition.get("venue", {})
        venue_name = venue_info.get("fullName", f"{home_team}_stadium")

        row: dict = {
            "game_id": game_id,
            "sport": "nfl",
            "season": season,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "total": total,
            "total_line": np.nan,
            "result_over": np.nan,
            "venue": venue_name,
        }

        # Per-team stat columns — use STAT_KEYS["nfl"] to guarantee coverage
        for stat in STAT_KEYS["nfl"]:
            row[f"home_{stat}"] = home_stats.get(stat, 0.0)
            row[f"away_{stat}"] = away_stats.get(stat, 0.0)

        # GROUP C
        row["wind_speed_mph"] = wind_speed_mph
        row["temperature_f"] = temperature_f
        row["is_dome"] = is_dome
        row["altitude_feet"] = altitude_feet
        row["is_playoff"] = 0.0

        # GROUP D
        row["line_movement"] = 0.0
        row["public_over_pct"] = 0.5
        row["sharp_over_pct"] = 0.5
        row["ref_foul_rate"] = 0.0
        row["ump_walk_rate"] = 0.0

        # GROUP E
        row["home_key_players_out"] = float(home_injuries)
        row["away_key_players_out"] = float(away_injuries)

        return row

    except Exception as exc:
        logger.warning("Failed to build row for game %s: %s", event.get("id"), exc)
        return None


def _compute_proxy_lines(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return _proxy(
        df,
        clip_lo=28.0,
        clip_hi=80.0,
        home_off_default=23.0,
        home_def_default=21.0,
        away_off_default=21.0,
        away_def_default=23.0,
        window=window,
    )


def fetch_nfl_games(
    seasons: list[int] | None = None,
    delay: float = 0.4,
) -> pd.DataFrame:
    """
    Fetch NFL regular-season game data from ESPN's public API for the given
    seasons and return a DataFrame matching the HistoricalLoader schema.

    Parameters
    ----------
    seasons:
        Calendar years to fetch (e.g. [2022, 2023]). Defaults to 2019-2024.
    delay:
        Seconds to sleep between game-summary requests. Scoreboard requests
        use delay / 4 to respect ESPN's rate limits.

    Returns
    -------
    pd.DataFrame
        One row per game with all columns required by HistoricalLoader.
        Returns an empty DataFrame if all fetches fail.
    """
    seasons = seasons or _DEFAULT_SEASONS
    logger.info("Fetching real NFL data for seasons: %s", seasons)

    rows: list[dict] = []

    with httpx.Client(follow_redirects=True) as client:
        for season in seasons:
            week_range = _SEASON_WEEKS.get(season, range(1, 19))
            logger.info(
                "  → NFL season %d (weeks %d-%d)", season, week_range.start, week_range.stop - 1
            )

            for week in week_range:
                events = _fetch_scoreboard(client, season, week)
                time.sleep(delay / 4)

                if not events:
                    continue

                # Filter to final games only before fetching summaries
                final_events = [
                    ev
                    for ev in events
                    if ev.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("name")
                    == "STATUS_FINAL"
                ]

                logger.debug(
                    "Season %d week %d: %d total events, %d final",
                    season,
                    week,
                    len(events),
                    len(final_events),
                )

                for event in final_events:
                    game_id = str(event.get("id", ""))
                    if not game_id:
                        continue

                    summary = _fetch_summary(client, game_id)
                    time.sleep(delay)

                    row = _build_game_row(event, summary, season)
                    if row is not None:
                        rows.append(row)

    if not rows:
        logger.error("No NFL game rows collected — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)

    # De-duplicate (same game_id may appear in multiple week pages near season edges)
    df = df.drop_duplicates(subset=["game_id"]).sort_values("game_date").reset_index(drop=True)

    df = _compute_proxy_lines(df)

    logger.info(
        "Real NFL dataset: %d games | over rate: %.1f%% | avg total: %.1f",
        len(df),
        df["result_over"].mean() * 100,
        df["total"].mean(),
    )
    return df
