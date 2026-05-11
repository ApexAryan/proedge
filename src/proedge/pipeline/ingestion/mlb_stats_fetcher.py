"""Fetch real MLB regular-season game data from the official MLB Stats API."""
from __future__ import annotations

import logging
import re
import time

import httpx
import numpy as np
import pandas as pd

from proedge.pipeline.ingestion.stats import STAT_KEYS
from proedge.pipeline.ingestion.utils import safe_float as _safe_float, safe_int as _safe_int, compute_proxy_lines as _proxy

logger = logging.getLogger(__name__)

_SCHEDULE_URL  = "https://statsapi.mlb.com/api/v1/schedule"
_BOXSCORE_URL  = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
_TEAMS_URL     = "https://statsapi.mlb.com/api/v1/teams"

_DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023]

# Teams with retractable or permanent roof (weather-neutral)
_MLB_DOME_TEAMS = frozenset({
    "ARI",  # Chase Field (retractable)
    "HOU",  # Minute Maid Park (retractable)
    "MIA",  # loanDepot Park (retractable)
    "MIL",  # American Family Field (retractable)
    "MIN",  # Target Field (open but included conservatively — cold climate)
    "SEA",  # T-Mobile Park (retractable)
    "TB",   # Tropicana Field (fixed dome)
    "TOR",  # Rogers Centre (retractable)
})

# Season date ranges (start, end) — 2020 was the COVID-shortened season
_SEASON_DATES: dict[int, tuple[str, str]] = {
    2019: ("2019-04-01", "2019-10-01"),
    2020: ("2020-07-23", "2020-09-30"),
    2021: ("2021-04-01", "2021-10-01"),
    2022: ("2022-04-01", "2022-10-01"),
    2023: ("2023-04-01", "2023-10-01"),
}

# Coors Field — the only MLB venue with meaningful altitude adjustment
_ALTITUDE_TEAM = "COL"
_COORS_ALTITUDE = 5280.0



# ---------------------------------------------------------------------------
# Team ID → abbreviation cache
# ---------------------------------------------------------------------------

def _fetch_team_map(client: httpx.Client) -> dict[int, str]:
    """
    Fetch all MLB teams from the Stats API and return {team_id: abbreviation}.
    """
    try:
        resp = client.get(_TEAMS_URL, params={"sportId": 1}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return {
            int(t["id"]): t.get("abbreviation", f"T{t['id']}")
            for t in data.get("teams", [])
            if t.get("id") and t.get("abbreviation")
        }
    except Exception as exc:
        logger.warning("Failed to fetch team map: %s — abbreviations may be numeric IDs", exc)
        return {}


# ---------------------------------------------------------------------------
# Schedule fetching
# ---------------------------------------------------------------------------

def _fetch_schedule(
    client: httpx.Client, start_date: str, end_date: str
) -> list[dict]:
    """
    Return the flat list of game dicts for a date range.
    Only games with detailedState == 'Final' are included.
    """
    games: list[dict] = []
    try:
        resp = client.get(
            _SCHEDULE_URL,
            params={
                "sportId":   1,
                "startDate": start_date,
                "endDate":   end_date,
                "gameType":  "R",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for date_block in data.get("dates", []):
            for g in date_block.get("games", []):
                if g.get("status", {}).get("detailedState") == "Final":
                    games.append(g)
    except Exception as exc:
        logger.warning(
            "Schedule fetch failed (%s – %s): %s", start_date, end_date, exc
        )
    return games


# ---------------------------------------------------------------------------
# Boxscore fetching and parsing
# ---------------------------------------------------------------------------

def _fetch_boxscore(client: httpx.Client, game_pk: int) -> dict:
    try:
        url = _BOXSCORE_URL.format(game_pk=game_pk)
        resp = client.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Boxscore fetch failed gamePk=%d: %s", game_pk, exc)
        return {}


_IL_STATUS_KEYWORDS = frozenset({"IL", "INJURED LIST", "DISABLED LIST", "DL", "DL-"})


def _count_il_players(side_data: dict) -> int:
    """Count players listed on any injured list from boxscore side data."""
    count = 0
    for player_entry in side_data.get("players", {}).values():
        status = str(player_entry.get("status", {}).get("code", "")).upper()
        if any(kw in status for kw in _IL_STATUS_KEYWORDS):
            count += 1
    return count


def _parse_side_stats(side_data: dict) -> dict[str, float]:
    """
    Parse batting + pitching teamStats from one side's boxscore data.
    Returns a dict with all STAT_KEYS["mlb"] fields populated.
    """
    batting  = side_data.get("teamStats", {}).get("batting", {})
    pitching = side_data.get("teamStats", {}).get("pitching", {})

    runs_scored  = _safe_int(batting.get("runs", 0))
    hits         = _safe_int(batting.get("hits", 0))
    home_runs    = _safe_int(batting.get("homeRuns", 0))
    bat_strikeouts = _safe_int(batting.get("strikeOuts", 0))
    walks        = _safe_int(batting.get("baseOnBalls", 0))
    batting_avg  = _safe_float(batting.get("avg", "0.000"))
    obp          = _safe_float(batting.get("obp", "0.000"))
    slg          = _safe_float(batting.get("slg", "0.000"))
    ops          = _safe_float(batting.get("ops", "0.000"))

    runs_allowed    = _safe_int(pitching.get("runs", 0))
    pit_strikeouts  = _safe_int(pitching.get("strikeOuts", 0))
    pit_walks       = _safe_int(pitching.get("baseOnBalls", 0))
    era             = _safe_float(pitching.get("era", "0.00"))
    whip            = _safe_float(pitching.get("whip", "0.00"))

    # Ground/fly ball rates from pitching stats
    fly_outs    = _safe_int(pitching.get("flyOuts", 0))
    ground_outs = _safe_int(pitching.get("groundOuts", 0))
    total_outs  = max(1, fly_outs + ground_outs)
    ground_ball_rate = ground_outs / total_outs
    fly_ball_rate    = fly_outs / total_outs

    # Derived
    kb_ratio = pit_strikeouts / max(1, pit_walks)

    # Errors come from the top-level boxscore side dict, not teamStats
    errors = _safe_int(side_data.get("errors", 0))

    return {
        "runsScored":    float(runs_scored),
        "runsAllowed":   float(runs_allowed),
        "hits":          float(hits),
        "errors":        float(errors),
        "walks":         float(walks),
        "strikeouts":    float(bat_strikeouts),
        "era":           era,
        "whip":          whip,
        "battingAvg":    batting_avg,
        "onBasePct":     obp,
        "sluggingPct":   slg,
        "ops":           ops,
        "homeRuns":      float(home_runs),
        "kBbRatio":       kb_ratio,
        "groundBallRate": ground_ball_rate,
        "flyBallRate":    fly_ball_rate,
    }


def _build_game_row(
    game: dict,
    boxscore: dict,
    team_map: dict[int, str],
    season: int,
) -> dict | None:
    """
    Combine a schedule game entry with its boxscore into a flat row dict.
    Returns None if the row cannot be built.
    """
    try:
        game_pk   = int(game.get("gamePk", 0))
        game_date_raw = game.get("gameDate", game.get("officialDate", ""))
        try:
            game_date = pd.Timestamp(game_date_raw).tz_localize(None)
        except Exception:
            game_date = pd.NaT

        teams_sched = game.get("teams", {})
        home_id = int(teams_sched.get("home", {}).get("team", {}).get("id", 0))
        away_id = int(teams_sched.get("away", {}).get("team", {}).get("id", 0))

        home_score = _safe_int(teams_sched.get("home", {}).get("score", 0))
        away_score = _safe_int(teams_sched.get("away", {}).get("score", 0))
        total = home_score + away_score

        home_abbr = team_map.get(home_id, f"T{home_id}")
        away_abbr = team_map.get(away_id, f"T{away_id}")

        box_teams = boxscore.get("teams", {})
        home_side = box_teams.get("home", {})
        away_side = box_teams.get("away", {})

        home_stats = _parse_side_stats(home_side)
        away_stats = _parse_side_stats(away_side)

        # Altitude — Coors Field for Colorado home games
        altitude = _COORS_ALTITUDE if home_abbr == _ALTITUDE_TEAM else 0.0

        # GROUP C — weather from schedule game object
        # MLB Stats API embeds weather as {"condition": "Sunny", "temp": "75", "wind": "8 mph, Out To CF"}
        weather_data = game.get("weather", {})
        temperature_f = _safe_float(weather_data.get("temp", 70.0))
        wind_raw = weather_data.get("wind", "0")
        try:
            m = re.match(r"(\d+)", str(wind_raw))
            wind_speed_mph = float(m.group(1)) if m else 0.0
        except Exception:
            wind_speed_mph = 0.0

        is_dome = float(home_abbr in _MLB_DOME_TEAMS)
        if is_dome:
            wind_speed_mph = 0.0

        row: dict = {
            "game_id":    str(game_pk),
            "sport":      "mlb",
            "season":     season,
            "game_date":  game_date,
            "home_team":  home_abbr,
            "away_team":  away_abbr,
            "home_score": home_score,
            "away_score": away_score,
            "total":      total,
            "total_line": np.nan,
            "result_over": np.nan,
            "venue":      f"{home_abbr}_stadium",
        }

        # Per-team stat columns — iterate STAT_KEYS["mlb"] for guaranteed coverage
        for stat in STAT_KEYS["mlb"]:
            row[f"home_{stat}"] = home_stats.get(stat, 0.0)
            row[f"away_{stat}"] = away_stats.get(stat, 0.0)

        # GROUP C
        row["wind_speed_mph"]  = wind_speed_mph
        row["temperature_f"]   = temperature_f if temperature_f > 0 else 70.0
        row["is_dome"]         = is_dome
        row["altitude_feet"]   = altitude
        row["is_playoff"]      = 0.0

        # GROUP D
        row["line_movement"]   = 0.0
        row["public_over_pct"] = 0.5
        row["sharp_over_pct"]  = 0.5
        row["ref_foul_rate"]   = 0.0
        row["ump_walk_rate"]   = 0.0

        # GROUP E — count players on the IL from the boxscore roster
        row["home_key_players_out"] = float(_count_il_players(home_side))
        row["away_key_players_out"] = float(_count_il_players(away_side))

        return row

    except Exception as exc:
        logger.warning(
            "Failed to build row for gamePk=%s: %s", game.get("gamePk"), exc
        )
        return None


# ---------------------------------------------------------------------------
# Proxy line
# ---------------------------------------------------------------------------

def _compute_proxy_lines(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return _proxy(df, clip_lo=4.0, clip_hi=25.0,
                  home_off_default=4.5, home_def_default=4.5,
                  away_off_default=4.5, away_def_default=4.5,
                  window=window)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_mlb_games(
    seasons: list[int] | None = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch MLB regular-season game data from the official MLB Stats API for the
    given seasons and return a DataFrame matching the HistoricalLoader schema.

    Parameters
    ----------
    seasons:
        Calendar years to fetch (e.g. [2022, 2023]). Defaults to 2019-2023.
    delay:
        Seconds to sleep between boxscore requests to respect the API.

    Returns
    -------
    pd.DataFrame
        One row per game with all columns required by HistoricalLoader.
        Returns an empty DataFrame if all fetches fail.
    """
    seasons = seasons or _DEFAULT_SEASONS
    logger.info("Fetching real MLB data for seasons: %s", seasons)

    rows: list[dict] = []

    with httpx.Client(follow_redirects=True) as client:
        # Build team ID → abbreviation lookup once
        team_map = _fetch_team_map(client)
        logger.info("Loaded %d MLB team abbreviations", len(team_map))

        for season in seasons:
            start_date, end_date = _SEASON_DATES.get(
                season, (f"{season}-04-01", f"{season}-10-01")
            )
            logger.info(
                "  → MLB season %d (%s – %s)", season, start_date, end_date
            )

            games = _fetch_schedule(client, start_date, end_date)
            logger.info(
                "    Found %d final games in schedule", len(games)
            )

            for game in games:
                game_pk = game.get("gamePk")
                if not game_pk:
                    continue

                boxscore = _fetch_boxscore(client, int(game_pk))
                time.sleep(delay)

                row = _build_game_row(game, boxscore, team_map, season)
                if row is not None:
                    rows.append(row)

    if not rows:
        logger.error("No MLB game rows collected — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)

    # De-duplicate by game_id in case schedule pages overlap
    df = (
        df.drop_duplicates(subset=["game_id"])
          .sort_values("game_date")
          .reset_index(drop=True)
    )

    df = _compute_proxy_lines(df)

    logger.info(
        "Real MLB dataset: %d games | over rate: %.1f%% | avg total: %.1f",
        len(df),
        df["result_over"].mean() * 100,
        df["total"].mean(),
    )
    return df
