"""NBA live team stats for scan-time feature injection.

Fetches the last 20 completed games for every team via nba_api,
computes the same rolling/EMA features the model was trained on,
and caches in memory with a 4-hour TTL.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NBA_TEAM_IDS: dict[str, int] = {
    "ATL": 1610612737, "BKN": 1610612751, "BOS": 1610612738,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}

# nba_api column → our STAT_KEYS["nba"] stat name
_COL_MAP: dict[str, str] = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "PF": "personalFouls",
    "FGM": "fieldGoalsMade",
    "FGA": "fieldGoalAttempts",
    "FG3M": "threesMade",
    "FG3A": "threePointAttempts",
    "FTM": "freeThrowsMade",
    "FTA": "freeThrowAttempts",
    "FG_PCT": "fieldGoalPct",
    "FG3_PCT": "threePointPct",
    "FT_PCT": "freeThrowPct",
    "OREB": "offensiveRebounds",
    "DREB": "defensiveRebounds",
    "PLUS_MINUS": "netRating",
}

_WINDOWS = [3, 5, 10, 20]
_EMA_ALPHAS = [0.3, 0.5]
_CACHE_TTL = 4 * 3600

_cache_ts: float = 0.0
_cache_data: dict[str, dict[str, float]] = {}


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_fga = df["fieldGoalAttempts"].replace(0, np.nan)
    safe_fgm = df["fieldGoalsMade"].replace(0, np.nan)
    safe_reb = df["rebounds"].replace(0, np.nan)
    poss = (
        df["fieldGoalAttempts"]
        - df["offensiveRebounds"]
        + df["turnovers"]
        + 0.44 * df["freeThrowAttempts"]
    )
    safe_poss = poss.replace(0, np.nan)

    df["trueShooting"] = df["points"] / (
        2 * (df["fieldGoalAttempts"] + 0.44 * df["freeThrowAttempts"])
    )
    df["ftRate"] = df["freeThrowAttempts"] / safe_fga
    df["threePointRate"] = df["threePointAttempts"] / safe_fga
    df["drebRate"] = df["defensiveRebounds"] / safe_reb
    df["possessions"] = poss
    df["pointsPerPossession"] = df["points"] / safe_poss
    df["pace"] = poss  # possessions per 48 min — proxy
    df["assistRate"] = df["assists"] / safe_fgm
    df["offensiveRating"] = df["points"] / safe_poss * 100
    df["defensiveRating"] = df["offensiveRating"] - df["netRating"]
    return df.fillna(0.0)


def _fetch_team(
    abbr: str, team_id: int, season: str, delay: float = 0.0
) -> tuple[str, pd.DataFrame]:
    if delay > 0:
        time.sleep(delay)
    try:
        from nba_api.stats.endpoints import teamgamelogs

        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season,
            last_n_games_nullable=20,
            timeout=15,
        )
        raw = logs.get_data_frames()[0]
        if raw.empty:
            return abbr, pd.DataFrame()

        df = pd.DataFrame()
        for api_col, stat_name in _COL_MAP.items():
            if api_col in raw.columns:
                df[stat_name] = pd.to_numeric(raw[api_col], errors="coerce").fillna(0.0)

        df = _compute_derived(df)
        # nba_api returns newest first — reverse so oldest is first for rolling
        return abbr, df.iloc[::-1].reset_index(drop=True)
    except Exception as exc:
        logger.warning("NBA live stats: fetch failed for %s: %s", abbr, exc)
        return abbr, pd.DataFrame()


def _team_to_features(df: pd.DataFrame, stat_cols: list[str]) -> dict[str, float]:
    """Compute rolling/EMA stats from a team's last-N-games DataFrame."""
    feats: dict[str, float] = {}
    for stat in stat_cols:
        if stat not in df.columns:
            continue
        series = df[stat]
        n = len(series)
        for w in _WINDOWS:
            sliced = series.iloc[max(0, n - w):]
            feats[f"{stat}_roll{w}_mean"] = float(sliced.mean()) if len(sliced) > 0 else 0.0
            feats[f"{stat}_roll{w}_std"] = float(sliced.std()) if len(sliced) > 1 else 0.0
            feats[f"{stat}_roll{w}_max"] = float(sliced.max()) if len(sliced) > 0 else 0.0
            feats[f"{stat}_roll{w}_min"] = float(sliced.min()) if len(sliced) > 0 else 0.0
        for alpha in _EMA_ALPHAS:
            ema = series.ewm(alpha=alpha, adjust=False).mean()
            feats[f"{stat}_ema{int(alpha * 10)}"] = float(ema.iloc[-1]) if len(ema) > 0 else 0.0
    return feats


def _current_season() -> str:
    """Return the current NBA season string (e.g. '2024-25')."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    # NBA season starts in October; after July 1 we're in the new season year
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _fetch_teams_needed(
    abbrs: list[str], season: str
) -> dict[str, dict[str, float]]:
    """Fetch live stats for a specific set of teams (no more than needed for a scan)."""
    from proedge.pipeline.ingestion.stats import STAT_KEYS
    stat_cols = STAT_KEYS["nba"]

    results: dict[str, pd.DataFrame] = {}
    # Sequential with a small delay — only 6-10 teams per scan, no rate limit risk
    for i, abbr in enumerate(abbrs):
        team_id = _NBA_TEAM_IDS.get(abbr.upper())
        if team_id is None:
            logger.warning("NBA live stats: unknown team abbreviation %s", abbr)
            results[abbr] = pd.DataFrame()
            continue
        if i > 0:
            time.sleep(0.3)  # brief pause between requests to respect NBA.com limits
        _, df = _fetch_team(abbr, team_id, season)
        results[abbr] = df

    team_features: dict[str, dict[str, float]] = {}
    for abbr, df in results.items():
        if df.empty:
            logger.warning("NBA live stats: empty data for %s — will use medians", abbr)
            team_features[abbr] = {}
        else:
            team_features[abbr] = _team_to_features(df, stat_cols)

    return team_features


def get_teams(abbrs: list[str]) -> dict[str, dict[str, float]]:
    """Return live rolling stats for the specified teams, using cache where valid.

    Only fetches teams missing from the cache or stale (>4h). Designed for
    scan-time use where we only need the handful of teams playing today.
    """
    global _cache_ts, _cache_data

    now = time.time()
    cache_valid = now - _cache_ts < _CACHE_TTL
    season = _current_season()

    # Find which teams need a fresh fetch
    missing = [a for a in abbrs if not cache_valid or a not in _cache_data or not _cache_data[a]]

    if missing:
        logger.info("NBA live stats: fetching %d teams (%s)", len(missing), ", ".join(missing))
        fresh = _fetch_teams_needed(missing, season)
        if cache_valid:
            _cache_data.update(fresh)
        else:
            _cache_data = fresh
            _cache_ts = now
    else:
        logger.info("NBA live stats: all %d teams found in cache", len(abbrs))

    return {a: _cache_data.get(a, {}) for a in abbrs}


def get_all() -> dict[str, dict[str, float]]:
    """Return cached features for ALL 30 teams (lazy-load, full refresh if stale)."""
    global _cache_ts, _cache_data

    if time.time() - _cache_ts < _CACHE_TTL and _cache_data:
        return _cache_data

    season = _current_season()
    logger.info("NBA live stats: full refresh (season=%s, %d teams)", season, len(_NBA_TEAM_IDS))
    _cache_data = _fetch_teams_needed(list(_NBA_TEAM_IDS.keys()), season)
    _cache_ts = time.time()
    logger.info(
        "NBA live stats: cache ready — %d/%d teams with data",
        sum(1 for v in _cache_data.values() if v),
        len(_NBA_TEAM_IDS),
    )
    return _cache_data


def inject_team_features(
    row: dict,
    team_abbr: str,
    prefix: str,
    cache: dict[str, dict[str, float]],
) -> None:
    """Overwrite median values in `row` with live rolling stats for one team.

    prefix is "home_" or "away_".
    Only sets keys that already exist in `row` — never adds new unknown features.
    """
    team_data = cache.get(team_abbr.upper(), {})
    if not team_data:
        return
    for stat_key, value in team_data.items():
        full_key = f"{prefix}{stat_key}"
        if full_key in row:
            row[full_key] = value
