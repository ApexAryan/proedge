"""MLB live team stats for scan-time feature injection.

Fetches the last 20 completed games for each team via the official MLB Stats API,
computes the same rolling/EMA features the model was trained on, and caches
in memory with a 4-hour TTL.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"
_CACHE_TTL = 4 * 3600

_cache_ts: float = 0.0
_cache_data: dict[str, dict[str, float]] = {}

_WINDOWS = [3, 5, 10, 20]
_EMA_ALPHAS = [0.3, 0.5]

# Lazy-loaded on first call
_team_id_map: dict[str, int] = {}


def _load_team_ids() -> dict[str, int]:
    global _team_id_map
    if _team_id_map:
        return _team_id_map
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{_BASE}/teams", params={"sportId": 1})
            resp.raise_for_status()
            teams = resp.json().get("teams", [])
            _team_id_map = {
                t["abbreviation"]: t["id"]
                for t in teams
                if t.get("abbreviation") and t.get("active", False)
            }
    except Exception as exc:
        logger.warning("MLB live stats: failed to load team IDs: %s", exc)
    return _team_id_map


def _fetch_team_game_log(
    team_id: int, group: str, season: int
) -> list[dict[str, Any]]:
    """Return per-game stat splits for hitting or pitching."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{_BASE}/teams/{team_id}/stats",
                params={"stats": "gameLog", "group": group, "season": season, "gameType": "R"},
            )
            resp.raise_for_status()
            stats = resp.json().get("stats", [])
            if stats:
                return stats[0].get("splits", [])
    except Exception as exc:
        logger.warning("MLB live stats: %s fetch failed for team %d: %s", group, team_id, exc)
    return []


def _splits_to_df(hitting_splits: list[dict], pitching_splits: list[dict]) -> pd.DataFrame:
    """Convert game log splits to per-game DataFrame with all MLB stat keys."""
    if not hitting_splits:
        return pd.DataFrame()

    # Use last 20 games
    hitting = hitting_splits[-20:]
    pitching_by_date = {s["date"]: s["stat"] for s in pitching_splits}

    rows = []
    for h in hitting:
        date = h["date"]
        hs = h["stat"]
        ps = pitching_by_date.get(date, {})

        # Hitting
        fga = max(1, int(hs.get("atBats", 0) or 0))
        h_hits = int(hs.get("hits", 0) or 0)
        h_walks = int(hs.get("baseOnBalls", 0) or 0)
        h_k = int(hs.get("strikeOuts", 0) or 0)
        h_hr = int(hs.get("homeRuns", 0) or 0)
        h_runs = int(hs.get("runs", 0) or 0)
        h_obp = float(hs.get("obp") or 0)
        h_slg = float(hs.get("slg") or 0)

        # Pitching
        p_runs = int(ps.get("runs", 0) or 0)
        p_go = int(ps.get("groundOuts", 0) or 0)
        p_ao = int(ps.get("airOuts", 0) or 0)
        p_k = int(ps.get("strikeOuts", 0) or 0)
        p_bb = int(ps.get("baseOnBalls", 0) or 0)
        p_hits = int(ps.get("hits", 0) or 0)
        p_ip_str = str(ps.get("inningsPitched") or "0")
        try:
            p_ip = float(p_ip_str)
        except ValueError:
            p_ip = 0.0

        # Derived
        batavg = h_hits / fga
        ops = h_obp + h_slg
        safe_ip = max(0.33, p_ip)
        era = (p_runs * 9) / safe_ip
        whip = (p_bb + p_hits) / safe_ip
        kbb = p_k / max(1, p_bb)
        total_balls = max(1, p_go + p_ao)
        gb_rate = p_go / total_balls
        fb_rate = p_ao / total_balls

        rows.append({
            "game_date": date,
            "runsScored": h_runs,
            "runsAllowed": p_runs,
            "hits": h_hits,
            "errors": 0.0,  # not in game log endpoint; leave at 0
            "walks": h_walks,
            "strikeouts": p_k,      # pitcher strikeouts (more signal than batter K)
            "era": era,
            "whip": whip,
            "battingAvg": batavg,
            "onBasePct": h_obp,
            "sluggingPct": h_slg,
            "ops": ops,
            "homeRuns": h_hr,
            "kBbRatio": kbb,
            "groundBallRate": gb_rate,
            "flyBallRate": fb_rate,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _current_season() -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    # MLB season: April–November. Before April → use previous year's data.
    return now.year if now.month >= 4 else now.year - 1


def _team_to_features(df: pd.DataFrame, stat_cols: list[str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    n = len(df)
    for stat in stat_cols:
        if stat not in df.columns:
            continue
        series = df[stat].fillna(0.0)
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


def _fetch_team(abbr: str, season: int) -> tuple[str, pd.DataFrame]:
    tid_map = _load_team_ids()
    team_id = tid_map.get(abbr.upper())
    if team_id is None:
        logger.warning("MLB live stats: unknown abbreviation %s", abbr)
        return abbr, pd.DataFrame()

    hitting = _fetch_team_game_log(team_id, "hitting", season)
    pitching = _fetch_team_game_log(team_id, "pitching", season)
    df = _splits_to_df(hitting, pitching)
    return abbr, df


def _fetch_teams_needed(abbrs: list[str], season: int) -> dict[str, dict[str, float]]:
    from proedge.pipeline.ingestion.stats import STAT_KEYS
    stat_cols = STAT_KEYS["mlb"]

    team_features: dict[str, dict[str, float]] = {}
    for i, abbr in enumerate(abbrs):
        if i > 0:
            time.sleep(0.2)
        abbr_out, df = _fetch_team(abbr, season)
        if df.empty:
            logger.warning("MLB live stats: no data for %s", abbr)
            team_features[abbr] = {}
        else:
            team_features[abbr] = _team_to_features(df, stat_cols)
            logger.debug("MLB live stats: %s — %d games, runsScored_roll5=%.1f",
                         abbr, len(df), team_features[abbr].get("runsScored_roll5_mean", 0))

    return team_features


def get_teams(abbrs: list[str]) -> dict[str, dict[str, float]]:
    """Return live rolling stats for the specified teams, using cache where valid."""
    global _cache_ts, _cache_data

    now = time.time()
    cache_valid = now - _cache_ts < _CACHE_TTL
    season = _current_season()

    missing = [a for a in abbrs if not cache_valid or a not in _cache_data or not _cache_data[a]]

    if missing:
        logger.info("MLB live stats: fetching %d teams (%s, season=%d)",
                    len(missing), ", ".join(missing), season)
        fresh = _fetch_teams_needed(missing, season)
        if cache_valid:
            _cache_data.update(fresh)
        else:
            _cache_data = fresh
            _cache_ts = now
    else:
        logger.info("MLB live stats: all %d teams in cache", len(abbrs))

    return {a: _cache_data.get(a, {}) for a in abbrs}


def inject_team_features(
    row: dict,
    team_abbr: str,
    prefix: str,
    cache: dict[str, dict[str, float]],
) -> None:
    """Overwrite median values with live rolling stats for a team. prefix = 'home_' or 'away_'."""
    team_data = cache.get(team_abbr.upper(), {})
    if not team_data:
        return
    for stat_key, value in team_data.items():
        full_key = f"{prefix}{stat_key}"
        if full_key in row:
            row[full_key] = value
