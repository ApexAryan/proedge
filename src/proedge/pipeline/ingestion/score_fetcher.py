"""ESPN scoreboard fetcher — final game totals for auto-settling predictions.

Uses the public ESPN API (no auth required). Returns completed games with
home/away scores so predictions can be settled automatically after tip-off.
"""

from __future__ import annotations

import logging
from datetime import date

import httpx

logger = logging.getLogger(__name__)

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_SPORT_PATH = {
    "nba": "basketball/nba",
    "mlb": "baseball/mlb",
    "nfl": "football/nfl",
}

# ESPN abbreviation → our internal abbreviations (and vice versa)
_ESPN_ALIASES: dict[str, str] = {
    "GS": "GSW",   # Warriors
    "NO": "NOP",   # Pelicans
    "NY": "NYK",   # Knicks (ESPN sometimes uses NY)
    "SA": "SAS",   # Spurs
    "OKC": "OKC",
    "ATH": "OAK",  # Athletics
    "WSH": "WAS",  # Nationals
    "CWS": "CHW",  # White Sox
}


def fetch_scores(sport: str, game_date: date | str | None = None) -> list[dict]:
    """Return completed games for a sport/date from ESPN.

    Returns list of dicts:
        {home_abbr, away_abbr, home_score, away_score, total}

    Only games with status.type.completed=True are included.
    game_date: date object or 'YYYYMMDD' string; defaults to today.
    """
    sport_lower = sport.lower()
    path = _SPORT_PATH.get(sport_lower)
    if not path:
        return []

    if game_date is None:
        game_date = date.today()

    if hasattr(game_date, "strftime"):
        date_str = game_date.strftime("%Y%m%d")
    else:
        date_str = str(game_date).replace("-", "")

    url = f"{_ESPN_BASE}/{path}/scoreboard"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params={"dates": date_str})
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("ESPN score fetch failed for %s %s: %s", sport, date_str, exc)
        return []

    data = resp.json()
    results = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        if not event.get("status", {}).get("type", {}).get("completed", False):
            continue
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue
        home_abbr = _normalize(home.get("team", {}).get("abbreviation", ""))
        away_abbr = _normalize(away.get("team", {}).get("abbreviation", ""))
        if not home_abbr or not away_abbr:
            continue
        results.append({
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_score": home_score,
            "away_score": away_score,
            "total": home_score + away_score,
        })

    logger.info("ESPN: %d completed %s games on %s", len(results), sport.upper(), date_str)
    return results


def match_score(home_team: str, away_team: str, scores: list[dict]) -> dict | None:
    """Match a game to an ESPN score entry by team abbreviation (fuzzy)."""
    home_up = _normalize(home_team)
    away_up = _normalize(away_team)
    for s in scores:
        h = s["home_abbr"]
        a = s["away_abbr"]
        if _teams_match(home_up, h) and _teams_match(away_up, a):
            return s
    return None


def _normalize(abbr: str) -> str:
    abbr = abbr.strip().upper()
    return _ESPN_ALIASES.get(abbr, abbr)


def _teams_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 2 and len(b) >= 2:
        # Prefix match (first 2-3 chars) covers most alias cases
        if a[:3] == b[:3] or a[:2] == b[:2]:
            return True
    return False
