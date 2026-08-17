"""Underdog Fantasy over/under line fetcher.

Fetches game total lines from the public Underdog Fantasy API.
Game totals appear as standalone lines (appearance_id=None in options) backed by
Kalshi or Nadex. Each line has:
  - over_under.title: "AWAY @ HOME Total Points O/U"  ← primary match key
  - options[].selection_header: "AWAY @ HOME"          ← fallback
  - options[].american_price: "-113" / "-105"

Note: Game totals are backed by Nadex (or Kalshi) and appear in the v3 endpoint only
when the prediction market is open — typically 1-4 hours before game time for MLB
(regular season) and up to 24+ hours for NBA playoffs. Fetching earlier will return
empty; this is expected and not an error. The v5 endpoint has player props only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.underdogfantasy.com/beta/v3/over_under_lines"

# Minimum stat_value that signals a game total (vs player prop)
_GAME_TOTAL_MIN = {"nba": 150.0, "nfl": 25.0, "mlb": 4.0}

# Keywords in over_under.title that identify a game total (not a player prop)
_TOTAL_TITLE_KEYWORDS = {"Total Points", "Total Runs", "Total Goals", "Game Total"}

# Kalshi uses some non-standard MLB team abbreviations that differ from what
# Underdog uses in their titles. Map Kalshi codes → Underdog-style codes.
_TEAM_ALIASES: dict[str, list[str]] = {
    # Kalshi → [possible Underdog variants]
    "LA":  ["LAD", "LA"],        # Dodgers: Kalshi="LA", Underdog="LAD"
    "LAD": ["LAD", "LA"],
    "DSD": ["SD", "SDP", "DSD"], # Padres: Kalshi="DSD", Underdog="SD"
    "SD":  ["SD", "SDP", "DSD"],
    "ATH": ["ATH", "OAK", "SAC"],# Athletics: Kalshi="ATH"
    "OAK": ["OAK", "ATH", "SAC"],
    "KC":  ["KC", "KCR"],        # Royals vs Chiefs (context-dependent)
    "KCR": ["KCR", "KC"],
    "AZ":  ["AZ", "ARI"],        # Diamondbacks
    "ARI": ["ARI", "AZ"],
    "TB":  ["TB", "TBR"],        # Rays
    "TBR": ["TBR", "TB"],
    "CWS": ["CWS", "CHW"],       # White Sox
    "CHW": ["CHW", "CWS"],
    "SF":  ["SF", "SFG"],        # Giants
    "SFG": ["SFG", "SF"],
    "WSH": ["WSH", "WAS"],       # Nationals
    "WAS": ["WAS", "WSH"],
}


@dataclass
class UnderdogGameTotal:
    sport: str
    stat_value: float
    over_american: str | None    # e.g. "-139"
    under_american: str | None   # e.g. "+113"
    over_payout: float | None    # decimal multiplier
    under_payout: float | None
    line_type: str               # "balanced" or "alternate"
    contract_url: str | None
    selection_header: str | None  # "AWAY @ HOME" from option
    ou_title: str | None          # "AWAY @ HOME Total Points O/U" from over_under.title
    provider_id: str | None       # "kalshi", "nadex", etc.


def fetch_game_totals(sport: str, timeout: float = 10.0) -> list[UnderdogGameTotal]:
    """Fetch all game-level over/under totals for a sport from Underdog Fantasy.

    Uses v3 endpoint (game totals only). MLB game totals are not currently exposed
    by Underdog's public API — this will return empty for MLB until they add support.
    """
    sport_lower = sport.lower()
    min_val = _GAME_TOTAL_MIN.get(sport_lower, 20.0)

    # v3 returns only current game totals; v4/v5 have more lines but no MLB totals
    url = _BASE_URL

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Underdog fetch error: %s", exc)
        return []

    data = resp.json()
    lines = data.get("over_under_lines", [])

    results: list[UnderdogGameTotal] = []
    for line in lines:
        val = line.get("stat_value")
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        if val < min_val:
            continue

        # Game totals have appearance_id=None in all options
        opts = line.get("options", [])
        if not opts or opts[0].get("appearance_id") is not None:
            continue

        # Confirm this is a game total (not a standalone player season prop)
        ou = line.get("over_under", {})
        ou_title = ou.get("title", "")
        if not any(kw in ou_title for kw in _TOTAL_TITLE_KEYWORDS):
            continue

        # Apply sport range sanity check
        if not _in_range_for_sport(val, sport_lower):
            continue

        selection_header = opts[0].get("selection_header")
        over_opt = next((o for o in opts if o.get("choice") == "higher"), None)
        under_opt = next((o for o in opts if o.get("choice") == "lower"), None)

        results.append(UnderdogGameTotal(
            sport=sport_lower,
            stat_value=val,
            over_american=over_opt.get("american_price") if over_opt else None,
            under_american=under_opt.get("american_price") if under_opt else None,
            over_payout=float(over_opt.get("payout_multiplier", 1)) if over_opt else None,
            under_payout=float(under_opt.get("payout_multiplier", 1)) if under_opt else None,
            line_type=line.get("line_type", "balanced"),
            contract_url=line.get("contract_url") or line.get("contract_terms_url"),
            selection_header=selection_header,
            ou_title=ou_title,
            provider_id=line.get("provider_id"),
        ))

    logger.info("Underdog: %d %s game total lines found — %s",
                len(results), sport.upper(),
                [r.ou_title for r in results])
    return sorted(results, key=lambda x: x.stat_value)


def find_game_totals(
    sport: str,
    home_team: str,
    away_team: str,
    timeout: float = 10.0,
) -> list[UnderdogGameTotal]:
    """Find Underdog game total lines for a specific matchup.

    Matching uses over_under.title ("AWAY @ HOME Total Points O/U") as the primary
    key since it reliably contains both team codes. Falls back to selection_header.

    Returns empty list if no game total is listed — this is expected for some games
    even when a total appears on the Underdog website (served via internal API).
    """
    totals = fetch_game_totals(sport, timeout=timeout)
    if not totals:
        return []

    home_up = home_team.upper().strip()
    away_up = away_team.upper().strip()

    # Build candidate codes for each team (Kalshi may use non-standard abbreviations)
    home_candidates = [home_up] + _TEAM_ALIASES.get(home_up, [])
    away_candidates = [away_up] + _TEAM_ALIASES.get(away_up, [])

    matched: list[UnderdogGameTotal] = []
    for t in totals:
        # Primary: ou_title contains both team codes
        title = (t.ou_title or "").upper()
        header = (t.selection_header or "").upper()
        search_str = title + " " + header

        home_hit = any(
            c in search_str or _abbrev_match(c, search_str)
            for c in home_candidates
        )
        away_hit = any(
            c in search_str or _abbrev_match(c, search_str)
            for c in away_candidates
        )
        if home_hit and away_hit:
            matched.append(t)

    available = [t.ou_title for t in totals]
    if matched:
        logger.info("Underdog: matched %s vs %s (%s) → %s",
                    home_team, away_team, sport.upper(), matched[0].ou_title)
    else:
        logger.info("Underdog: no game total for %s vs %s (%s). Available: %s",
                    home_team, away_team, sport.upper(), available)

    return matched


def find_game_total(
    sport: str,
    home_team: str,
    away_team: str,
    kalshi_implied_line: float | None = None,
    timeout: float = 10.0,
) -> UnderdogGameTotal | None:
    """Return the primary (balanced) Underdog game total for a matchup, or None."""
    lines = find_game_totals(sport, home_team, away_team, timeout=timeout)
    if not lines:
        return None
    balanced = [l for l in lines if l.line_type == "balanced"]
    return balanced[0] if balanced else lines[0]


def american_to_prob(american: str | None) -> float | None:
    """Convert American odds string (e.g. '-110', '+120') to implied probability."""
    if not american:
        return None
    try:
        n = float(american)
        if n < 0:
            return round(-n / (-n + 100), 4)
        else:
            return round(100 / (n + 100), 4)
    except (ValueError, TypeError):
        return None


def _abbrev_match(code: str, text: str) -> bool:
    """Check if team code (first 3 chars) appears as a word in text."""
    abbrev = code[:3]
    return bool(re.search(r'\b' + re.escape(abbrev) + r'\b', text))


def _in_range_for_sport(val: float, sport: str) -> bool:
    """Is this value plausible as a game total for this sport?"""
    ranges = {"nba": (150, 300), "nfl": (25, 85), "mlb": (4, 25)}
    lo, hi = ranges.get(sport, (0, 9999))
    return lo <= val <= hi
