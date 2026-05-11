"""Injury data: derive from completed box scores and fetch current game-day status."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from proedge.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Player comment substrings that indicate a genuine injury absence
# (vs "DNP - Coach's Decision" which is a tactical choice)
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
        "REST",  # load management = functionally absent
    }
)
_NON_INJURY = frozenset({"COACH'S DECISION", "COACHES DECISION"})

# ESPN team roster and teams endpoints
_ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/teams/{team_id}/roster"
_ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/teams"
_ESPN_SPORT_PATHS = {"nba": "basketball/nba", "nfl": "football/nfl", "mlb": "baseball/mlb"}

# ESPN status names that indicate a meaningful absence
_INJURY_STATUSES = frozenset(
    {
        "Out",
        "Injured Reserve",
        "Questionable",
        "Doubtful",
        "Day-To-Day",
        "PUP",
        "Non-Football Injury",
        "Physically Unable to Perform",
        "Injured List",
        "10-Day IL",
        "15-Day IL",
        "60-Day IL",
    }
)
# Statuses we count as "key player out" in the model feature
_KEY_IMPACT_STATUSES = frozenset(
    {
        "Out",
        "Injured Reserve",
        "Injured List",
        "10-Day IL",
        "15-Day IL",
        "60-Day IL",
        "Doubtful",
    }
)

# Legacy impact scoring (kept for backward compat with old InjuryIngester callers)
STATUS_IMPACT: dict[str, float] = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.40,
    "probable": 0.10,
    "active": 0.0,
}
POSITION_IMPACT: dict[str, dict[str, float]] = {
    "nfl": {
        "QB": 0.90,
        "WR": 0.45,
        "RB": 0.40,
        "TE": 0.35,
        "OL": 0.30,
        "CB": 0.35,
        "DE": 0.40,
        "LB": 0.30,
        "S": 0.25,
    },
    "nba": {
        "PG": 0.75,
        "SG": 0.65,
        "SF": 0.70,
        "PF": 0.65,
        "C": 0.60,
    },
    "mlb": {
        "SP": 0.80,
        "RP": 0.30,
        "C": 0.50,
        "1B": 0.45,
        "2B": 0.50,
        "3B": 0.55,
        "SS": 0.60,
        "OF": 0.45,
    },
}


@dataclass
class InjuredPlayer:
    name: str
    team: str
    status: str
    is_key: bool  # Out/IR/Doubtful = key impact; Questionable = minor
    comment: str = ""


@dataclass
class TeamInjuryReport:
    team: str
    sport: str
    injured: list[InjuredPlayer] = field(default_factory=list)

    @property
    def key_players_out(self) -> int:
        """Count used directly as model feature `home_key_players_out`."""
        return sum(1 for p in self.injured if p.is_key)

    @property
    def total_players_out(self) -> int:
        return len(self.injured)


# ── Derive injuries from a completed box score ────────────────────────────────


def injuries_from_boxscore(
    team_players: list[dict[str, Any]],
    team_abbr: str,
    sport: str = "nba",
) -> TeamInjuryReport:
    """
    Parse player `comment` fields from a BoxScoreTraditionalV3 response.
    Distinguishes genuine injury/illness absences from coach's-decision DNPs.

    Example comments:
      "DNP - Injury/Illness"       → counted (key=True)
      "DNP - Rest"                 → counted (key=True, load management)
      "DNP - Coach's Decision"     → NOT counted
      "DND - Left Knee Soreness"   → counted (key=True)
    """
    report = TeamInjuryReport(team=team_abbr, sport=sport)
    for player in team_players:
        comment: str = (player.get("comment") or "").strip()
        if not comment:
            continue
        comment_upper = comment.upper()
        # Explicitly skip coach's-decision DNPs
        if any(kw in comment_upper for kw in _NON_INJURY):
            continue
        # Count any remaining DNP/DND with injury keywords
        if any(kw in comment_upper for kw in _INJURY_KEYWORDS):
            is_key = any(kw in comment_upper for kw in ("DND", "INJURY", "ILL", "REST"))
            name = (
                f"{player.get('firstName', '')} {player.get('familyName', '')}".strip()
                or player.get("nameI", "Unknown")
            )
            report.injured.append(
                InjuredPlayer(
                    name=name,
                    team=team_abbr,
                    status="DNP",
                    is_key=is_key,
                    comment=comment,
                )
            )
    return report


# ── Fetch current game-day status from ESPN ──────────────────────────────────


class InjuryFetcher:
    """
    Fetches today's roster injury status for all teams from ESPN.
    Returns key_players_out counts suitable for passing to the prediction API.
    Falls back to 0 gracefully when ESPN returns no data.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._team_id_cache: dict[str, dict[str, str]] = {}

    def fetch_all(self, sport: str) -> dict[str, TeamInjuryReport]:
        """Return {team_abbr: TeamInjuryReport} for all active teams."""
        path = _ESPN_SPORT_PATHS.get(sport.lower())
        if path is None:
            return {}

        team_ids = self._get_team_ids(sport, path)
        reports: dict[str, TeamInjuryReport] = {}

        with httpx.Client(timeout=self.timeout) as client:
            for abbr, tid in team_ids.items():
                try:
                    url = _ESPN_ROSTER_URL.format(path=path, team_id=tid)
                    resp = client.get(url)
                    if resp.status_code != 200:
                        continue
                    report = TeamInjuryReport(team=abbr, sport=sport)
                    for group in resp.json().get("athletes", []):
                        items = group if isinstance(group, list) else [group]
                        for player in items:
                            status_name = player.get("status", {}).get("name", "Active")
                            if status_name in _INJURY_STATUSES:
                                is_key = status_name in _KEY_IMPACT_STATUSES
                                report.injured.append(
                                    InjuredPlayer(
                                        name=player.get("displayName", "Unknown"),
                                        team=abbr,
                                        status=status_name,
                                        is_key=is_key,
                                    )
                                )
                    reports[abbr] = report
                    time.sleep(0.05)
                except Exception as exc:
                    logger.debug("ESPN roster fetch failed for %s %s: %s", sport, abbr, exc)

        total_out = sum(r.key_players_out for r in reports.values())
        logger.info(
            "ESPN injury report %s: %d teams checked, %d key players out",
            sport.upper(),
            len(reports),
            total_out,
        )
        return reports

    def key_players_out(self, sport: str, team: str) -> int:
        """Best-effort: key players unavailable for a single team right now."""
        try:
            return (
                self.fetch_all(sport)
                .get(team.upper(), TeamInjuryReport(team, sport))
                .key_players_out
            )
        except Exception as exc:
            logger.warning("Injury fetch failed for %s %s: %s", sport, team, exc)
            return 0

    def _get_team_ids(self, sport: str, path: str) -> dict[str, str]:
        if sport in self._team_id_cache:
            return self._team_id_cache[sport]
        try:
            resp = httpx.get(_ESPN_TEAMS_URL.format(path=path), timeout=self.timeout)
            teams_raw = resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            mapping = {t["team"]["abbreviation"]: t["team"]["id"] for t in teams_raw if "team" in t}
            self._team_id_cache[sport] = mapping
            return mapping
        except Exception as exc:
            logger.warning("ESPN team list failed for %s: %s", sport, exc)
            return {}


# ── Legacy InjuryIngester (backward compat) ───────────────────────────────────


class InjuryIngester:
    """Kept for backward compatibility — wraps InjuryFetcher."""

    def compute_team_injury_impact(self, team_id: str, injury_reports: list[dict]) -> float:
        total = sum(
            r.get("impact_score", 0.0) for r in injury_reports if r.get("team_id") == team_id
        )
        return min(total, 1.0)
