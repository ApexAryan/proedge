"""PrizePicks projections fetcher — player props, game spreads, and totals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.prizepicks.com"
_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
}

# PrizePicks internal league IDs
LEAGUE_IDS: dict[str, int] = {
    "nba": 7,
    "nfl": 9,
    "mlb": 2,
    "nhl": 6,
    "nba_g": 14,  # NBA G-League
}

# Stat types that represent game-level lines (not individual player stats)
_GAME_LEVEL_STATS = {
    "Total Points",
    "Game Total Points",
    "Spread",
    "Moneyline",
    "1st Half Total",
    "Total Runs",
    "Run Line",
    "Total Goals",
    "Puck Line",
    "Game Spread",
}


@dataclass
class PlayerProjection:
    projection_id: str
    player_name: str
    team: str
    position: str
    stat_type: str
    line: float
    game_id: str
    home_team: str
    away_team: str
    start_time: datetime | None
    status: str  # "pre_game", "locked", "disabled"
    is_promo: bool
    odds_type: str  # "standard" | "demon" (harder) | "goblin" (easier)
    projection_type: str  # stat category label e.g. "Single Stat", "Fantasy Score"
    sport: str


@dataclass
class GameLine:
    game_id: str
    home_team: str
    away_team: str
    start_time: datetime | None
    stat_type: str  # "Total Points", "Spread", etc.
    line: float
    sport: str


@dataclass
class PrizePicksBoard:
    sport: str
    fetched_at: datetime
    player_projections: list[PlayerProjection] = field(default_factory=list)
    game_lines: list[GameLine] = field(default_factory=list)

    # Derived convenience collections
    @property
    def games(self) -> list[dict]:
        """Unique games on the board, sorted by start time."""
        seen: dict[str, dict] = {}
        for p in self.player_projections:
            if p.game_id not in seen:
                seen[p.game_id] = {
                    "game_id": p.game_id,
                    "home_team": p.home_team,
                    "away_team": p.away_team,
                    "start_time": p.start_time,
                    "player_count": 0,
                }
            seen[p.game_id]["player_count"] += 1
        for gl in self.game_lines:
            if gl.game_id not in seen:
                seen[gl.game_id] = {
                    "game_id": gl.game_id,
                    "home_team": gl.home_team,
                    "away_team": gl.away_team,
                    "start_time": gl.start_time,
                    "player_count": 0,
                }
        return sorted(seen.values(), key=lambda g: g["start_time"] or datetime.max)

    def total_line_for(self, home_team: str, away_team: str) -> float | None:
        """Return the game total line if PrizePicks has one for this matchup."""
        for gl in self.game_lines:
            if gl.stat_type in (
                "Total Points",
                "Game Total Points",
                "Total Runs",
                "Total Goals",
            ) and (gl.home_team == home_team or gl.away_team == away_team):
                return gl.line
        return None

    def spread_for(self, home_team: str, away_team: str) -> float | None:
        """Return the home team spread if available."""
        for gl in self.game_lines:
            if gl.stat_type in ("Spread", "Run Line", "Puck Line", "Game Spread") and (
                gl.home_team == home_team or gl.away_team == away_team
            ):
                return gl.line
        return None


async def fetch_board(sport: str, timeout: float = 15.0) -> PrizePicksBoard:
    """Fetch all live projections for a sport from PrizePicks."""
    league_id = LEAGUE_IDS.get(sport.lower())
    if league_id is None:
        raise ValueError(f"Unknown sport '{sport}'. Supported: {list(LEAGUE_IDS)}")

    url = f"{_BASE_URL}/projections"
    params = {
        "league_id": league_id,
        "per_page": 500,
        "single_stat": "true",
    }

    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

    return _parse_board(sport.lower(), payload)


def fetch_board_sync(sport: str, timeout: float = 15.0) -> PrizePicksBoard:
    """Synchronous wrapper for CLI / notebook use."""
    league_id = LEAGUE_IDS.get(sport.lower())
    if league_id is None:
        raise ValueError(f"Unknown sport '{sport}'. Supported: {list(LEAGUE_IDS)}")

    url = f"{_BASE_URL}/projections"
    params = {
        "league_id": league_id,
        "per_page": 500,
        "single_stat": "true",
    }

    with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

    return _parse_board(sport.lower(), payload)


def _parse_board(sport: str, payload: dict[str, Any]) -> PrizePicksBoard:
    now = datetime.now(timezone.utc)
    board = PrizePicksBoard(sport=sport, fetched_at=now)

    # Build lookup maps from the `included` array (JSON:API side-loaded resources)
    players: dict[str, dict] = {}
    games: dict[str, dict] = {}

    for item in payload.get("included", []):
        item_type = item.get("type", "")
        attrs = item.get("attributes", {})
        item_id = item.get("id", "")

        if item_type == "new_player":
            players[item_id] = attrs
        elif item_type == "game":
            # Teams are nested: metadata.game_info.teams.{home,away}.abbreviation
            meta = attrs.get("metadata") or {}
            game_info = meta.get("game_info") or {}
            teams = game_info.get("teams") or {}
            home_abbrev = (teams.get("home") or {}).get("abbreviation", "")
            away_abbrev = (teams.get("away") or {}).get("abbreviation", "")
            if not home_abbrev or not away_abbrev:
                logger.warning(
                    "PrizePicks game %s missing team abbreviation(s): home=%r away=%r — "
                    "API schema may have changed",
                    item_id,
                    home_abbrev,
                    away_abbrev,
                )
            games[item_id] = {
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "start_time": attrs.get("start_time"),
                "status": attrs.get("status", ""),
                "is_live": attrs.get("is_live", False),
                "external_game_id": attrs.get("external_game_id", ""),
            }

    for proj in payload.get("data", []):
        if proj.get("type") != "projection":
            continue

        attrs = proj.get("attributes", {})
        rels = proj.get("relationships", {})
        proj_id = proj.get("id", "")

        stat_type: str = attrs.get("stat_type") or attrs.get("stat_display_name") or ""
        line_raw = attrs.get("line_score", attrs.get("line", 0))
        try:
            line = float(line_raw)
        except (TypeError, ValueError):
            continue

        status: str = attrs.get("status", "pre_game")
        is_promo: bool = bool(attrs.get("is_promo", False))
        # odds_type: "standard" | "demon" (harder target) | "goblin" (easier target)
        # projection_type: stat category label (e.g. "Single Stat", "Fantasy Score")
        odds_type: str = attrs.get("odds_type") or "standard"
        proj_type: str = attrs.get("projection_type") or "standard"
        event_type: str = attrs.get("event_type", "player")  # "player" | "team" | "game"
        start_raw = attrs.get("start_time") or attrs.get("board_time")
        start_time = _parse_dt(start_raw)

        # Resolve related game (relationship key is "game" in the actual API)
        game_rel = rels.get("game") or rels.get("new_game") or {}
        game_ref = game_rel.get("data") or {}
        game_id = str(game_ref.get("id", ""))
        game_attrs = games.get(game_id, {})

        home_team = game_attrs.get("home_team", attrs.get("description", ""))
        away_team = game_attrs.get("away_team", "")
        if not start_time:
            start_time = _parse_dt(game_attrs.get("start_time"))

        # Game-level lines: either explicit stat type or event_type == "game"
        is_game_line = stat_type in _GAME_LEVEL_STATS or event_type == "game"

        if is_game_line:
            board.game_lines.append(
                GameLine(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    start_time=start_time,
                    stat_type=stat_type,
                    line=line,
                    sport=sport,
                )
            )
        else:
            # Resolve player — event_type "team" means team-level prop (e.g. team total)
            player_rel = rels.get("new_player") or rels.get("player") or {}
            player_ref = player_rel.get("data") or {}
            player_id = str(player_ref.get("id", ""))
            player_attrs = players.get(player_id, {})

            player_name = (
                player_attrs.get("display_name")
                or player_attrs.get("name")
                or attrs.get("description")
                or "Unknown"
            )
            team = _abbrev(
                player_attrs.get("team")
                or (attrs.get("description") if event_type == "team" else "")
                or ""
            )
            position = player_attrs.get("position", "")

            board.player_projections.append(
                PlayerProjection(
                    projection_id=proj_id,
                    player_name=player_name,
                    team=team,
                    position=position,
                    stat_type=stat_type,
                    line=line,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    start_time=start_time,
                    status=status,
                    is_promo=is_promo,
                    odds_type=odds_type,
                    projection_type=proj_type,
                    sport=sport,
                )
            )

    logger.info(
        "PrizePicks %s: %d player props, %d game lines across %d games",
        sport.upper(),
        len(board.player_projections),
        len(board.game_lines),
        len(board.games),
    )
    return board


def _parse_dt(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _abbrev(name: str) -> str:
    """Best-effort: return as-is if already short, else keep as full name."""
    return (name or "").strip()
