"""Real bookmaker lines from The Odds API — totals and spreads."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.the-odds-api.com/v4"

# Maps ProEdge sport keys → The Odds API sport keys
_SPORT_KEY_MAP: dict[str, str] = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
}


@dataclass
class GameOdds:
    game_id: str  # The Odds API event ID
    sport: str
    home_team: str  # full name e.g. "Los Angeles Lakers"
    away_team: str
    commence_time: datetime
    total_line: float | None  # consensus over/under (median across bookmakers)
    spread: float | None  # home team spread (negative = favored)
    home_ml: int | None  # home moneyline
    away_ml: int | None
    bookmaker_count: int
    sources: list[str] = field(default_factory=list)  # bookmaker keys used


class OddsFetcher:
    """
    Fetches current and upcoming game odds from The Odds API.

    Free tier: 500 requests/month.  The API key is read from settings
    (``settings.odds_api_key``), but can also be passed directly to __init__.
    """

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_game_odds(self, sport: str) -> list[GameOdds]:
        """Fetch all current/upcoming game odds for a sport.

        Returns an empty list if the API key is missing, invalid, or if the
        request is rate-limited (HTTP 429).
        """
        if not self.api_key:
            logger.warning("OddsFetcher: api_key is empty — skipping fetch for %s", sport)
            return []

        sport_key = self._sport_key(sport)
        url = f"{_BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "totals,spreads,h2h",
            "oddsFormat": "american",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, params=params)
        except httpx.RequestError as exc:
            logger.warning("OddsFetcher: network error fetching %s odds: %s", sport, exc)
            return []

        if resp.status_code == 401:
            logger.warning("OddsFetcher: invalid API key (401) — check settings.odds_api_key")
            return []
        if resp.status_code == 429:
            logger.warning(
                "OddsFetcher: rate-limited (429) — monthly quota likely exhausted for %s",
                sport,
            )
            return []
        if resp.status_code != 200:
            logger.warning(
                "OddsFetcher: unexpected HTTP %d for %s — %s",
                resp.status_code,
                sport,
                resp.text[:200],
            )
            return []

        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info(
            "OddsFetcher: fetched %s odds | requests remaining: %s", sport.upper(), remaining
        )

        events: list[dict[str, Any]] = resp.json()
        return [self._parse_event(event, sport) for event in events]

    def get_total_line(self, sport: str, home_team: str, away_team: str) -> float | None:
        """Best-effort lookup: return consensus total for a specific matchup.

        Matches teams by partial, case-insensitive substring comparison so
        short names ("Lakers") match full names ("Los Angeles Lakers").
        """
        games = self.fetch_game_odds(sport)
        home_lower = home_team.lower()
        away_lower = away_team.lower()

        for game in games:
            ht = game.home_team.lower()
            at = game.away_team.lower()
            if (home_lower in ht or ht in home_lower) and (away_lower in at or at in away_lower):
                return game.total_line

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sport_key(self, sport: str) -> str:
        """Translate a ProEdge sport string to the Odds API sport key."""
        key = _SPORT_KEY_MAP.get(sport.lower())
        if key is None:
            raise ValueError(f"Unknown sport '{sport}'. Supported: {list(_SPORT_KEY_MAP)}")
        return key

    def _parse_event(self, event: dict[str, Any], sport: str) -> GameOdds:
        """Parse a single event dict from the Odds API response."""
        game_id: str = event.get("id", "")
        home_team: str = event.get("home_team", "")
        away_team: str = event.get("away_team", "")

        commence_raw: str = event.get("commence_time", "")
        try:
            commence_time = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            commence_time = datetime.now(timezone.utc)

        bookmakers: list[dict[str, Any]] = event.get("bookmakers", [])

        totals: list[float] = []
        spreads: list[float] = []
        home_mls: list[int] = []
        away_mls: list[int] = []
        sources: list[str] = []

        for bm in bookmakers:
            bm_key: str = bm.get("key", "")
            sources.append(bm_key)
            for market in bm.get("markets", []):
                mkt_key: str = market.get("key", "")
                outcomes: list[dict[str, Any]] = market.get("outcomes", [])

                if mkt_key == "totals":
                    over = next((o for o in outcomes if o.get("name") == "Over"), None)
                    if over and over.get("point") is not None:
                        try:
                            totals.append(float(over["point"]))
                        except (TypeError, ValueError):
                            pass

                elif mkt_key == "spreads":
                    home_spread = next(
                        (o for o in outcomes if o.get("name", "").lower() == home_team.lower()),
                        None,
                    )
                    if home_spread and home_spread.get("point") is not None:
                        try:
                            spreads.append(float(home_spread["point"]))
                        except (TypeError, ValueError):
                            pass

                elif mkt_key == "h2h":
                    for outcome in outcomes:
                        name: str = outcome.get("name", "")
                        price_raw = outcome.get("price")
                        if price_raw is None:
                            continue
                        try:
                            price = int(price_raw)
                        except (TypeError, ValueError):
                            continue
                        if name.lower() == home_team.lower():
                            home_mls.append(price)
                        elif name.lower() == away_team.lower():
                            away_mls.append(price)

        # Consensus = median (more robust than mean against outliers)
        total_line: float | None = statistics.median(totals) if totals else None
        spread: float | None = statistics.median(spreads) if spreads else None
        home_ml: int | None = int(statistics.median(home_mls)) if home_mls else None
        away_ml: int | None = int(statistics.median(away_mls)) if away_mls else None

        return GameOdds(
            game_id=game_id,
            sport=sport.lower(),
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            total_line=total_line,
            spread=spread,
            home_ml=home_ml,
            away_ml=away_ml,
            bookmaker_count=len(bookmakers),
            sources=sources,
        )


# ---------------------------------------------------------------------------
# Async convenience function
# ---------------------------------------------------------------------------


async def fetch_game_odds_async(sport: str, api_key: str) -> list[GameOdds]:
    """Async variant of OddsFetcher.fetch_game_odds.

    Returns an empty list if the API key is missing, invalid, or rate-limited.
    """
    if not api_key:
        logger.warning("fetch_game_odds_async: api_key is empty — skipping fetch for %s", sport)
        return []

    sport_key = _SPORT_KEY_MAP.get(sport.lower())
    if sport_key is None:
        raise ValueError(f"Unknown sport '{sport}'. Supported: {list(_SPORT_KEY_MAP)}")

    url = f"{_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals,spreads,h2h",
        "oddsFormat": "american",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
    except httpx.RequestError as exc:
        logger.warning("fetch_game_odds_async: network error fetching %s odds: %s", sport, exc)
        return []

    if resp.status_code == 401:
        logger.warning("fetch_game_odds_async: invalid API key (401) — check settings.odds_api_key")
        return []
    if resp.status_code == 429:
        logger.warning(
            "fetch_game_odds_async: rate-limited (429) — monthly quota likely exhausted for %s",
            sport,
        )
        return []
    if resp.status_code != 200:
        logger.warning(
            "fetch_game_odds_async: unexpected HTTP %d for %s",
            resp.status_code,
            sport,
        )
        return []

    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info(
        "fetch_game_odds_async: fetched %s odds | requests remaining: %s",
        sport.upper(),
        remaining,
    )

    fetcher = OddsFetcher(api_key=api_key)
    events: list[dict[str, Any]] = resp.json()
    return [fetcher._parse_event(event, sport) for event in events]
