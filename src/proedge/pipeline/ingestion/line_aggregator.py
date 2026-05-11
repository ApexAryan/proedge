"""Multi-source line aggregator.

Fetches game totals from The Odds API (sportsbooks), PrizePicks, and Kalshi
in parallel and merges them into a LineComparison for each matchup.

Line discrepancy signals:
  pp_vs_book   — PrizePicks line minus book consensus. Positive = PP projects
                 a higher-scoring game than the books.
  kalshi_vs_book — Kalshi market-implied line minus book consensus. The
                 prediction market is often sharper than PrizePicks because
                 real money backs every position.

These discrepancies can be used as features after the next retrain,
or read directly from /lines/compare to inform manual picks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_CACHE_TTL = 900  # 15 minutes — shared across all sources to conserve Odds API credits


@dataclass
class LineComparison:
    sport: str
    home_team: str
    away_team: str
    book_line: float | None = None        # The Odds API consensus (median across bookmakers)
    prizepicks_line: float | None = None  # PrizePicks posted game total
    kalshi_line: float | None = None      # Kalshi prediction market implied line (50% threshold)
    kalshi_nearest_threshold: int | None = None
    kalshi_nearest_prob: float | None = None   # yes probability at nearest threshold
    consensus_line: float | None = None         # book → kalshi → prizepicks priority
    pp_vs_book: float | None = None             # prizepicks_line - book_line
    kalshi_vs_book: float | None = None         # kalshi_line - book_line
    sources: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Per-sport cache: sport → (timestamp, list[LineComparison])
_cache: dict[str, tuple[float, list[LineComparison]]] = {}


async def get_line_comparison(
    sport: str,
    home_team: str,
    away_team: str,
    odds_api_key: str = "",
) -> LineComparison:
    """Return a LineComparison for a specific matchup. Uses cached board if fresh."""
    board = await _get_board(sport, odds_api_key)

    home_up = home_team.upper()
    away_up = away_team.upper()

    for comp in board:
        ht = comp.home_team.upper()
        at = comp.away_team.upper()
        if (home_up in ht or ht in home_up) and (away_up in at or at in away_up):
            return comp

    # No match — return empty shell so callers always get a typed object
    return LineComparison(sport=sport, home_team=home_team, away_team=away_team)


async def get_full_board(sport: str, odds_api_key: str = "") -> list[LineComparison]:
    """Return all LineComparisons currently on the board for a sport."""
    return await _get_board(sport, odds_api_key)


async def _get_board(sport: str, odds_api_key: str) -> list[LineComparison]:
    now = time.monotonic()
    cached = _cache.get(sport)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    board = await _build_board(sport, odds_api_key)
    _cache[sport] = (now, board)
    return board


async def _build_board(sport: str, odds_api_key: str) -> list[LineComparison]:
    loop = asyncio.get_running_loop()

    # Run all three sources concurrently
    results = await asyncio.gather(
        _safe(asyncio.create_task(_fetch_prizepicks(sport)), "prizepicks"),
        _safe(loop.run_in_executor(None, _fetch_kalshi, sport), "kalshi"),
        _safe(
            loop.run_in_executor(None, _fetch_books, sport, odds_api_key),
            "books",
        ),
        return_exceptions=False,
    )

    pp_games: list[dict] = results[0]
    kalshi_games: list[dict] = results[1]
    book_games: list[dict] = results[2]

    # Index book and Kalshi lines by team-pair key for fast lookup
    book_index = _index_by_teams(book_games)
    kalshi_index = _index_by_teams(kalshi_games)

    # Build comparisons anchored on PrizePicks games (widest coverage of upcoming games)
    # then fall back to Kalshi-only games
    seen: set[str] = set()
    comparisons: list[LineComparison] = []

    all_anchors = pp_games + [g for g in kalshi_games if not _find_in_index(book_index, g["home"], g["away"])]

    for g in all_anchors:
        key = _team_key(g["home"], g["away"])
        if key in seen:
            continue
        seen.add(key)

        comp = _build_comparison(
            sport=sport,
            home_team=g["home"],
            away_team=g["away"],
            pp_line=g.get("total"),
            kalshi=_find_in_index(kalshi_index, g["home"], g["away"]),
            book=_find_in_index(book_index, g["home"], g["away"]),
        )
        comparisons.append(comp)

    # Add book-only games not already covered
    for g in book_games:
        key = _team_key(g["home"], g["away"])
        if key in seen:
            continue
        seen.add(key)
        comp = _build_comparison(
            sport=sport,
            home_team=g["home"],
            away_team=g["away"],
            pp_line=None,
            kalshi=_find_in_index(kalshi_index, g["home"], g["away"]),
            book=g,
        )
        comparisons.append(comp)

    logger.info(
        "LineAggregator %s: %d games | pp=%d book=%d kalshi=%d",
        sport.upper(),
        len(comparisons),
        len(pp_games),
        len(book_games),
        len(kalshi_games),
    )
    return comparisons


def _build_comparison(
    *,
    sport: str,
    home_team: str,
    away_team: str,
    pp_line: float | None,
    kalshi: dict | None,
    book: dict | None,
) -> LineComparison:
    book_line = book.get("total") if book else None
    kalshi_line = kalshi.get("implied_line") if kalshi else None

    # Consensus: prefer sharper sources — books > kalshi > prizepicks
    consensus = book_line if book_line is not None else (
        kalshi_line if kalshi_line is not None else pp_line
    )

    sources = []
    if book_line is not None:
        sources.append("odds_api")
    if pp_line is not None:
        sources.append("prizepicks")
    if kalshi_line is not None:
        sources.append("kalshi")

    pp_vs_book = (
        round(pp_line - book_line, 2)
        if pp_line is not None and book_line is not None
        else None
    )
    kalshi_vs_book = (
        round(kalshi_line - book_line, 2)
        if kalshi_line is not None and book_line is not None
        else None
    )

    return LineComparison(
        sport=sport,
        home_team=home_team,
        away_team=away_team,
        book_line=book_line,
        prizepicks_line=pp_line,
        kalshi_line=kalshi_line,
        kalshi_nearest_threshold=kalshi.get("nearest_threshold") if kalshi else None,
        kalshi_nearest_prob=kalshi.get("nearest_prob") if kalshi else None,
        consensus_line=consensus,
        pp_vs_book=pp_vs_book,
        kalshi_vs_book=kalshi_vs_book,
        sources=sources,
    )


# ── Source fetchers ───────────────────────────────────────────────────────────


async def _fetch_prizepicks(sport: str) -> list[dict]:
    try:
        from proedge.pipeline.ingestion.prizepicks_fetcher import fetch_board

        board = await asyncio.wait_for(fetch_board(sport), timeout=10.0)
        results = []
        for game in board.games:
            total = board.total_line_for(game["home_team"], game["away_team"])
            if total is not None:
                results.append({
                    "home": game["home_team"],
                    "away": game["away_team"],
                    "total": total,
                })
        return results
    except Exception as exc:
        logger.warning("PrizePicks fetch error: %s", exc)
        return []


def _fetch_kalshi(sport: str) -> list[dict]:
    try:
        from proedge.pipeline.ingestion.kalshi_fetcher import fetch_totals

        games = fetch_totals(sport)
        return [
            {
                "home": g.team1,
                "away": g.team2,
                "implied_line": g.implied_line,
                "nearest_threshold": g.nearest_threshold,
                "nearest_prob": g.nearest_prob,
            }
            for g in games
        ]
    except Exception as exc:
        logger.warning("Kalshi fetch error: %s", exc)
        return []


def _fetch_books(sport: str, api_key: str) -> list[dict]:
    if not api_key:
        return []
    try:
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher

        games = OddsFetcher(api_key=api_key, timeout=8.0).fetch_game_odds(sport)
        return [
            {
                "home": g.home_team,
                "away": g.away_team,
                "total": g.total_line,
            }
            for g in games
            if g.total_line is not None
        ]
    except Exception as exc:
        logger.warning("Odds API fetch error: %s", exc)
        return []


# ── Matching helpers ──────────────────────────────────────────────────────────


def _team_key(home: str, away: str) -> str:
    return f"{home.upper()}|{away.upper()}"


def _index_by_teams(games: list[dict]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for g in games:
        idx[_team_key(g.get("home", ""), g.get("away", ""))] = g
    return idx


def _find_in_index(idx: dict[str, dict], home: str, away: str) -> dict | None:
    home_up = home.upper()
    away_up = away.upper()
    for key, g in idx.items():
        ht, at = key.split("|")
        home_hit = home_up in ht or ht in home_up
        away_hit = away_up in at or at in away_up
        if home_hit and away_hit:
            return g
    return None


async def _safe(coro_or_future, label: str):
    try:
        return await coro_or_future
    except Exception as exc:
        logger.warning("LineAggregator source '%s' failed: %s", label, exc)
        return []
