"""Lines router — PrizePicks spreads, totals, and player projections."""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict

import httpx
from fastapi import APIRouter, HTTPException, Query, status

from proedge.api.schemas import (
    GameLineResponse,
    GameSummaryResponse,
    PlayerProjectionResponse,
    PrizePicksBoardResponse,
)
from proedge.pipeline.ingestion.prizepicks_fetcher import (
    LEAGUE_IDS,
    GameLine,
    PlayerProjection,
    PrizePicksBoard,
    fetch_board,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lines", tags=["lines"])


@router.get(
    "/prizepicks/{sport}",
    response_model=PrizePicksBoardResponse,
    summary="Fetch live PrizePicks board for a sport",
    description=(
        "Returns all player projections and game lines (spreads, totals) currently "
        "on the PrizePicks board for NBA, NFL, or MLB. Games include a `projected_total` "
        "derived from summing player point props when no explicit game total is posted."
    ),
)
async def get_prizepicks_board(
    sport: str,
    include_promos: bool = Query(False, description="Include promo/boosted lines"),
    status_filter: str = Query(
        "all", description="Filter by projection status: pre_game | locked | all"
    ),
):
    sport_lower = sport.lower()
    if sport_lower not in LEAGUE_IDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported sport '{sport}'. Supported: {list(LEAGUE_IDS.keys())}",
        )

    try:
        board = await fetch_board(sport_lower)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"PrizePicks API returned {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not reach PrizePicks API: {exc}",
        )

    return _build_response(board, include_promos=include_promos, status_filter=status_filter)


@router.get(
    "/prizepicks/{sport}/game/{home_team}/{away_team}",
    response_model=GameSummaryResponse,
    summary="Lines and props for a specific matchup",
)
async def get_prizepicks_game(sport: str, home_team: str, away_team: str):
    sport_lower = sport.lower()
    if sport_lower not in LEAGUE_IDS:
        raise HTTPException(status_code=422, detail=f"Unsupported sport '{sport}'")

    try:
        board = await fetch_board(sport_lower)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    home_up = home_team.upper()
    away_up = away_team.upper()

    # Match by abbreviation or partial name (case-insensitive)
    matched_props = [
        p for p in board.player_projections
        if home_up in p.home_team.upper() or home_up in p.away_team.upper()
        or away_up in p.home_team.upper() or away_up in p.away_team.upper()
    ]
    matched_lines = [
        gl for gl in board.game_lines
        if home_up in gl.home_team.upper() or home_up in gl.away_team.upper()
        or away_up in gl.home_team.upper() or away_up in gl.away_team.upper()
    ]

    if not matched_props and not matched_lines:
        raise HTTPException(
            status_code=404,
            detail=f"No PrizePicks lines found for {home_team} vs {away_team}",
        )

    game_id = (matched_props or matched_lines)[0].game_id
    sample = matched_props[0] if matched_props else matched_lines[0]

    return _game_summary(
        game_id=game_id,
        home_team=sample.home_team,
        away_team=sample.away_team,
        start_time=sample.start_time,
        sport=sport_lower,
        props=matched_props,
        gl_list=matched_lines,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_response(
    board: PrizePicksBoard,
    *,
    include_promos: bool,
    status_filter: str,
) -> PrizePicksBoardResponse:
    props = [
        p for p in board.player_projections
        if (include_promos or not p.is_promo)
        and (status_filter == "all" or p.status == status_filter or
             # treat "pre_game" as the normal pre-game state
             (status_filter == "pre_game" and p.status in ("pre_game", "normal")))
    ]

    # Group props and game lines by game_id
    props_by_game: dict[str, list[PlayerProjection]] = defaultdict(list)
    for p in props:
        props_by_game[p.game_id].append(p)

    lines_by_game: dict[str, list[GameLine]] = defaultdict(list)
    for gl in board.game_lines:
        lines_by_game[gl.game_id].append(gl)

    all_game_ids = set(props_by_game) | set(lines_by_game)

    game_summaries: list[GameSummaryResponse] = []
    for game_id in all_game_ids:
        game_props = props_by_game.get(game_id, [])
        game_gl = lines_by_game.get(game_id, [])

        # Infer home/away from first available record
        sample = (game_props or game_gl)[0]
        summary = _game_summary(
            game_id=game_id,
            home_team=sample.home_team,
            away_team=sample.away_team,
            start_time=sample.start_time,
            sport=board.sport,
            props=game_props,
            gl_list=game_gl,
        )
        game_summaries.append(summary)

    game_summaries.sort(key=lambda g: g.start_time or "9999")

    return PrizePicksBoardResponse(
        sport=board.sport,
        fetched_at=board.fetched_at,
        game_count=len(game_summaries),
        player_prop_count=len(props),
        game_line_count=len(board.game_lines),
        games=game_summaries,
    )


def _game_summary(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    start_time,
    sport: str,
    props: list[PlayerProjection],
    gl_list: list[GameLine],
) -> GameSummaryResponse:
    # Pull total and spread from game lines
    total_line: float | None = None
    spread: float | None = None
    for gl in gl_list:
        if gl.stat_type in ("Total Points", "Game Total Points", "Total Runs", "Total Goals"):
            total_line = gl.line
        elif gl.stat_type in ("Spread", "Run Line", "Puck Line", "Game Spread"):
            spread = gl.line

    # When no explicit game total exists, derive from player Points props.
    # Use only standard lines (not demon/goblin). Deduplicate per player by
    # taking their median line, then sum the top scorers (≥ 8 to be reliable).
    projected_total: float | None = None
    if total_line is None:
        point_stat = {"nba": "Points", "nfl": "Points", "mlb": "Runs", "nhl": "Goals"}.get(
            sport, "Points"
        )
        standard_pts = [
            p for p in props
            if p.stat_type == point_stat and p.odds_type not in ("demon", "goblin")
        ]
        if standard_pts:
            # Group by player, take median line per player
            player_lines: dict[str, list[float]] = defaultdict(list)
            for p in standard_pts:
                player_lines[p.player_name].append(p.line)
            medians = [statistics.median(v) for v in player_lines.values()]
            if len(medians) >= 6:
                projected_total = round(sum(medians), 1)

    return GameSummaryResponse(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        start_time=start_time,
        sport=sport,
        total_line=total_line,
        spread=spread,
        projected_total=projected_total,
        game_lines=[
            GameLineResponse(
                game_id=gl.game_id,
                home_team=gl.home_team,
                away_team=gl.away_team,
                start_time=gl.start_time,
                stat_type=gl.stat_type,
                line=gl.line,
                sport=sport,
            )
            for gl in gl_list
        ],
        player_projections=[
            PlayerProjectionResponse(
                projection_id=p.projection_id,
                player_name=p.player_name,
                team=p.team,
                position=p.position,
                stat_type=p.stat_type,
                line=p.line,
                game_id=p.game_id,
                home_team=p.home_team,
                away_team=p.away_team,
                start_time=p.start_time,
                status=p.status,
                is_promo=p.is_promo,
                odds_type=p.odds_type,
                projection_type=p.projection_type,
                sport=sport,
            )
            for p in props
        ],
    )
