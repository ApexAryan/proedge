"""Lines router — PrizePicks spreads, totals, player projections, and cross-source comparison."""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections import defaultdict

import httpx
from fastapi import APIRouter, HTTPException, Query, status

import pandas as pd

from proedge.api.schemas import (
    GameLineResponse,
    GameSummaryResponse,
    KalshiThresholdData,
    LineComparisonResponse,
    LineMatrixResponse,
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
        p
        for p in board.player_projections
        if home_up in p.home_team.upper()
        or home_up in p.away_team.upper()
        or away_up in p.home_team.upper()
        or away_up in p.away_team.upper()
    ]
    matched_lines = [
        gl
        for gl in board.game_lines
        if home_up in gl.home_team.upper()
        or home_up in gl.away_team.upper()
        or away_up in gl.home_team.upper()
        or away_up in gl.away_team.upper()
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


# ── Cross-source comparison endpoints ────────────────────────────────────────


@router.get(
    "/compare/{sport}",
    response_model=list[LineComparisonResponse],
    summary="All games for a sport with lines from every source",
    description=(
        "Fetches game totals from The Odds API (sportsbook consensus), PrizePicks, "
        "and Kalshi prediction markets in parallel and returns them side-by-side. "
        "Results are cached 15 minutes. `pp_vs_book` and `kalshi_vs_book` show "
        "discrepancies that can signal sharp vs. public disagreement."
    ),
)
async def compare_lines_board(sport: str):
    sport_lower = sport.lower()
    if sport_lower not in ("nba", "nfl", "mlb"):
        raise HTTPException(status_code=422, detail=f"Unsupported sport '{sport}'")

    from proedge.config import get_settings
    from proedge.pipeline.ingestion.line_aggregator import get_full_board

    board = await get_full_board(sport_lower, get_settings().odds_api_key)
    return [_comp_to_response(c) for c in board]


@router.get(
    "/compare/{sport}/{home_team}/{away_team}",
    response_model=LineComparisonResponse,
    summary="Line comparison for a specific matchup",
)
async def compare_lines_matchup(sport: str, home_team: str, away_team: str):
    sport_lower = sport.lower()
    if sport_lower not in ("nba", "nfl", "mlb"):
        raise HTTPException(status_code=422, detail=f"Unsupported sport '{sport}'")

    from proedge.config import get_settings
    from proedge.pipeline.ingestion.line_aggregator import get_line_comparison

    comp = await get_line_comparison(
        sport_lower, home_team, away_team, get_settings().odds_api_key
    )
    if not comp.sources:
        raise HTTPException(
            status_code=404,
            detail=f"No lines found for {home_team} vs {away_team} ({sport})",
        )
    return _comp_to_response(comp)


_matrix_model_cache: dict[str, object] = {}


@router.get(
    "/matrix/{sport}/{home_team}/{away_team}",
    response_model=LineMatrixResponse,
    summary="Full line matrix: all Kalshi thresholds with model edge + Underdog/PrizePicks comparison",
    description=(
        "For a specific matchup, returns every Kalshi total threshold (e.g. 198.5–228.5 "
        "for an NBA game) with its Yes/No ask prices, market-implied probability, and the "
        "model's predicted probability at that threshold. EV is calculated for both sides. "
        "Also includes the PrizePicks and Underdog game total lines for cross-platform comparison."
    ),
)
async def get_line_matrix(sport: str, home_team: str, away_team: str):
    sport_lower = sport.lower()
    if sport_lower not in ("nba", "nfl", "mlb"):
        raise HTTPException(status_code=422, detail=f"Unsupported sport '{sport}'")

    loop = asyncio.get_running_loop()

    # Fetch Kalshi (sync, run in executor)
    kalshi_game = await loop.run_in_executor(
        None, _fetch_kalshi_game, sport_lower, home_team, away_team
    )

    # Fetch PrizePicks async + Underdog sync concurrently
    pp_task = asyncio.create_task(_fetch_pp_total(sport_lower, home_team, away_team))
    ud_task = loop.run_in_executor(
        None, _fetch_underdog_total, sport_lower, home_team, away_team,
        kalshi_game.implied_line if kalshi_game else None
    )
    pp_line, ud_total = await asyncio.gather(pp_task, ud_task, return_exceptions=True)
    if isinstance(pp_line, Exception):
        pp_line = None
    if isinstance(ud_total, Exception):
        ud_total = None

    # Load model for this sport (cached)
    model, feature_names, feature_medians = _load_model_meta(sport_lower)

    # Score model at each threshold's line value individually — gives meaningful
    # variation (MLB 4.5 vs 9.5 are very different) instead of a uniform log-odds shift.
    thresholds: list[KalshiThresholdData] = []
    if kalshi_game and kalshi_game.threshold_data and model is not None and feature_names:
        try:
            rows = []
            for td in kalshi_game.threshold_data:
                r = {f: feature_medians.get(f, 0.0) for f in feature_names}
                r["total_line"] = td.threshold
                rows.append(r)
            X_batch = pd.DataFrame(rows)[feature_names].fillna(0)
            thresh_probs_list = [float(p) for p in model.predict_proba(X_batch)]
        except Exception:
            thresh_probs_list = [None] * len(kalshi_game.threshold_data)

        for td, model_prob in zip(kalshi_game.threshold_data, thresh_probs_list):
            ev_yes = ev_no = best_bet = edge = None
            if model_prob is not None:
                ev_yes = round(model_prob - td.yes_ask, 4)
                ev_no = round((1 - model_prob) - td.no_ask, 4)
                # Bet direction must agree with model's directional prediction.
                # Also restrict to thresholds within ±3 of the implied line —
                # per-threshold model scoring is unreliable far from the training
                # distribution of actual game lines.
                implied = kalshi_game.implied_line or 0
                near_implied = abs(td.threshold - implied) <= 3.0
                if 0.12 < td.implied_prob < 0.88 and near_implied:
                    if model_prob > 0.5 and ev_yes > 0:
                        best_bet, edge = "over", round(ev_yes, 4)
                    elif model_prob < 0.5 and ev_no > 0:
                        best_bet, edge = "under", round(ev_no, 4)
            thresholds.append(KalshiThresholdData(
                threshold=td.threshold,
                yes_ask=td.yes_ask,
                no_ask=td.no_ask,
                market_prob=td.implied_prob,
                model_prob=round(model_prob, 4) if model_prob is not None else None,
                ev_yes=ev_yes,
                ev_no=ev_no,
                best_bet=best_bet,
                edge=edge,
            ))
    elif kalshi_game and kalshi_game.threshold_data:
        for td in kalshi_game.threshold_data:
            thresholds.append(KalshiThresholdData(
                threshold=td.threshold,
                yes_ask=td.yes_ask,
                no_ask=td.no_ask,
                market_prob=td.implied_prob,
                model_prob=None,
                ev_yes=None,
                ev_no=None,
                best_bet=None,
                edge=None,
            ))

    return LineMatrixResponse(
        sport=sport_lower,
        home_team=home_team.upper(),
        away_team=away_team.upper(),
        kalshi_implied_line=kalshi_game.implied_line if kalshi_game else None,
        prizepicks_line=pp_line,
        underdog_line=ud_total.stat_value if ud_total else None,
        underdog_over_american=ud_total.over_american if ud_total else None,
        underdog_under_american=ud_total.under_american if ud_total else None,
        thresholds=thresholds,
    )


def _fetch_kalshi_game(sport: str, home: str, away: str):
    try:
        from proedge.pipeline.ingestion.kalshi_fetcher import get_implied_line
        return get_implied_line(sport, home, away)
    except Exception as exc:
        logger.warning("Kalshi game fetch error: %s", exc)
        return None


async def _fetch_pp_total(sport: str, home: str, away: str) -> float | None:
    try:
        from proedge.pipeline.ingestion.prizepicks_fetcher import fetch_board
        board = await asyncio.wait_for(fetch_board(sport), timeout=10.0)
        return board.total_line_for(home, away)
    except Exception:
        return None


def _fetch_underdog_total(sport: str, home: str, away: str, kalshi_line: float | None):
    try:
        from proedge.pipeline.ingestion.underdog_fetcher import find_game_total
        return find_game_total(sport, home, away, timeout=10.0)
    except Exception as exc:
        logger.warning("Underdog fetch error: %s", exc)
        return None


def _load_model_meta(sport: str):
    """Load (model, feature_names, feature_medians) with module-level cache."""
    if sport not in _matrix_model_cache:
        try:
            from proedge.pipeline.models.registry import ModelRegistry
            reg = ModelRegistry()
            model = reg.load(sport)
            meta = reg.load_meta(sport)
            _matrix_model_cache[sport] = (
                model,
                meta.get("feature_names", []),
                meta.get("feature_medians", {}),
            )
        except Exception as exc:
            logger.warning("Matrix: could not load %s model: %s", sport, exc)
            _matrix_model_cache[sport] = (None, [], {})
    return _matrix_model_cache[sport]


def _score_at_line(model, feature_names: list[str], medians: dict, total_line: float) -> float:
    """Run model at total_line with all other features at training medians."""
    row = {f: medians.get(f, 0.0) for f in feature_names}
    row["total_line"] = total_line
    X = pd.DataFrame([row])[feature_names].fillna(0)
    probs = model.predict_proba(X)
    # OverUnderEnsemble.predict_proba returns 1-D P(over) array
    return float(probs[0])



def _comp_to_response(comp) -> LineComparisonResponse:
    return LineComparisonResponse(
        sport=comp.sport,
        home_team=comp.home_team,
        away_team=comp.away_team,
        book_line=comp.book_line,
        prizepicks_line=comp.prizepicks_line,
        kalshi_line=comp.kalshi_line,
        kalshi_nearest_threshold=comp.kalshi_nearest_threshold,
        kalshi_nearest_prob=comp.kalshi_nearest_prob,
        consensus_line=comp.consensus_line,
        pp_vs_book=comp.pp_vs_book,
        kalshi_vs_book=comp.kalshi_vs_book,
        sources=comp.sources,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_response(
    board: PrizePicksBoard,
    *,
    include_promos: bool,
    status_filter: str,
) -> PrizePicksBoardResponse:
    props = [
        p
        for p in board.player_projections
        if (include_promos or not p.is_promo)
        and (
            status_filter == "all"
            or p.status == status_filter
            or
            # treat "pre_game" as the normal pre-game state
            (status_filter == "pre_game" and p.status in ("pre_game", "normal"))
        )
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
            p for p in props if p.stat_type == point_stat and p.odds_type not in ("demon", "goblin")
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
