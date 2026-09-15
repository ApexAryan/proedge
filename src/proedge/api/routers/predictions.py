"""Prediction endpoints — the primary API surface."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.api.schemas import (
    KalshiThresholdData,
    LineComparisonResponse,
    MarketScanGameResult,
    MarketScanRequest,
    MarketScanResponse,
    PredictionRequest,
    PredictionResponse,
    SettleRequest,
    SettleResponse,
)
from proedge.db.repositories import AlertRepository, GameRepository, PredictionRepository
from proedge.db.session import get_db
from proedge.monitoring.alerts import get_alert_manager
from proedge.monitoring.metrics import (
    INFERENCE_FEATURE_MISSING,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_PROB_OVER,
)
from proedge.pipeline.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])
_registry = ModelRegistry()
_model_cache: dict[str, object] = {}

# In-memory line movement tracking: event_ticker → (first_implied_line, first_seen_at_iso)
# Persists for the lifetime of the server process; cleared on restart.
_line_history: dict[str, tuple[float, str]] = {}


def _get_model(sport: str):
    if sport not in _model_cache:
        try:
            _model_cache[sport] = _registry.load(sport)
        except FileNotFoundError:
            return None
    return _model_cache[sport]


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction(req: PredictionRequest, db: AsyncSession = Depends(get_db)):
    t0 = time.perf_counter()
    sport = req.sport.value

    model = _get_model(sport)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No trained model available for sport '{sport}'. Run training first.",
        )

    meta = _registry.load_meta(sport)
    model_version = meta.get("version", "unknown")
    feature_names: list[str] = meta.get("feature_names", [])
    feature_medians: dict[str, float] = meta.get("feature_medians", {})

    from proedge.config import get_settings
    from proedge.pipeline.ingestion.line_aggregator import get_line_comparison

    _settings = get_settings()
    line_comp = None
    try:
        line_comp = await get_line_comparison(
            sport, req.home_team, req.away_team, _settings.odds_api_key
        )
        if line_comp.consensus_line is not None:
            req = req.model_copy(update={"total_line": line_comp.consensus_line})
    except Exception:
        pass  # best-effort; keep caller-supplied line

    # Auto-populate injury counts from ESPN if caller didn't provide them
    home_out = req.home_key_players_out
    away_out = req.away_key_players_out
    if home_out == 0 and away_out == 0:
        try:
            from proedge.pipeline.ingestion.injuries import InjuryFetcher

            fetcher = InjuryFetcher(timeout=5.0)
            reports = fetcher.fetch_all(sport)
            home_out = reports.get(
                req.home_team, type("_", (), {"key_players_out": 0})()
            ).key_players_out
            away_out = reports.get(
                req.away_team, type("_", (), {"key_players_out": 0})()
            ).key_players_out
        except Exception:
            pass  # injury fetch is best-effort; fall back to 0

    # Build inference feature row with available context
    X = _build_inference_features(req, feature_names, home_out, away_out, feature_medians, line_comp)

    # Validate feature dimensions — warn on mismatch to surface train-serve skew
    expected = set(feature_names)
    actual = set(X.columns)
    missing = expected - actual
    if missing:
        logger.warning(
            "sport=%s: %d features present in training but missing at inference: %s",
            sport,
            len(missing),
            sorted(missing)[:10],
        )
        try:
            INFERENCE_FEATURE_MISSING.labels(sport=sport).inc(len(missing))
        except Exception:
            pass

    predictions = model.predict_with_intervals(X)
    pred = predictions[0]

    direction = "over" if pred["prob_over"] >= 0.5 else "under"
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Persist game and prediction
    game_repo = GameRepository(db)
    pred_repo = PredictionRepository(db)

    game = await game_repo.create(
        sport=sport,
        home_team=req.home_team,
        away_team=req.away_team,
        game_date=req.game_date,
        total_line=req.total_line,
        status="scheduled",
    )

    features_snapshot = (
        {col: float(X[col].iloc[0]) for col in feature_names if col in X.columns}
        if req.include_features
        else None
    )

    db_pred = await pred_repo.create(
        game_id=game.id,
        model_version=model_version,
        sport=sport,
        prob_over=pred["prob_over"],
        prob_under=pred["prob_under"],
        ci_lower=pred["ci_lower"],
        ci_upper=pred["ci_upper"],
        predicted_direction=direction,
        confidence=pred["confidence"],
        features_snapshot=features_snapshot,
        latency_ms=latency_ms,
    )

    # Emit Prometheus metrics
    PREDICTION_COUNT.labels(sport=sport, direction=direction).inc()
    PREDICTION_CONFIDENCE.labels(sport=sport).observe(pred["confidence"])
    PREDICTION_PROB_OVER.labels(sport=sport).observe(pred["prob_over"])

    # Fire alert if confidence exceeds threshold; persist to DB for durability
    try:
        alert = get_alert_manager().evaluate(
            {
                "sport": sport,
                "home_team": req.home_team,
                "away_team": req.away_team,
                "game_date": str(req.game_date),
                "prob_over": pred["prob_over"],
                "prob_under": pred["prob_under"],
                "confidence": pred["confidence"],
                "total_line": req.total_line,
                "predicted_direction": direction,
            }
        )
        if alert is not None:
            alert_repo = AlertRepository(db)
            await alert_repo.create(
                alert_id=alert.alert_id,
                sport=alert.sport,
                home_team=alert.home_team,
                away_team=alert.away_team,
                game_date=alert.game_date,
                direction=alert.direction,
                prob_over=alert.prob_over,
                confidence=alert.confidence,
                edge=alert.edge,
                total_line=alert.total_line,
                fired=alert.fired,
                webhook_response=alert.webhook_response,
                created_at=alert.created_at,
            )
    except Exception:
        pass  # alerts are best-effort

    line_comparison_resp = None
    if line_comp is not None and line_comp.sources:
        line_comparison_resp = LineComparisonResponse(
            sport=line_comp.sport,
            home_team=line_comp.home_team,
            away_team=line_comp.away_team,
            book_line=line_comp.book_line,
            prizepicks_line=line_comp.prizepicks_line,
            kalshi_line=line_comp.kalshi_line,
            kalshi_nearest_threshold=line_comp.kalshi_nearest_threshold,
            kalshi_nearest_prob=line_comp.kalshi_nearest_prob,
            consensus_line=line_comp.consensus_line,
            pp_vs_book=line_comp.pp_vs_book,
            kalshi_vs_book=line_comp.kalshi_vs_book,
            sources=line_comp.sources,
        )

    return PredictionResponse(
        prediction_id=db_pred.id,
        game_id=game.id,
        sport=req.sport,
        home_team=req.home_team,
        away_team=req.away_team,
        game_date=req.game_date,
        total_line=req.total_line,
        model_version=model_version,
        prob_over=pred["prob_over"],
        prob_under=pred["prob_under"],
        ci_lower=pred["ci_lower"],
        ci_upper=pred["ci_upper"],
        predicted_direction=direction,
        confidence=pred["confidence"],
        latency_ms=latency_ms,
        features=features_snapshot,
        line_comparison=line_comparison_resp,
    )


@router.post("/scan", response_model=MarketScanResponse, status_code=status.HTTP_200_OK)
async def scan_markets(req: MarketScanRequest, db: AsyncSession = Depends(get_db)):
    """Fetch all live Kalshi markets, run model on each game, return sorted by confidence."""
    import asyncio as _asyncio
    from datetime import datetime, timezone

    from proedge.pipeline.ingestion import kalshi_fetcher

    game_repo = GameRepository(db)
    pred_repo = PredictionRepository(db)
    all_results: list[MarketScanGameResult] = []
    active_sports: list[str] = []

    for sport in req.sports:
        model = _get_model(sport)
        if model is None:
            continue

        meta = _registry.load_meta(sport)
        model_version = meta.get("version", "unknown")
        feature_names: list[str] = meta.get("feature_names", [])
        feature_medians: dict[str, float] = meta.get("feature_medians", {})

        try:
            kalshi_games = kalshi_fetcher.fetch_totals(sport)
        except Exception as exc:
            logger.warning("scan: kalshi fetch failed for %s: %s", sport, exc)
            continue

        if not kalshi_games:
            logger.info("scan: no Kalshi markets for %s", sport)
            continue

        active_sports.append(sport)

        # Fetch injuries once per sport — keyed by team abbreviation
        injury_counts: dict[str, int] = {}
        try:
            from proedge.pipeline.ingestion.injuries import InjuryFetcher
            inj_reports = InjuryFetcher(timeout=8.0).fetch_all(sport)
            injury_counts = {team: r.key_players_out for team, r in inj_reports.items()}
            logger.info("scan: %s injury data loaded (%d teams)", sport.upper(), len(injury_counts))
        except Exception as exc:
            logger.warning("scan: injury fetch failed for %s: %s", sport, exc)

        # NBA: fetch live rolling stats (cached 4h) to replace training medians.
        # Only fetch the teams actually playing today — avoids rate limits.
        nba_live_cache: dict[str, dict[str, float]] = {}
        if sport == "nba" and kalshi_games:
            try:
                from proedge.pipeline.ingestion import nba_live_stats
                teams_needed = list({
                    t.upper()
                    for kg in kalshi_games
                    for t in [kg.team1, kg.team2]
                })
                nba_live_cache = await _asyncio.get_event_loop().run_in_executor(
                    None, nba_live_stats.get_teams, teams_needed
                )
                logger.info(
                    "scan: NBA live stats loaded for %d/%d teams",
                    sum(1 for v in nba_live_cache.values() if v),
                    len(teams_needed),
                )
            except Exception as exc:
                logger.warning("scan: NBA live stats fetch failed: %s", exc)

        # MLB: fetch live rolling stats (cached 4h) — same pattern as NBA.
        mlb_live_cache: dict[str, dict[str, float]] = {}
        if sport == "mlb" and kalshi_games:
            try:
                from proedge.pipeline.ingestion import mlb_live_stats
                teams_needed = list({
                    t.upper()
                    for kg in kalshi_games
                    for t in [kg.team1, kg.team2]
                })
                mlb_live_cache = await _asyncio.get_event_loop().run_in_executor(
                    None, mlb_live_stats.get_teams, teams_needed
                )
                logger.info(
                    "scan: MLB live stats loaded for %d/%d teams",
                    sum(1 for v in mlb_live_cache.values() if v),
                    len(teams_needed),
                )
            except Exception as exc:
                logger.warning("scan: MLB live stats fetch failed: %s", exc)

        _month = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

        for kg in kalshi_games:
            home_team = kg.team1.upper()
            away_team = kg.team2.upper()
            implied_line = kg.implied_line

            # Track line movement from first-seen implied line this session
            _prev = _line_history.get(kg.event_ticker)
            if _prev is not None:
                _first_line, _ = _prev
                line_movement: float | None = round(implied_line - _first_line, 2)
            else:
                line_movement = None
                _line_history[kg.event_ticker] = (implied_line, datetime.now(timezone.utc).isoformat())

            # Parse game date from ticker
            try:
                gd = kg.game_date  # "YYMONDD"
                yr = 2000 + int(gd[:2])
                mon = _month.get(gd[2:5].upper(), 1)
                day = int(gd[5:7])
                game_dt = datetime(yr, mon, day, 19, 0, tzinfo=timezone.utc)
            except Exception:
                game_dt = datetime.now(timezone.utc)

            # Build feature row with contextual enrichment beyond just the line
            row = {f: feature_medians.get(f, 0.0) for f in feature_names}
            row["total_line"] = implied_line
            row["home_advantage"] = 1.0

            # NBA/MLB-specific context
            if sport == "nba":
                row["is_dome"] = 1.0
                row["dome_flag"] = 1.0
                # NBA April–June = playoffs (lower pace, more defense → under lean)
                if game_dt.month in (4, 5, 6):
                    row["is_playoff"] = 1.0
            elif sport == "mlb":
                row["is_dome"] = 0.0
                row["dome_flag"] = 0.0

            # Injury enrichment — reduces scoring on the affected side
            home_out = injury_counts.get(home_team, 0)
            away_out = injury_counts.get(away_team, 0)
            row["home_key_players_out"] = float(home_out)
            row["away_key_players_out"] = float(away_out)
            row["injury_pts_impact"] = (home_out - away_out) * -3.0

            # NBA live rolling stats — overwrite medians with real team context
            if nba_live_cache:
                from proedge.pipeline.ingestion.nba_live_stats import inject_team_features
                inject_team_features(row, home_team, "home_", nba_live_cache)
                inject_team_features(row, away_team, "away_", nba_live_cache)

            # MLB live rolling stats — same injection pattern
            if mlb_live_cache:
                from proedge.pipeline.ingestion.mlb_live_stats import inject_team_features as mlb_inject
                mlb_inject(row, home_team, "home_", mlb_live_cache)
                mlb_inject(row, away_team, "away_", mlb_live_cache)

            X = pd.DataFrame([row])

            try:
                intervals = model.predict_with_intervals(X)
                p_data = intervals[0]
                probs_raw = model.predict_proba(X)
                model_anchor_prob = float(probs_raw[0])
            except Exception as exc:
                logger.warning("scan: model failed for %s %s/%s: %s", sport, home_team, away_team, exc)
                continue

            direction = "over" if p_data["prob_over"] >= 0.5 else "under"

            best_threshold: float | None = None
            best_bet: str | None = None
            best_edge: float | None = None
            threshold_data: list[KalshiThresholdData] = []

            # Score model at each threshold's line value individually so model_prob
            # reflects genuine sensitivity to the total (e.g. 4.5 vs 8.5 in MLB),
            # rather than a uniform log-odds shift that pushes all thresholds the same way.
            if kg.threshold_data:
                thresh_rows = []
                for td in kg.threshold_data:
                    thresh_row = dict(row)
                    thresh_row["total_line"] = td.threshold
                    thresh_rows.append(thresh_row)
                try:
                    X_thresh = pd.DataFrame(thresh_rows)
                    thresh_probs = [float(p) for p in model.predict_proba(X_thresh)]
                except Exception as exc:
                    logger.warning("scan: per-threshold scoring failed for %s: %s", kg.event_ticker, exc)
                    thresh_probs = [model_anchor_prob] * len(kg.threshold_data)
            else:
                thresh_probs = []

            for td, model_prob in zip(kg.threshold_data, thresh_probs):
                ev_yes = round(model_prob - td.yes_ask, 4)
                ev_no = round((1 - model_prob) - td.no_ask, 4)

                # Bet direction must agree with model's directional prediction:
                # only recommend OVER when model itself thinks >50% likely, and
                # UNDER when model thinks <50% likely. This prevents confusing cases
                # like "model says 82% OVER → best bet UNDER" that occur when a tiny
                # market/model gap produces technically-positive-EV on the wrong side.
                #
                # Also restrict best_bet to thresholds within ±3 of the implied line.
                # The model's total_line feature is trained on typical game lines; at
                # extreme thresholds (e.g. MLB 13.5 when line is 8.9) the per-threshold
                # probability is unreliable and can produce spuriously large EV values.
                tradeable = 0.12 < td.implied_prob < 0.88
                near_implied = abs(td.threshold - implied_line) <= 3.0
                bet: str | None = None
                bet_edge: float = 0.0
                if tradeable and near_implied:
                    if model_prob > 0.5 and ev_yes > 0:
                        bet = "over"
                        bet_edge = ev_yes
                    elif model_prob < 0.5 and ev_no > 0:
                        bet = "under"
                        bet_edge = ev_no

                if bet and (best_edge is None or bet_edge > best_edge):
                    best_edge = round(bet_edge, 4)
                    best_threshold = td.threshold
                    best_bet = bet

                threshold_data.append(KalshiThresholdData(
                    threshold=td.threshold,
                    yes_ask=td.yes_ask,
                    no_ask=td.no_ask,
                    market_prob=td.implied_prob,
                    model_prob=round(model_prob, 4),
                    ev_yes=ev_yes,
                    ev_no=ev_no,
                    best_bet=bet if bet else None,
                    edge=round(bet_edge, 4) if bet else None,
                ))

            # Persist game + prediction to DB (best-effort; scan still returns if DB is down)
            game = None
            db_pred = None
            try:
                existing = await game_repo.get_by_teams_date(sport, home_team, away_team, game_dt)
                if existing is None:
                    existing = await game_repo.get_by_teams_date(sport, away_team, home_team, game_dt)
                    if existing:
                        home_team, away_team = away_team, home_team
                if existing is None:
                    existing = await game_repo.get_by_external_id(kg.event_ticker)

                if existing is None:
                    game = await game_repo.create(
                        sport=sport,
                        home_team=home_team,
                        away_team=away_team,
                        game_date=game_dt,
                        total_line=implied_line,
                        status="scheduled",
                        external_id=kg.event_ticker,
                    )
                else:
                    game = existing
                    updates: dict = {}
                    if game.home_team != home_team:
                        updates["home_team"] = home_team
                    if game.away_team != away_team:
                        updates["away_team"] = away_team
                    if abs((game.total_line or 0) - implied_line) > 0.05:
                        updates["total_line"] = implied_line
                    if updates:
                        await game_repo.update_fields(game.id, **updates)

                existing_pred = await pred_repo.get_latest_for_game(game.id)
                _can_reuse = (
                    existing_pred is not None
                    and existing_pred.model_version == model_version
                    and not (sport == "nba" and nba_live_cache)
                    and not (sport == "mlb" and mlb_live_cache)
                )
                if _can_reuse:
                    db_pred = existing_pred
                else:
                    db_pred = await pred_repo.create(
                        game_id=game.id,
                        model_version=model_version,
                        sport=sport,
                        prob_over=p_data["prob_over"],
                        prob_under=p_data["prob_under"],
                        ci_lower=p_data["ci_lower"],
                        ci_upper=p_data["ci_upper"],
                        predicted_direction=direction,
                        confidence=p_data["confidence"],
                        features_snapshot=None,
                        latency_ms=0.0,
                    )
            except Exception as exc:
                logger.warning("scan: DB persist failed for %s %s: %s", sport, kg.event_ticker, exc)

            live_stats = False
            if sport == "nba":
                live_stats = bool(nba_live_cache.get(home_team) or nba_live_cache.get(away_team))
            elif sport == "mlb":
                live_stats = bool(mlb_live_cache.get(home_team) or mlb_live_cache.get(away_team))

            all_results.append(MarketScanGameResult(
                game_id=game.id if game else None,
                prediction_id=db_pred.id if db_pred else None,
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                game_date=game_dt,
                kalshi_implied_line=implied_line,
                model_prob_over=p_data["prob_over"],
                model_prob_under=p_data["prob_under"],
                predicted_direction=direction,
                confidence=p_data["confidence"],
                best_threshold=best_threshold,
                best_bet=best_bet,
                best_edge=best_edge,
                thresholds=threshold_data,
                line_movement=line_movement,
                ci_lower=p_data.get("ci_lower"),
                ci_upper=p_data.get("ci_upper"),
                home_key_players_out=home_out,
                away_key_players_out=away_out,
                live_stats=live_stats,
            ))

    all_results.sort(key=lambda r: r.confidence, reverse=True)

    return MarketScanResponse(
        scanned_at=datetime.now(timezone.utc),
        sports=active_sports,
        total_games=len(all_results),
        results=all_results,
    )


@router.get("/performance", response_model=dict, status_code=status.HTTP_200_OK)
async def get_performance(db: AsyncSession = Depends(get_db)):
    """Hit rate and hypothetical P&L for settled predictions, per sport.

    P&L assumes $100/bet at -110 odds (win $90.91, lose $100).
    Predictions are deduplicated per game (latest settled prediction wins).
    """
    from sqlalchemy import func as _func, select as _sel
    from proedge.db.models import Prediction

    results = {}
    try:
        for sport in ["nba", "mlb", "nfl"]:
            ranked = (
                _sel(
                    Prediction.is_correct,
                    _func.row_number()
                    .over(
                        partition_by=Prediction.game_id,
                        order_by=Prediction.predicted_at.desc(),
                    )
                    .label("rn"),
                )
                .where(
                    Prediction.sport == sport,
                    Prediction.is_correct.isnot(None),
                )
                .subquery()
            )
            total_r = await db.execute(
                _sel(_func.count()).select_from(ranked).where(ranked.c.rn == 1)
            )
            total = total_r.scalar() or 0

            wins_r = await db.execute(
                _sel(_func.count())
                .select_from(ranked)
                .where(ranked.c.rn == 1, ranked.c.is_correct.is_(True))
            )
            wins = wins_r.scalar() or 0
            losses = total - wins
            pnl = round(wins * 90.91 - losses * 100.0, 2) if total else None
            results[sport] = {
                "total_settled": total,
                "wins": wins,
                "losses": losses,
                "hit_rate": round(wins / total, 4) if total else None,
                "pnl": pnl,
            }

        logged_r = await db.execute(_sel(_func.count(Prediction.id)))
        pending_r = await db.execute(
            _sel(_func.count(Prediction.id)).where(Prediction.is_correct.is_(None))
        )
        logged = logged_r.scalar() or 0
        pending = pending_r.scalar() or 0
        wins = sum(results[s]["wins"] for s in ["nba", "mlb", "nfl"])
        losses = sum(results[s]["losses"] for s in ["nba", "mlb", "nfl"])
        settled = wins + losses
        pnl = round(wins * 90.91 - losses * 100.0, 2) if settled else None
        roi = round((wins * 0.9091 - losses) / settled, 4) if settled else None
        results["overall"] = {
            "logged": logged,
            "pending": pending,
            "total_settled": settled,
            "wins": wins,
            "losses": losses,
            "hit_rate": round(wins / settled, 4) if settled else None,
            "roi": roi,
            "pnl": pnl,
        }
    except Exception as exc:
        logger.warning("performance: DB query failed: %s", exc)
        empty = {"total_settled": 0, "wins": 0, "losses": 0, "hit_rate": None, "pnl": None}
        for sport in ["nba", "mlb", "nfl"]:
            results[sport] = dict(empty)
        results["overall"] = {
            **empty,
            "logged": 0,
            "pending": 0,
            "roi": None,
        }

    return results


@router.post("/settle/auto", response_model=dict, status_code=status.HTTP_200_OK)
async def auto_settle(db: AsyncSession = Depends(get_db)):
    """Fetch final scores from ESPN and settle any pending predictions.

    Checks today + yesterday for each active sport. Idempotent: skips
    predictions that are already settled (is_correct not null).
    """
    import asyncio as _asyncio
    from datetime import date, timedelta
    from proedge.pipeline.ingestion import score_fetcher

    game_repo = GameRepository(db)
    pred_repo = PredictionRepository(db)

    settled = 0
    already_settled = 0
    no_score_match = 0

    today = date.today()

    try:
        for sport in ["nba", "mlb", "nfl"]:
            for delta in range(2):  # today and yesterday
                check_date = today - timedelta(days=delta)
                scores = await _asyncio.get_event_loop().run_in_executor(
                    None, score_fetcher.fetch_scores, sport, check_date
                )
                if not scores:
                    continue

                from datetime import datetime, timezone
                day_start = datetime(check_date.year, check_date.month, check_date.day, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)

                games = await game_repo.list_by_sport_date(sport, day_start, day_end)
                for game in games:
                    score = score_fetcher.match_score(game.home_team, game.away_team, scores)
                    if not score:
                        no_score_match += 1
                        continue

                    total_line = game.total_line
                    if not total_line:
                        continue

                    preds = await pred_repo.get_by_game(game.id)
                    for pred in preds:
                        if pred.is_correct is not None:
                            already_settled += 1
                            continue
                        await pred_repo.settle(
                            prediction_id=pred.id,
                            actual_total=float(score["total"]),
                            closing_line=total_line,
                            predicted_direction=pred.predicted_direction,
                            bet_line=total_line,
                        )
                        settled += 1

                    await game_repo.update_fields(
                        game.id,
                        home_score=score["home_score"],
                        away_score=score["away_score"],
                        result_over=score["total"] > total_line,
                        status="final",
                    )

    except Exception as exc:
        logger.warning("auto_settle: failed: %s", exc)
        return {"error": str(exc), "settled": settled}

    return {
        "settled": settled,
        "already_settled": already_settled,
        "no_score_match": no_score_match,
    }


@router.delete("/purge", response_model=dict)
async def purge_old_predictions(days_old: int = 7, db: AsyncSession = Depends(get_db)):
    """Delete predictions (and orphaned games) older than `days_old` days."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import delete as _delete, select as _select
    from proedge.db.models import Game, Prediction

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_old)
    stale = (await db.execute(
        _select(Prediction.id, Prediction.game_id).where(Prediction.predicted_at < cutoff)
    )).all()
    if not stale:
        return {"deleted_predictions": 0, "deleted_games": 0}

    pred_ids = [r[0] for r in stale]
    candidate_game_ids = list({r[1] for r in stale})
    await db.execute(_delete(Prediction).where(Prediction.id.in_(pred_ids)))

    surviving = (await db.execute(
        _select(Prediction.game_id).where(Prediction.game_id.in_(candidate_game_ids))
    )).scalars().all()
    orphaned = [gid for gid in candidate_game_ids if gid not in set(surviving)]
    deleted_games = 0
    if orphaned:
        res = await db.execute(_delete(Game).where(Game.id.in_(orphaned)))
        deleted_games = res.rowcount

    return {"deleted_predictions": len(pred_ids), "deleted_games": deleted_games}


@router.get("/recent", response_model=list[dict])
async def get_recent_predictions(
    sport: str | None = None,
    limit: int = 50,
    sort_by: str = "newest",
    db: AsyncSession = Depends(get_db),
):
    """Recent predictions with full game context. sort_by: newest | strongest | weakest"""
    pred_repo = PredictionRepository(db)
    # Fetch more than requested so dedup doesn't starve the result set
    rows = await pred_repo.get_recent_with_games(sport=sport, limit=limit * 5)

    # One prediction per game — DB returns newest-first so first hit per game_id is latest
    seen_games: set = set()
    deduped = []
    for row in rows:
        gid = row[1].id
        if gid not in seen_games:
            seen_games.add(gid)
            deduped.append(row)
    rows = deduped

    if sort_by == "strongest":
        rows = sorted(rows, key=lambda row: row[0].confidence or 0, reverse=True)
    elif sort_by == "weakest":
        rows = sorted(rows, key=lambda row: row[0].confidence or 0)

    rows = rows[:limit]
    return [
        {
            "prediction_id": str(p.id),
            "game_id": str(p.game_id),
            "sport": p.sport,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "game_date": g.game_date.isoformat() if g.game_date else None,
            "total_line": g.total_line,
            "model_version": p.model_version,
            "prob_over": p.prob_over,
            "prob_under": p.prob_under,
            "ci_lower": p.ci_lower,
            "ci_upper": p.ci_upper,
            "predicted_direction": p.predicted_direction,
            "confidence": p.confidence,
            "predicted_at": p.predicted_at.isoformat() if p.predicted_at else None,
            "is_correct": p.is_correct,
            "clv": p.clv,
            "actual_total": p.actual_total,
            "closing_line": p.closing_line,
            "settled_at": p.settled_at.isoformat() if p.settled_at else None,
        }
        for p, g in rows
    ]


@router.get("/alerts/recent", response_model=list[dict])
async def get_recent_alerts(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Recent high-confidence alerts. Reads from DB for durability; falls back to in-memory."""
    try:
        alert_repo = AlertRepository(db)
        records = await alert_repo.get_recent(limit=limit)
        return [
            {
                "alert_id": r.alert_id,
                "sport": r.sport,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "game_date": r.game_date,
                "direction": r.direction,
                "confidence": r.confidence,
                "prob_over": r.prob_over,
                "edge": r.edge,
                "total_line": r.total_line,
                "fired": r.fired,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    except Exception:
        # DB unavailable — serve from in-memory deque
        mgr = get_alert_manager()
        return [
            {
                "alert_id": a.alert_id,
                "sport": a.sport,
                "home_team": a.home_team,
                "away_team": a.away_team,
                "game_date": a.game_date,
                "direction": a.direction,
                "confidence": a.confidence,
                "prob_over": a.prob_over,
                "edge": a.edge,
                "total_line": a.total_line,
                "fired": a.fired,
                "created_at": a.created_at.isoformat(),
            }
            for a in mgr.recent(limit)
        ]


@router.get("/{game_id}", response_model=list[PredictionResponse])
async def get_predictions_for_game(game_id: str, db: AsyncSession = Depends(get_db)):
    import uuid as _uuid

    try:
        gid = _uuid.UUID(game_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid game_id UUID")

    pred_repo = PredictionRepository(db)
    game_repo = GameRepository(db)

    game = await game_repo.get_by_id(gid)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    preds = await pred_repo.get_by_game(gid)
    return [
        PredictionResponse(
            prediction_id=p.id,
            game_id=game.id,
            sport=game.sport,
            home_team=game.home_team,
            away_team=game.away_team,
            game_date=game.game_date,
            total_line=game.total_line or 0,
            model_version=p.model_version,
            prob_over=p.prob_over,
            prob_under=p.prob_under,
            ci_lower=p.ci_lower,
            ci_upper=p.ci_upper,
            predicted_direction=p.predicted_direction,
            confidence=p.confidence,
            latency_ms=p.latency_ms or 0,
        )
        for p in preds
    ]


@router.post("/{prediction_id}/settle", response_model=SettleResponse)
async def settle_prediction(
    prediction_id: str,
    req: SettleRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record final score and closing line; compute CLV and correctness."""
    import uuid as _uuid

    try:
        pid = _uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction_id UUID")

    pred_repo = PredictionRepository(db)
    game_repo = GameRepository(db)
    pred = await pred_repo.get_by_id(pid)
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    game = await game_repo.get_by_id(pred.game_id)
    bet_line = game.total_line if game and game.total_line else req.closing_line
    await pred_repo.settle(
        prediction_id=pid,
        actual_total=req.actual_total,
        closing_line=req.closing_line,
        predicted_direction=pred.predicted_direction,
        bet_line=bet_line,
    )

    result_over = req.actual_total > req.closing_line
    is_correct = (pred.predicted_direction == "over") == result_over
    clv = (
        (req.closing_line - bet_line)
        if pred.predicted_direction == "over"
        else (bet_line - req.closing_line)
    )

    clv_desc = f"+{clv:.1f}" if clv >= 0 else f"{clv:.1f}"
    return SettleResponse(
        prediction_id=pid,
        actual_total=req.actual_total,
        closing_line=req.closing_line,
        clv=round(clv, 2),
        is_correct=is_correct,
        predicted_direction=pred.predicted_direction,
        message=f"{'✓ Correct' if is_correct else '✗ Wrong'} | CLV {clv_desc} | actual={req.actual_total} close={req.closing_line}",
    )


def _build_inference_features(
    req: PredictionRequest,
    feature_names: list[str],
    home_key_out: int = 0,
    away_key_out: int = 0,
    feature_medians: dict[str, float] | None = None,
    line_comp=None,
) -> pd.DataFrame:
    """
    Constructs a single-row feature DataFrame for inference.
    Known signals from the request are filled explicitly. Unknown rolling
    features default to their training-set median (stored in model meta)
    rather than 0 so the model sees a realistic baseline for unseen matchups.
    """
    # Seed with training medians — much better default than zero for rolling stats
    medians = feature_medians or {}
    row: dict[str, float] = {f: medians.get(f, 0.0) for f in feature_names}

    # Core game context
    row["total_line"] = req.total_line
    row["home_advantage"] = 1.0

    if req.home_rest_days is not None:
        row["home_rest_days"] = float(req.home_rest_days)
    if req.away_rest_days is not None:
        row["away_rest_days"] = float(req.away_rest_days)

    if req.home_rest_days is not None and req.away_rest_days is not None:
        row["home_rest_advantage"] = float(req.home_rest_days - req.away_rest_days)
        row["home_back_to_back"] = float(req.home_rest_days <= 1)
        row["away_back_to_back"] = float(req.away_rest_days <= 1)

    # GROUP C — situational context
    row["wind_speed_mph"] = req.wind_speed_mph
    row["temperature_f"] = req.temperature_f
    row["is_dome"] = float(req.is_dome)
    row["altitude_feet"] = req.altitude_feet
    row["is_playoff"] = float(req.is_playoff)
    row["altitude_boost"] = req.altitude_feet / 5280.0
    row["dome_flag"] = float(req.is_dome)
    row["wind_under_signal"] = float(req.wind_speed_mph > 15)
    row["wind_severity"] = req.wind_speed_mph / 30.0
    row["cold_game"] = float(req.temperature_f < 40)
    row["hot_game"] = float(req.temperature_f > 85)

    # GROUP D — market / sharp signals
    row["line_movement"] = req.line_movement
    row["public_over_pct"] = req.public_over_pct
    row["sharp_over_pct"] = req.sharp_over_pct
    row["ref_foul_rate"] = req.ref_foul_rate
    row["ump_walk_rate"] = req.ump_walk_rate
    row["sharp_vs_public"] = req.sharp_over_pct - req.public_over_pct
    row["line_move_magnitude"] = abs(req.line_movement)
    row["line_move_direction"] = float(np.sign(req.line_movement))

    # GROUP E — injury counts (prefer auto-fetched values when caller sent 0)
    h_out = max(req.home_key_players_out, home_key_out)
    a_out = max(req.away_key_players_out, away_key_out)
    row["home_key_players_out"] = float(h_out)
    row["away_key_players_out"] = float(a_out)
    row["injury_pts_impact"] = (h_out - a_out) * -3.0

    # Legacy
    row["home_injury_impact"] = req.home_injury_impact
    row["away_injury_impact"] = req.away_injury_impact

    # GROUP F — cross-source line discrepancy signals
    # These are only non-zero when live line data is available at inference time.
    # The model ignores unknown features for now; they become active after the next retrain.
    if line_comp is not None:
        row["pp_vs_book"] = line_comp.pp_vs_book or 0.0
        row["kalshi_vs_book"] = line_comp.kalshi_vs_book or 0.0
        row["kalshi_implied_prob"] = line_comp.kalshi_nearest_prob or 0.5
        row["has_kalshi"] = float(line_comp.kalshi_line is not None)
        row["has_prizepicks"] = float(line_comp.prizepicks_line is not None)

    return pd.DataFrame([row])
