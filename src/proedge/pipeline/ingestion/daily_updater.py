"""
Daily game updater: fetches completed games, appends to historical data,
clears the feature cache, and optionally triggers a model retrain.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proedge.pipeline.ingestion.injuries import injuries_from_boxscore

logger = logging.getLogger(__name__)

_DATA_DIR = Path("./data")
_FEATURES_DIR = Path("./data/features")

# BoxScoreTraditionalV3 team stat key → internal stat name (matches STAT_KEYS in stats.py)
_BSCORE_STAT_MAP: dict[str, str] = {
    "fieldGoalsMade": "fieldGoalsMade",
    "fieldGoalsAttempted": "fieldGoalAttempts",
    "fieldGoalsPercentage": "fieldGoalPct",
    "threePointersMade": "threesMade",
    "threePointersAttempted": "threePointAttempts",
    "threePointersPercentage": "threePointPct",
    "freeThrowsMade": "freeThrowsMade",
    "freeThrowsAttempted": "freeThrowAttempts",
    "freeThrowsPercentage": "freeThrowPct",
    "reboundsOffensive": "offensiveRebounds",
    "reboundsDefensive": "defensiveRebounds",
    "reboundsTotal": "rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "foulsPersonal": "personalFouls",
    "points": "points",
}

_ALTITUDE_MAP = {"DEN": 5280.0, "COL": 5280.0, "UTA": 4300.0}
_MIN_NEW_GAMES_TO_RETRAIN = 30  # retrain after accumulating this many new games


@dataclass
class UpdateResult:
    sport: str
    date: str
    games_found: int = 0
    games_added: int = 0
    games_skipped: int = 0  # already in historical (dedup)
    retrain_triggered: bool = False
    retrain_metrics: dict = field(default_factory=dict)
    error: str | None = None


class DailyUpdater:
    """
    Orchestrates: fetch → append → cache-clear → optional retrain.

    Usage:
        updater = DailyUpdater("nba")
        result = updater.run(date(2026, 4, 24))
    """

    def __init__(self, sport: str, data_dir: str = "./data", auto_retrain: bool = False):
        self.sport = sport.lower()
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.auto_retrain = auto_retrain
        self.historical_path = self.data_dir / f"{sport}_historical.parquet"

    def run(self, target_date: date | None = None) -> UpdateResult:
        """Run the full update pipeline for a given date (default: yesterday)."""
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        date_str = target_date.strftime("%Y-%m-%d")
        result = UpdateResult(sport=self.sport, date=date_str)

        try:
            new_games = self._fetch_completed_games(target_date)
            result.games_found = len(new_games)

            if new_games.empty:
                logger.info("No completed %s games on %s", self.sport.upper(), date_str)
                return result

            added, skipped = self._append_to_historical(new_games)
            result.games_added = added
            result.games_skipped = skipped
            logger.info(
                "Daily update %s %s: +%d games (%d already existed)",
                self.sport.upper(),
                date_str,
                added,
                skipped,
            )

            if added > 0:
                self._clear_feature_cache()
                self._settle_predictions(new_games)

            if added > 0 and self.auto_retrain:
                total_new = self._count_new_since_last_retrain()
                if total_new >= _MIN_NEW_GAMES_TO_RETRAIN:
                    result.retrain_triggered = True
                    result.retrain_metrics = self._retrain()

        except Exception as exc:
            logger.exception("Daily update failed for %s %s: %s", self.sport, date_str, exc)
            result.error = str(exc)

        return result

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _fetch_completed_games(self, target_date: date) -> pd.DataFrame:
        _dispatch = {
            "nba": self._fetch_nba_games,
            "nfl": self._fetch_nfl_games,
            "mlb": self._fetch_mlb_games,
        }
        fetch = _dispatch.get(self.sport)
        if fetch is None:
            logger.info("Real-data fetch not implemented for %s — skipping", self.sport.upper())
            return pd.DataFrame()
        return fetch(target_date)

    def _fetch_nba_games(self, target_date: date) -> pd.DataFrame:
        from nba_api.stats.endpoints import scoreboardv3, boxscoretraditionalv3

        date_str = target_date.strftime("%m/%d/%Y")
        logger.info("Fetching NBA games for %s", date_str)

        sb = scoreboardv3.ScoreboardV3(game_date=date_str, timeout=30)
        games_meta = sb.get_dict().get("scoreboard", {}).get("games", [])
        if not games_meta:
            return pd.DataFrame()

        rows: list[dict] = []
        injury_records: list[tuple] = []
        for meta in games_meta:
            if meta.get("gameStatus", 1) != 3:  # 3 = Final
                logger.debug("Skipping non-final game %s", meta.get("gameId"))
                continue

            game_id = meta["gameId"]
            time.sleep(0.7)  # respect NBA API rate limits

            try:
                bx = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30)
                bx_data = bx.get_dict().get("boxScoreTraditional", {})
            except Exception as exc:
                logger.warning("BoxScore fetch failed for %s: %s", game_id, exc)
                continue

            home = bx_data.get("homeTeam", {})
            away = bx_data.get("awayTeam", {})
            home_abbr = home.get("teamTricode", "")
            away_abbr = away.get("teamTricode", "")
            home_stats = home.get("statistics", {})
            away_stats = away.get("statistics", {})

            h_pts = int(home_stats.get("points", 0))
            a_pts = int(away_stats.get("points", 0))
            total = h_pts + a_pts

            # Compute possessions (Hollinger)
            h_fga = float(home_stats.get("fieldGoalsAttempted", 85))
            h_fta = float(home_stats.get("freeThrowsAttempted", 22))
            h_oreb = float(home_stats.get("reboundsOffensive", 10))
            h_tov = float(home_stats.get("turnovers", 14))
            a_fga = float(away_stats.get("fieldGoalsAttempted", 85))
            a_fta = float(away_stats.get("freeThrowsAttempted", 22))
            a_oreb = float(away_stats.get("reboundsOffensive", 10))
            a_tov = float(away_stats.get("turnovers", 14))
            h_poss = max(1.0, h_fga - h_oreb + h_tov + 0.44 * h_fta)
            a_poss = max(1.0, a_fga - a_oreb + a_tov + 0.44 * a_fta)

            h_fg3a = float(home_stats.get("threePointersAttempted", 33))
            h_dreb = float(home_stats.get("reboundsDefensive", 34))
            a_fg3a = float(away_stats.get("threePointersAttempted", 33))
            a_dreb = float(away_stats.get("reboundsDefensive", 34))

            # Injury counts from player comments
            h_injury = injuries_from_boxscore(home.get("players", []), home_abbr, "nba")
            a_injury = injuries_from_boxscore(away.get("players", []), away_abbr, "nba")

            game_date = pd.Timestamp(meta.get("gameTimeUTC", str(target_date)))
            season = int(game_date.year) if game_date.month >= 10 else int(game_date.year) - 1

            row: dict[str, Any] = {
                "game_id": game_id,
                "sport": "nba",
                "season": season,
                "game_date": game_date,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_score": h_pts,
                "away_score": a_pts,
                "total": total,
                "total_line": np.nan,  # filled below by proxy
                "result_over": np.nan,
                "venue": f"{home_abbr}_arena",
            }

            # Raw box-score stats → historical schema
            for bscore_key, stat_name in _BSCORE_STAT_MAP.items():
                row[f"home_{stat_name}"] = float(home_stats.get(bscore_key, 0) or 0)
                row[f"away_{stat_name}"] = float(away_stats.get(bscore_key, 0) or 0)

            # Derived advanced stats
            row["home_possessions"] = h_poss
            row["away_possessions"] = a_poss
            row["home_pointsPerPossession"] = h_pts / h_poss
            row["away_pointsPerPossession"] = a_pts / a_poss
            row["home_trueShooting"] = h_pts / max(1.0, 2 * (h_fga + 0.44 * h_fta))
            row["away_trueShooting"] = a_pts / max(1.0, 2 * (a_fga + 0.44 * a_fta))
            row["home_offensiveRating"] = 100.0 * h_pts / h_poss
            row["away_offensiveRating"] = 100.0 * a_pts / a_poss
            row["home_defensiveRating"] = 100.0 * a_pts / h_poss
            row["away_defensiveRating"] = 100.0 * h_pts / a_poss
            row["home_pace"] = h_poss
            row["away_pace"] = a_poss
            row["home_assistRate"] = float(home_stats.get("assists", 25)) / h_poss
            row["away_assistRate"] = float(away_stats.get("assists", 25)) / a_poss
            row["home_drebRate"] = h_dreb / max(1.0, h_dreb + a_oreb)
            row["away_drebRate"] = a_dreb / max(1.0, a_dreb + h_oreb)
            row["home_ftRate"] = h_fta / max(1.0, h_fga)
            row["away_ftRate"] = a_fta / max(1.0, a_fga)
            row["home_threePointRate"] = h_fg3a / max(1.0, h_fga)
            row["away_threePointRate"] = a_fg3a / max(1.0, a_fga)
            row["home_netRating"] = float(home_stats.get("plusMinusPoints", 0) or 0)
            row["away_netRating"] = float(away_stats.get("plusMinusPoints", 0) or 0)

            # Situational context
            row["wind_speed_mph"] = 0.0
            row["temperature_f"] = 72.0
            row["is_dome"] = 1.0
            row["altitude_feet"] = _ALTITUDE_MAP.get(home_abbr, 0.0)
            row["is_playoff"] = float("00424" in game_id or "00425" in game_id)

            # Market signals (0 at training time; overridden at inference)
            row["line_movement"] = 0.0
            row["public_over_pct"] = 0.5
            row["sharp_over_pct"] = 0.5
            row["ref_foul_rate"] = 0.0
            row["ump_walk_rate"] = 0.0

            # Injury counts derived from box score comments
            row["home_key_players_out"] = float(h_injury.key_players_out)
            row["away_key_players_out"] = float(a_injury.key_players_out)

            if h_injury.injured or a_injury.injured:
                logger.info(
                    "Injuries %s: %s out=[%s] | %s out=[%s]",
                    game_id,
                    home_abbr,
                    ", ".join(p.name for p in h_injury.injured if p.is_key) or "none",
                    away_abbr,
                    ", ".join(p.name for p in a_injury.injured if p.is_key) or "none",
                )

            rows.append(row)
            injury_records.append((home_abbr, "nba", h_injury.injured))
            injury_records.append((away_abbr, "nba", a_injury.injured))

        if not rows:
            return pd.DataFrame()

        self._persist_injury_reports(injury_records)
        df = pd.DataFrame(rows)
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True).dt.tz_localize(None)
        return self._compute_proxy_lines(df)

    def _fetch_nfl_games(self, target_date: date) -> pd.DataFrame:
        """Fetch completed NFL games for a specific date via ESPN scoreboard."""
        try:
            import httpx
            from proedge.pipeline.ingestion.espn_nfl_fetcher import (
                _fetch_scoreboard,
                _fetch_summary,
                _build_game_row,
                _compute_proxy_lines,
            )

            season = target_date.year if target_date.month >= 9 else target_date.year - 1
            season_start = date(season, 9, 1)
            days_in = max(0, (target_date - season_start).days)
            week = min(18, days_in // 7 + 1)

            rows: list[dict] = []
            with httpx.Client(follow_redirects=True) as client:
                events = _fetch_scoreboard(client, season, week)
                for ev in events:
                    comp = ev.get("competitions", [{}])[0]
                    if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                        continue
                    game_id = str(ev.get("id", ""))
                    if not game_id:
                        continue
                    # filter by date
                    ev_date_str = comp.get("date", "")[:10]
                    if ev_date_str and ev_date_str != target_date.isoformat():
                        continue
                    summary = _fetch_summary(client, game_id)
                    row = _build_game_row(ev, summary, season)
                    if row is not None:
                        rows.append(row)

            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)
            return _compute_proxy_lines(df)
        except Exception as exc:
            logger.warning("NFL daily fetch failed for %s: %s", target_date, exc)
            return pd.DataFrame()

    def _fetch_mlb_games(self, target_date: date) -> pd.DataFrame:
        """Fetch completed MLB games for a specific date via MLB Stats API."""
        try:
            import httpx
            from proedge.pipeline.ingestion.mlb_stats_fetcher import (
                _fetch_team_map,
                _fetch_schedule,
                _fetch_boxscore,
                _build_game_row,
                _compute_proxy_lines,
            )

            date_str = target_date.strftime("%Y-%m-%d")
            season = target_date.year

            rows: list[dict] = []
            with httpx.Client(follow_redirects=True) as client:
                team_map = _fetch_team_map(client)
                games = _fetch_schedule(client, date_str, date_str)
                for game in games:
                    game_pk = game.get("gamePk")
                    if not game_pk:
                        continue
                    boxscore = _fetch_boxscore(client, int(game_pk))
                    time.sleep(0.3)
                    row = _build_game_row(game, boxscore, team_map, season)
                    if row is not None:
                        rows.append(row)

            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)
            return _compute_proxy_lines(df)
        except Exception as exc:
            logger.warning("MLB daily fetch failed for %s: %s", target_date, exc)
            return pd.DataFrame()

    def _compute_proxy_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """NBA proxy bookmaker total using vectorized rolling averages from history."""
        if self.historical_path.exists():
            hist = pd.read_parquet(
                self.historical_path,
                columns=["home_team", "away_team", "home_score", "away_score", "game_date"],
            ).tail(500)
            home_view = hist[["game_date", "home_team", "home_score", "away_score"]].rename(
                columns={"home_team": "team", "home_score": "scored", "away_score": "allowed"}
            )
            away_view = hist[["game_date", "away_team", "away_score", "home_score"]].rename(
                columns={"away_team": "team", "away_score": "scored", "home_score": "allowed"}
            )
            all_apps = pd.concat([home_view, away_view]).sort_values("game_date")
            team_avgs = (
                all_apps.groupby("team")
                .tail(20)
                .groupby("team")
                .agg(scored=("scored", "mean"), allowed=("allowed", "mean"))
            )
            scored_mean = team_avgs["scored"].to_dict()
            allowed_mean = team_avgs["allowed"].to_dict()
        else:
            scored_mean: dict[str, float] = {}
            allowed_mean: dict[str, float] = {}

        rng = np.random.default_rng(int(datetime.now().timestamp()))
        lines: list[float] = []
        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            h_off = scored_mean.get(home, 113.0)
            h_def = allowed_mean.get(home, 113.0)
            a_off = scored_mean.get(away, 111.0)
            a_def = allowed_mean.get(away, 111.0)
            expected = (h_off + a_def) / 2 + 1.5 + (a_off + h_def) / 2
            line = expected + float(rng.normal(0, 3.0))
            lines.append(float(np.clip(round(line * 2) / 2, 180.0, 280.0)))

        df = df.copy()
        df["total_line"] = lines
        df["result_over"] = (df["total"] > df["total_line"]).astype(int)
        return df

    # ── Append ────────────────────────────────────────────────────────────────

    def _append_to_historical(self, new_games: pd.DataFrame) -> tuple[int, int]:
        """Appends new games to the historical parquet; returns (added, skipped)."""
        if self.historical_path.exists():
            hist = pd.read_parquet(self.historical_path)
            existing_ids = set(hist["game_id"].astype(str))
            fresh = new_games[~new_games["game_id"].astype(str).isin(existing_ids)]
            if fresh.empty:
                return 0, len(new_games)
            # Align columns — new games may have columns historical doesn't, and vice versa
            combined = pd.concat([hist, fresh], ignore_index=True, sort=False)
        else:
            combined = new_games
            fresh = new_games

        combined = combined.sort_values("game_date").reset_index(drop=True)
        combined.to_parquet(self.historical_path, index=False)
        return len(fresh), len(new_games) - len(fresh)

    # ── Cache management ──────────────────────────────────────────────────────

    def _clear_feature_cache(self):
        cleared = 0
        for f in self.features_dir.glob(f"{self.sport}_*.parquet"):
            f.unlink()
            cleared += 1
        if cleared:
            logger.info("Cleared %d stale feature cache files for %s", cleared, self.sport)

    def _count_new_since_last_retrain(self) -> int:
        """Rough proxy: games in historical added after the active model was trained."""
        from proedge.pipeline.models.registry import ModelRegistry

        try:
            meta = ModelRegistry().load_meta(self.sport)
            trained_at = datetime.fromisoformat(meta.get("trained_at", "2000-01-01"))
            dates = pd.read_parquet(self.historical_path, columns=["game_date"])["game_date"]
            return int((pd.to_datetime(dates) > trained_at).sum())
        except Exception:
            return 0

    def _persist_injury_reports(self, injury_records: list[tuple]) -> None:
        """Write InjuredPlayer entries to the injury_reports table."""
        # Deduplicate by team so the same team appearing home+away in the same day
        # doesn't produce duplicate rows.
        seen: set[str] = set()
        entries: list[tuple] = []
        for abbr, sport, players in injury_records:
            if abbr not in seen and players:
                entries.append((abbr, sport, players))
                seen.add(abbr)

        if not entries:
            return
        try:
            from proedge.db.models import InjuryReport
            from proedge.db.session import SyncSessionLocal

            with SyncSessionLocal() as session:
                reports = [
                    InjuryReport(
                        player_id=f"{sport}_{abbr}_{p.name.lower().replace(' ', '_')}"[:50],
                        player_name=p.name[:100],
                        team_id=abbr,
                        sport=sport,
                        status="out" if p.is_key else "questionable",
                        injury_type=(p.comment[:100] if p.comment else None),
                        impact_score=1.0 if p.is_key else 0.4,
                    )
                    for abbr, sport, players in entries
                    for p in players
                ]
                session.add_all(reports)
                session.commit()
            logger.info(
                "Persisted injury reports: %d players across %d teams",
                len(reports),
                len(entries),
            )
        except Exception as exc:
            logger.warning("Could not persist injury reports: %s", exc)

    def _settle_predictions(self, new_games: pd.DataFrame) -> int:
        """Settle open DB predictions for newly completed games. Returns count settled."""
        try:
            from sqlalchemy import select, update as sa_update
            from proedge.db.models import Game, Prediction
            from proedge.db.session import SyncSessionLocal

            settled = 0
            with SyncSessionLocal() as session:
                for _, row in new_games.iterrows():
                    ext_id = str(row["game_id"])
                    actual_total = float(row.get("total", row["home_score"] + row["away_score"]))

                    game = session.execute(
                        select(Game).where(Game.external_id == ext_id, Game.sport == self.sport)
                    ).scalar_one_or_none()
                    if game is None or game.total_line is None:
                        continue

                    result_over = actual_total > game.total_line
                    session.execute(
                        sa_update(Game)
                        .where(Game.id == game.id)
                        .values(
                            status="final",
                            home_score=int(row["home_score"]),
                            away_score=int(row["away_score"]),
                            result_over=result_over,
                        )
                    )

                    preds = (
                        session.execute(
                            select(Prediction).where(
                                Prediction.game_id == game.id,
                                Prediction.settled_at.is_(None),
                            )
                        )
                        .scalars()
                        .all()
                    )

                    now = datetime.now(timezone.utc)
                    for pred in preds:
                        is_correct = (pred.predicted_direction == "over") == result_over
                        session.execute(
                            sa_update(Prediction)
                            .where(Prediction.id == pred.id)
                            .values(
                                actual_total=actual_total,
                                closing_line=game.total_line,
                                clv=0.0,
                                is_correct=is_correct,
                                settled_at=now,
                            )
                        )
                        settled += 1

                session.commit()

            if settled:
                logger.info("Auto-settled %d predictions for %s", settled, self.sport.upper())
            return settled
        except Exception as exc:
            logger.warning("Could not auto-settle predictions: %s", exc)
            return 0

    def _retrain(self) -> dict:
        from proedge.pipeline.training.trainer import train

        logger.info("Auto-retraining %s after daily update", self.sport.upper())
        metrics = train(self.sport)
        try:
            from proedge.api.routers.predictions import _model_cache
            from proedge.pipeline.models.registry import ModelRegistry

            _model_cache[self.sport] = ModelRegistry().load(self.sport)
            logger.info("Model cache refreshed for %s after retrain", self.sport.upper())
        except Exception as exc:
            logger.warning("Could not refresh model cache after retrain: %s", exc)
        return metrics
