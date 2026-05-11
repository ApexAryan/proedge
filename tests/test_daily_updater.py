"""Unit tests for DailyUpdater — no network, no real DB required."""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from proedge.pipeline.ingestion.daily_updater import (
    DailyUpdater, UpdateResult, _MIN_NEW_GAMES_TO_RETRAIN,
)


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def updater(tmp_data_dir):
    u = DailyUpdater("nba", data_dir=str(tmp_data_dir), auto_retrain=False)
    u.features_dir.mkdir(parents=True, exist_ok=True)
    return u


def _make_game_df(n: int = 5, sport: str = "nba") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({
            "game_id": f"g{i:04d}",
            "sport": sport,
            "season": 2024,
            "game_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "home_team": "BOS",
            "away_team": "LAL",
            "home_score": int(rng.integers(95, 130)),
            "away_score": int(rng.integers(95, 130)),
            "total": 220.0,
            "total_line": 222.5,
            "result_over": 0,
            "venue": "BOS_arena",
        })
    return pd.DataFrame(rows)


# ── _append_to_historical ─────────────────────────────────────────────────────

def test_append_creates_parquet_when_none_exists(updater, tmp_data_dir):
    df = _make_game_df(3)
    added, skipped = updater._append_to_historical(df)
    assert added == 3
    assert skipped == 0
    assert updater.historical_path.exists()


def test_append_deduplicates_on_game_id(updater):
    df = _make_game_df(4)
    updater._append_to_historical(df)
    added, skipped = updater._append_to_historical(df)
    assert added == 0
    assert skipped == 4


def test_append_adds_only_new_games(updater):
    first_batch = _make_game_df(3)
    updater._append_to_historical(first_batch)

    second_batch = _make_game_df(6)  # first 3 overlap, 3 are new
    added, skipped = updater._append_to_historical(second_batch)
    assert added == 3
    assert skipped == 3


def test_append_preserves_sort_order(updater):
    df = _make_game_df(5)
    updater._append_to_historical(df)
    result = pd.read_parquet(updater.historical_path)
    dates = pd.to_datetime(result["game_date"])
    assert list(dates) == sorted(dates)


# ── _clear_feature_cache ──────────────────────────────────────────────────────

def test_clear_cache_removes_sport_parquets(updater):
    cache_file = updater.features_dir / "nba_features_v1.parquet"
    cache_file.touch()
    other_file = updater.features_dir / "nfl_features_v1.parquet"
    other_file.touch()

    updater._clear_feature_cache()

    assert not cache_file.exists()
    assert other_file.exists()  # different sport — not touched


def test_clear_cache_no_op_when_empty(updater):
    updater._clear_feature_cache()  # should not raise


# ── _compute_proxy_lines ─────────────────────────────────────────────────────

def test_proxy_lines_adds_columns(updater):
    df = _make_game_df(5)
    result = updater._compute_proxy_lines(df)
    assert "total_line" in result.columns
    assert "result_over" in result.columns


def test_proxy_lines_within_clip_range(updater):
    df = _make_game_df(10)
    result = updater._compute_proxy_lines(df)
    assert (result["total_line"] >= 180.0).all()
    assert (result["total_line"] <= 280.0).all()


def test_proxy_lines_uses_historical_when_available(updater):
    hist = _make_game_df(20)
    hist["home_score"] = 120
    hist["away_score"] = 110
    hist["total"] = 230.0
    updater._append_to_historical(hist)

    new_df = _make_game_df(2)
    result = updater._compute_proxy_lines(new_df)
    assert "total_line" in result.columns
    assert not result["total_line"].isna().any()


# ── _count_new_since_last_retrain ────────────────────────────────────────────

def test_count_new_returns_zero_when_no_historical(updater):
    with patch("proedge.pipeline.models.registry.ModelRegistry") as MockReg:
        MockReg.return_value.load_meta.side_effect = Exception("no model")
        assert updater._count_new_since_last_retrain() == 0


def test_count_new_counts_games_after_trained_at(updater):
    df = _make_game_df(6)
    updater._append_to_historical(df)

    meta = {"trained_at": "2024-01-04T00:00:00"}
    with patch("proedge.pipeline.models.registry.ModelRegistry") as MockReg:
        MockReg.return_value.load_meta.return_value = meta
        count = updater._count_new_since_last_retrain()
    assert count == 2  # games 5 and 6 are after 2024-01-04


# ── run() integration (mocked fetch) ─────────────────────────────────────────

def test_run_returns_update_result_no_games(updater):
    with patch.object(updater, "_fetch_completed_games", return_value=pd.DataFrame()):
        result = updater.run(date(2026, 4, 25))
    assert isinstance(result, UpdateResult)
    assert result.games_found == 0
    assert result.games_added == 0
    assert result.error is None


def test_run_adds_games_and_clears_cache(updater):
    df = _make_game_df(3)
    cache_file = updater.features_dir / "nba_cached.parquet"
    cache_file.touch()

    with patch.object(updater, "_fetch_completed_games", return_value=df):
        result = updater.run(date(2026, 4, 25))

    assert result.games_added == 3
    assert not cache_file.exists()


def test_run_records_error_on_exception(updater):
    with patch.object(updater, "_fetch_completed_games", side_effect=RuntimeError("boom")):
        result = updater.run(date(2026, 4, 25))
    assert result.error == "boom"


# ── UpdateResult dataclass ────────────────────────────────────────────────────

def test_update_result_defaults():
    r = UpdateResult(sport="nba", date="2026-04-25")
    assert r.games_found == 0
    assert r.retrain_triggered is False
    assert r.error is None


# ── _fetch_completed_games unsupported sport ──────────────────────────────────

def test_fetch_completed_games_unsupported_sport(updater):
    result = updater._fetch_completed_games.__func__(
        DailyUpdater("hockey", data_dir="/tmp"), date(2026, 4, 25)
    )
    assert result.empty


# ── run() auto-retrain path ───────────────────────────────────────────────────

def test_run_triggers_retrain_when_threshold_met(tmp_data_dir):
    u = DailyUpdater("nba", data_dir=str(tmp_data_dir), auto_retrain=True)
    u.features_dir.mkdir(parents=True, exist_ok=True)
    df = _make_game_df(_MIN_NEW_GAMES_TO_RETRAIN + 5)

    with (
        patch.object(u, "_fetch_completed_games", return_value=df),
        patch.object(u, "_settle_predictions", return_value=0),
        patch.object(u, "_count_new_since_last_retrain",
                     return_value=_MIN_NEW_GAMES_TO_RETRAIN + 5),
        patch.object(u, "_retrain", return_value={"accuracy": 0.56}) as mock_retrain,
    ):
        result = u.run(date(2026, 4, 25))

    assert result.retrain_triggered is True
    assert result.retrain_metrics == {"accuracy": 0.56}
    mock_retrain.assert_called_once()


def test_run_no_retrain_when_below_threshold(tmp_data_dir):
    u = DailyUpdater("nba", data_dir=str(tmp_data_dir), auto_retrain=True)
    u.features_dir.mkdir(parents=True, exist_ok=True)
    df = _make_game_df(3)

    with (
        patch.object(u, "_fetch_completed_games", return_value=df),
        patch.object(u, "_settle_predictions", return_value=0),
        patch.object(u, "_count_new_since_last_retrain", return_value=5),
        patch.object(u, "_retrain") as mock_retrain,
    ):
        result = u.run(date(2026, 4, 25))

    assert result.retrain_triggered is False
    mock_retrain.assert_not_called()


# ── _retrain() ────────────────────────────────────────────────────────────────

def test_retrain_calls_train_and_returns_metrics(updater):
    metrics = {"accuracy": 0.57, "auc": 0.61, "log_loss": 0.67}
    with (
        patch("proedge.pipeline.training.trainer.train", return_value=metrics) as mock_train,
        patch("proedge.pipeline.models.registry.ModelRegistry"),
    ):
        result = updater._retrain()

    mock_train.assert_called_once_with("nba")
    assert result == metrics


def test_retrain_model_cache_refresh_failure_swallowed(updater):
    """If the model-cache refresh fails after retrain, the error must be swallowed."""
    with (
        patch("proedge.pipeline.training.trainer.train", return_value={"accuracy": 0.55}),
        patch("proedge.api.routers.predictions._model_cache",
              side_effect=Exception("import fail")),
    ):
        result = updater._retrain()  # must not raise
    assert "accuracy" in result


# ── _fetch_nba_games() ────────────────────────────────────────────────────────

def _bx_stats(pts: int = 112) -> dict:
    return {
        "points": pts,
        "fieldGoalsMade": 42, "fieldGoalsAttempted": 85,
        "fieldGoalsPercentage": 0.494,
        "threePointersMade": 14, "threePointersAttempted": 33,
        "threePointersPercentage": 0.424,
        "freeThrowsMade": 14, "freeThrowsAttempted": 22,
        "freeThrowsPercentage": 0.636,
        "reboundsOffensive": 10, "reboundsDefensive": 34, "reboundsTotal": 44,
        "assists": 25, "steals": 8, "blocks": 4,
        "turnovers": 14, "foulsPersonal": 20, "plusMinusPoints": 4,
    }


def _mock_nba_api(game_status: int = 3, game_id: str = "0021234567",
                  bx_raises: Exception | None = None):
    mock_sb = MagicMock()
    mock_sb.get_dict.return_value = {
        "scoreboard": {"games": [{
            "gameId": game_id,
            "gameStatus": game_status,
            "gameTimeUTC": "2026-04-25T00:00:00Z",
        }]}
    }

    mock_bx = MagicMock()
    if bx_raises:
        mock_bx.get_dict.side_effect = bx_raises
    else:
        mock_bx.get_dict.return_value = {
            "boxScoreTraditional": {
                "homeTeam": {
                    "teamTricode": "BOS",
                    "statistics": _bx_stats(112),
                    "players": [],
                },
                "awayTeam": {
                    "teamTricode": "LAL",
                    "statistics": _bx_stats(108),
                    "players": [],
                },
            }
        }
    return mock_sb, mock_bx


def test_fetch_nba_games_returns_dataframe(updater):
    mock_sb, mock_bx = _mock_nba_api()
    with (
        patch("nba_api.stats.endpoints.scoreboardv3.ScoreboardV3",
              return_value=mock_sb),
        patch("nba_api.stats.endpoints.boxscoretraditionalv3.BoxScoreTraditionalV3",
              return_value=mock_bx),
        patch.object(updater, "_persist_injury_reports"),
        patch("time.sleep"),
    ):
        result = updater._fetch_nba_games(date(2026, 4, 25))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "total_line" in result.columns
    assert result.iloc[0]["home_team"] == "BOS"


def test_fetch_nba_games_empty_scoreboard_returns_empty(updater):
    mock_sb = MagicMock()
    mock_sb.get_dict.return_value = {"scoreboard": {"games": []}}
    with patch("nba_api.stats.endpoints.scoreboardv3.ScoreboardV3",
               return_value=mock_sb):
        result = updater._fetch_nba_games(date(2026, 4, 25))
    assert result.empty


def test_fetch_nba_games_non_final_game_skipped(updater):
    mock_sb, mock_bx = _mock_nba_api(game_status=2)  # 2 = in-progress
    with (
        patch("nba_api.stats.endpoints.scoreboardv3.ScoreboardV3",
              return_value=mock_sb),
        patch("nba_api.stats.endpoints.boxscoretraditionalv3.BoxScoreTraditionalV3",
              return_value=mock_bx),
        patch("time.sleep"),
    ):
        result = updater._fetch_nba_games(date(2026, 4, 25))
    assert result.empty


def test_fetch_nba_games_boxscore_exception_skips_game(updater):
    mock_sb, mock_bx = _mock_nba_api(bx_raises=RuntimeError("timeout"))
    with (
        patch("nba_api.stats.endpoints.scoreboardv3.ScoreboardV3",
              return_value=mock_sb),
        patch("nba_api.stats.endpoints.boxscoretraditionalv3.BoxScoreTraditionalV3",
              return_value=mock_bx),
        patch("time.sleep"),
    ):
        result = updater._fetch_nba_games(date(2026, 4, 25))
    assert result.empty


# ── _fetch_nfl_games() ────────────────────────────────────────────────────────

def _nfl_event(game_id: str = "401547656", target_date: str = "2026-01-15",
               status: str = "STATUS_FINAL") -> dict:
    return {
        "id": game_id,
        "competitions": [{
            "status": {"type": {"name": status}},
            "date": f"{target_date}T18:00:00Z",
        }],
    }


def _nfl_row() -> dict:
    return {
        "game_id": "401547656", "sport": "nfl", "season": 2025,
        "game_date": "2026-01-15", "home_team": "KC", "away_team": "BUF",
        "home_score": 27, "away_score": 24, "total": 51, "total_line": 48.5,
        "result_over": 1, "venue": "KC_stadium",
    }


def _mock_httpx_client():
    mock_client = MagicMock()
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return mock_cls, mock_client


def test_fetch_nfl_games_returns_dataframe(updater):
    event = _nfl_event(target_date="2026-01-15")
    row = _nfl_row()
    result_df = pd.DataFrame([row])
    result_df["total_line"] = 48.5
    result_df["result_over"] = 1

    mock_cls, _ = _mock_httpx_client()
    with (
        patch("httpx.Client", mock_cls),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._fetch_scoreboard",
              return_value=[event]),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._fetch_summary",
              return_value={}),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._build_game_row",
              return_value=row),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._compute_proxy_lines",
              return_value=result_df),
    ):
        result = updater._fetch_nfl_games(date(2026, 1, 15))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_fetch_nfl_games_non_final_event_skipped(updater):
    event = _nfl_event(status="STATUS_IN_PROGRESS")
    mock_cls, _ = _mock_httpx_client()
    with (
        patch("httpx.Client", mock_cls),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._fetch_scoreboard",
              return_value=[event]),
    ):
        result = updater._fetch_nfl_games(date(2026, 1, 15))
    assert result.empty


def test_fetch_nfl_games_exception_returns_empty(updater):
    with patch("httpx.Client", side_effect=RuntimeError("network")):
        result = updater._fetch_nfl_games(date(2026, 1, 15))
    assert result.empty


def test_fetch_nfl_games_build_row_none_skipped(updater):
    event = _nfl_event(target_date="2026-01-15")
    mock_cls, _ = _mock_httpx_client()
    with (
        patch("httpx.Client", mock_cls),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._fetch_scoreboard",
              return_value=[event]),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._fetch_summary",
              return_value={}),
        patch("proedge.pipeline.ingestion.espn_nfl_fetcher._build_game_row",
              return_value=None),
    ):
        result = updater._fetch_nfl_games(date(2026, 1, 15))
    assert result.empty


# ── _fetch_mlb_games() ────────────────────────────────────────────────────────

def _mlb_row() -> dict:
    return {
        "game_id": "716789", "sport": "mlb", "season": 2026,
        "game_date": "2026-04-25", "home_team": "NYY", "away_team": "BOS",
        "home_score": 5, "away_score": 3, "total": 8, "total_line": 8.5,
        "result_over": 0, "venue": "NYY_stadium",
    }


def test_fetch_mlb_games_returns_dataframe(updater):
    row = _mlb_row()
    result_df = pd.DataFrame([row])
    result_df["total_line"] = 8.5
    result_df["result_over"] = 0

    mock_cls, _ = _mock_httpx_client()
    with (
        patch("httpx.Client", mock_cls),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_team_map",
              return_value={"147": "NYY", "111": "BOS"}),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_schedule",
              return_value=[{"gamePk": 716789}]),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_boxscore",
              return_value={}),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._build_game_row",
              return_value=row),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._compute_proxy_lines",
              return_value=result_df),
        patch("time.sleep"),
    ):
        result = updater._fetch_mlb_games(date(2026, 4, 25))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_fetch_mlb_games_no_game_pk_skipped(updater):
    mock_cls, _ = _mock_httpx_client()
    with (
        patch("httpx.Client", mock_cls),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_team_map",
              return_value={}),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_schedule",
              return_value=[{}]),  # no gamePk
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_boxscore",
              return_value={}),
    ):
        result = updater._fetch_mlb_games(date(2026, 4, 25))
    assert result.empty


def test_fetch_mlb_games_exception_returns_empty(updater):
    with patch("httpx.Client", side_effect=RuntimeError("network")):
        result = updater._fetch_mlb_games(date(2026, 4, 25))
    assert result.empty


def test_fetch_mlb_games_build_row_none_skipped(updater):
    mock_cls, _ = _mock_httpx_client()
    with (
        patch("httpx.Client", mock_cls),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_team_map",
              return_value={}),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_schedule",
              return_value=[{"gamePk": 999}]),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._fetch_boxscore",
              return_value={}),
        patch("proedge.pipeline.ingestion.mlb_stats_fetcher._build_game_row",
              return_value=None),
    ):
        result = updater._fetch_mlb_games(date(2026, 4, 25))
    assert result.empty
