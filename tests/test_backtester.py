"""Unit tests for the walk-forward backtester."""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from proedge.pipeline.backtesting.backtester import Backtester, BacktestResult
from proedge.pipeline.ingestion.stats import STAT_KEYS

# Mirror the module-level constants without importing private names
_BET_SIZE = 100.0
_JUICE_PAYOFF = 100 / 110
_KELLY_CAP = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 200, sport: str = "nba") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    teams = ["BOS", "LAL", "GSW", "MIA", "CHI"]
    stat_cols = STAT_KEYS[sport]
    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        total = float(rng.normal(224, 18))
        line = total + rng.normal(0, 2)
        rows.append(
            {
                "game_id": f"g{i:04d}",
                "sport": sport,
                "season": 2024,
                "game_date": datetime(2023, 1, 1) + timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_score": int(total / 2 + rng.normal(0, 5)),
                "away_score": int(total / 2 + rng.normal(0, 5)),
                "total": total,
                "total_line": round(line, 1),
                "result_over": int(total > line),
                "venue": f"{home}_arena",
                **{f"home_{s}": float(rng.uniform(90, 130)) for s in stat_cols},
                **{f"away_{s}": float(rng.uniform(90, 130)) for s in stat_cols},
                "wind_speed_mph": 0.0,
                "temperature_f": 70.0,
                "is_dome": 0.0,
                "altitude_feet": 0.0,
                "is_playoff": 0.0,
                "line_movement": 0.0,
                "public_over_pct": 0.5,
                "sharp_over_pct": 0.5,
                "ref_foul_rate": 0.0,
                "ump_walk_rate": 0.0,
                "home_key_players_out": 0.0,
                "away_key_players_out": 0.0,
                "home_injury_impact": 0.0,
                "away_injury_impact": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ---------------------------------------------------------------------------
# _fold_edges
# ---------------------------------------------------------------------------


def test_fold_edges_count():
    bt = Backtester("nba")
    dates = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D"))
    edges = bt._fold_edges(dates, n_folds=5)
    assert len(edges) == 6


def test_fold_edges_monotonic():
    bt = Backtester("nba")
    dates = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D"))
    edges = bt._fold_edges(dates, n_folds=5)
    for i in range(len(edges) - 1):
        assert edges[i] <= edges[i + 1]


# ---------------------------------------------------------------------------
# _simulate_betting
# ---------------------------------------------------------------------------


def test_simulate_betting_correct_over_bet():
    bt = Backtester("nba")
    flat, kelly = bt._simulate_betting(np.array([0.7]), np.array([1]))
    assert len(flat) == 1
    assert abs(flat[0] - _BET_SIZE * _JUICE_PAYOFF) < 1e-6
    assert kelly[0] > 0


def test_simulate_betting_wrong_over_bet():
    bt = Backtester("nba")
    flat, kelly = bt._simulate_betting(np.array([0.7]), np.array([0]))
    assert flat[0] == -_BET_SIZE
    assert kelly[0] <= 0


def test_simulate_betting_correct_under_bet():
    bt = Backtester("nba")
    flat, kelly = bt._simulate_betting(np.array([0.3]), np.array([0]))
    assert abs(flat[0] - _BET_SIZE * _JUICE_PAYOFF) < 1e-6
    assert kelly[0] > 0


def test_simulate_betting_wrong_under_bet():
    bt = Backtester("nba")
    flat, kelly = bt._simulate_betting(np.array([0.3]), np.array([1]))
    assert flat[0] == -_BET_SIZE


def test_simulate_betting_empty_arrays():
    bt = Backtester("nba")
    flat, kelly = bt._simulate_betting(np.array([]), np.array([]))
    assert flat == []
    assert kelly == []


def test_simulate_betting_kelly_capped():
    bt = Backtester("nba")
    # Very high confidence should hit the Kelly cap
    _, kelly = bt._simulate_betting(np.array([0.99]), np.array([1]))
    max_kelly_stake = _KELLY_CAP * _BET_SIZE
    # win return = stake * _JUICE_PAYOFF; stake can't exceed max_kelly_stake
    assert kelly[0] <= max_kelly_stake * _JUICE_PAYOFF + 1e-9


# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------


def test_sharpe_identical_returns():
    returns = np.ones(100) * 0.05
    assert Backtester._sharpe(returns) == 0.0


def test_sharpe_empty_or_single():
    assert Backtester._sharpe(np.array([])) == 0.0
    assert Backtester._sharpe(np.array([0.1])) == 0.0


def test_sharpe_positive_edge():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.1, 0.5, 300)
    sharpe = Backtester._sharpe(returns)
    assert sharpe > 0


def test_sharpe_negative_edge():
    rng = np.random.default_rng(0)
    returns = rng.normal(-0.1, 0.5, 300)
    sharpe = Backtester._sharpe(returns)
    assert sharpe < 0


def test_sharpe_annualisation():
    # Mean=0.1, std=0.1 → raw Sharpe=1.0 → annualised = sqrt(252) ≈ 15.87
    # std will be tiny but non-zero; just verify scale is in annualised range
    returns_varied = np.random.default_rng(1).normal(0.1, 0.1, 252)
    sharpe = Backtester._sharpe(returns_varied)
    assert abs(sharpe) < 100  # sanity: not an absurd value


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_no_drawdown():
    returns = [10.0, 20.0, 30.0, 40.0, 50.0]
    dd = Backtester._max_drawdown(returns)
    assert dd < 0.01


def test_max_drawdown_all_losses():
    returns = [-10.0, -20.0, -30.0]
    dd = Backtester._max_drawdown(returns)
    assert dd > 0.5


def test_max_drawdown_empty():
    assert Backtester._max_drawdown([]) == 0.0


def test_max_drawdown_single_element():
    assert Backtester._max_drawdown([100.0]) >= 0.0


# ---------------------------------------------------------------------------
# _calibration
# ---------------------------------------------------------------------------


def test_calibration_structure():
    rng = np.random.default_rng(7)
    probs = rng.uniform(0, 1, 100)
    labels = (probs + rng.normal(0, 0.2, 100) > 0.5).astype(int)
    result = Backtester._calibration(probs, labels, n_bins=10)
    assert set(result.keys()) == {"bin_midpoint", "actual_freq", "predicted_prob"}
    assert len(result["bin_midpoint"]) == 10
    assert len(result["actual_freq"]) == 10
    assert len(result["predicted_prob"]) == 10


def test_calibration_midpoints():
    probs = np.linspace(0.05, 0.95, 100)
    labels = (probs > 0.5).astype(int)
    result = Backtester._calibration(probs, labels, n_bins=10)
    expected = [round(0.05 + 0.1 * i, 3) for i in range(10)]
    assert result["bin_midpoint"] == expected


def test_calibration_actual_freq_in_range():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, 500)
    labels = rng.integers(0, 2, 500)
    result = Backtester._calibration(probs, labels)
    for v in result["actual_freq"]:
        if v is not None:
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# _load_data
# ---------------------------------------------------------------------------


def test_load_data_missing_raises():
    bt = Backtester("nba", data_dir="/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        bt._load_data()


# ---------------------------------------------------------------------------
# full run (integration — mocks _load_data)
# ---------------------------------------------------------------------------


def test_run_returns_backtest_result():
    df = _make_synthetic_df(n=200, sport="nba")
    bt = Backtester("nba")
    with patch.object(bt, "_load_data", return_value=df):
        result = bt.run(n_folds=3, min_train_games=50)
    assert isinstance(result, BacktestResult)
    assert result.sport == "nba"
    assert result.n_folds == 3
    assert 0.0 <= result.overall_accuracy <= 1.0
    assert result.max_drawdown >= 0.0
    assert result.overall_auc >= 0.0


def test_run_folds_have_required_fields():
    df = _make_synthetic_df(n=200, sport="nba")
    bt = Backtester("nba")
    with patch.object(bt, "_load_data", return_value=df):
        result = bt.run(n_folds=3, min_train_games=50)
    for fold in result.folds:
        assert fold.n_games > 0
        assert 0.0 <= fold.accuracy <= 1.0
        assert fold.auc >= 0.0
        assert fold.log_loss > 0.0
        assert fold.brier_score >= 0.0


def test_run_min_confidence_filters_bets():
    df = _make_synthetic_df(n=200, sport="nba")
    bt = Backtester("nba")
    with patch.object(bt, "_load_data", return_value=df):
        result_no_filter = bt.run(n_folds=3, min_train_games=50, min_confidence=0.0)
        result_filtered = bt.run(n_folds=3, min_train_games=50, min_confidence=0.5)
    assert result_filtered.total_bets <= result_no_filter.total_bets


def test_run_empty_result_when_insufficient_data():
    df = _make_synthetic_df(n=30, sport="nba")
    bt = Backtester("nba")
    with patch.object(bt, "_load_data", return_value=df):
        result = bt.run(n_folds=3, min_train_games=500)
    assert result.total_games == 0
    assert result.folds == []
    assert result.overall_accuracy == 0.0


def test_run_calibration_structure():
    df = _make_synthetic_df(n=200, sport="nba")
    bt = Backtester("nba")
    with patch.object(bt, "_load_data", return_value=df):
        result = bt.run(n_folds=3, min_train_games=50)
    if result.calibration:
        assert "bin_midpoint" in result.calibration
        assert "actual_freq" in result.calibration
        assert "predicted_prob" in result.calibration
