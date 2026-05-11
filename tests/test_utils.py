"""Tests for proedge.pipeline.ingestion.utils — safe_int, safe_float, compute_proxy_lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proedge.pipeline.ingestion.utils import compute_proxy_lines, safe_float, safe_int


# ---------------------------------------------------------------------------
# safe_int
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_integer_string(self):
        assert safe_int("42") == 42

    def test_float_string_returns_default(self):
        # int("3.9") raises ValueError — safe_int returns the default, not 3
        assert safe_int("3.9") == 0

    def test_plain_int(self):
        assert safe_int(7) == 7

    def test_invalid_returns_default(self):
        assert safe_int("abc") == 0

    def test_none_returns_default(self):
        assert safe_int(None) == 0

    def test_empty_string_returns_default(self):
        assert safe_int("") == 0

    def test_custom_default(self):
        assert safe_int("bad", default=99) == 99

    def test_negative(self):
        assert safe_int("-5") == -5

    def test_whitespace_stripped(self):
        assert safe_int("  8  ") == 8


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_float_string(self):
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert safe_float("10") == pytest.approx(10.0)

    def test_plain_float(self):
        assert safe_float(2.5) == pytest.approx(2.5)

    def test_invalid_returns_default(self):
        assert safe_float("xyz") == pytest.approx(0.0)

    def test_none_returns_default(self):
        assert safe_float(None) == pytest.approx(0.0)

    def test_custom_default(self):
        assert safe_float("bad", default=1.5) == pytest.approx(1.5)

    def test_negative(self):
        assert safe_float("-2.7") == pytest.approx(-2.7)

    def test_whitespace_stripped(self):
        assert safe_float("  4.0  ") == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_proxy_lines
# ---------------------------------------------------------------------------


def _sport_df(n: int = 20, home_score: int = 112, away_score: int = 108) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    teams = ["BOS", "LAL", "GSW", "MIA"]
    rows = []
    for i in range(n):
        h = teams[i % len(teams)]
        a = teams[(i + 1) % len(teams)]
        rows.append(
            {
                "home_team": h,
                "away_team": a,
                "home_score": home_score + int(rng.integers(-5, 5)),
                "away_score": away_score + int(rng.integers(-5, 5)),
            }
        )
    df = pd.DataFrame(rows)
    df["total"] = df["home_score"] + df["away_score"]
    return df


class TestComputeProxyLines:
    _NBA_KWARGS = dict(
        clip_lo=180.0,
        clip_hi=280.0,
        home_off_default=113.0,
        home_def_default=113.0,
        away_off_default=111.0,
        away_def_default=111.0,
        noise_std=3.0,
        home_advantage=1.5,
    )
    _NFL_KWARGS = dict(
        clip_lo=28.0,
        clip_hi=80.0,
        home_off_default=23.0,
        home_def_default=21.0,
        away_off_default=21.0,
        away_def_default=23.0,
        noise_std=3.5,
    )

    def test_adds_total_line_column(self):
        result = compute_proxy_lines(_sport_df(), **self._NBA_KWARGS)
        assert "total_line" in result.columns

    def test_adds_result_over_column(self):
        result = compute_proxy_lines(_sport_df(), **self._NBA_KWARGS)
        assert "result_over" in result.columns

    def test_nba_lines_within_clip_range(self):
        result = compute_proxy_lines(_sport_df(40), **self._NBA_KWARGS)
        assert result["total_line"].between(180.0, 280.0).all()

    def test_nfl_lines_within_clip_range(self):
        df = _sport_df(20, home_score=24, away_score=20)
        result = compute_proxy_lines(df, **self._NFL_KWARGS)
        assert result["total_line"].between(28.0, 80.0).all()

    def test_result_over_is_binary(self):
        result = compute_proxy_lines(_sport_df(30), **self._NBA_KWARGS)
        assert set(result["result_over"].unique()).issubset({0, 1})

    def test_result_over_matches_total_vs_line(self):
        result = compute_proxy_lines(_sport_df(20), **self._NBA_KWARGS)
        expected = (result["total"] > result["total_line"]).astype(int)
        pd.testing.assert_series_equal(result["result_over"], expected, check_names=False)

    def test_original_df_not_mutated(self):
        df = _sport_df(10)
        cols_before = set(df.columns)
        compute_proxy_lines(df, **self._NBA_KWARGS)
        assert set(df.columns) == cols_before

    def test_rolling_history_updates_per_row(self):
        """Lines should vary as rolling averages accumulate — not all identical."""
        result = compute_proxy_lines(_sport_df(40), **self._NBA_KWARGS)
        assert result["total_line"].nunique() > 5

    def test_single_row_uses_defaults(self):
        """With no prior history, defaults are used — line should be near default sum."""
        df = _sport_df(1)
        result = compute_proxy_lines(df, **self._NBA_KWARGS)
        assert len(result) == 1
        # default expected ≈ (113+113)/2 + (111+113)/2 + 1.5 ≈ 225.5 ± noise
        assert 180.0 <= result["total_line"].iloc[0] <= 280.0
