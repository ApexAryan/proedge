"""Matchup-adjusted features: opponent defensive quality and H2H history."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_matchup_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    opponent_window: int = 10,
) -> pd.DataFrame:
    """
    For each game, compute the opponent's defensive efficiency over the last
    N games (how much they allow vs. league average).  This produces
    matchup-adjusted versions of every stat.
    """
    df = df.sort_values("game_date").copy()

    # Per-row expanding league average — only uses games played before each row.
    # Avoids leaking future scoring levels into the ratio denominator.
    expanding_league_avg: dict[str, pd.Series] = {}
    for col in stat_cols:
        home_col, away_col = f"home_{col}", f"away_{col}"
        cols_present = [c for c in [home_col, away_col] if c in df.columns]
        if cols_present:
            game_mean = df[cols_present].mean(axis=1)
            expanding = game_mean.expanding().mean().shift(1)
            expanding_league_avg[col] = expanding.fillna(game_mean.mean())

    # Opponent defensive rating relative to expanding league average
    for col in stat_cols:
        avg_series = expanding_league_avg.get(col, pd.Series(1.0, index=df.index))
        df[f"opp_def_{col}_ratio"] = 1.0
        for team in df["home_team"].unique():
            mask = df["home_team"] == team
            away_col = f"away_{col}"
            if away_col in df.columns:
                allowed_mean = (
                    df.loc[mask, away_col].shift(1).rolling(opponent_window, min_periods=1).mean()
                )
                safe_avg = avg_series.loc[mask].replace(0, np.nan).fillna(1.0)
                df.loc[mask, f"opp_def_{col}_ratio"] = allowed_mean / safe_avg

    # H2H head-to-head over rate
    df["h2h_over_rate"] = _compute_h2h_over_rate(df)
    df["h2h_avg_total"] = _compute_h2h_avg_total(df)

    return df


def _compute_h2h_over_rate(df: pd.DataFrame) -> pd.Series:
    """Historical over% for this specific matchup (home vs away teams)."""
    rate = pd.Series(np.nan, index=df.index)
    df_sorted = df.sort_values("game_date")

    for idx, row in df_sorted.iterrows():
        past = df_sorted[
            (df_sorted["game_date"] < row["game_date"])
            & (
                (
                    (df_sorted["home_team"] == row["home_team"])
                    & (df_sorted["away_team"] == row["away_team"])
                )
                | (
                    (df_sorted["home_team"] == row["away_team"])
                    & (df_sorted["away_team"] == row["home_team"])
                )
            )
        ]
        if len(past) >= 2 and "result_over" in past.columns:
            rate[idx] = past["result_over"].mean()
    return rate.fillna(0.5)


def _compute_h2h_avg_total(df: pd.DataFrame) -> pd.Series:
    avg_total = pd.Series(np.nan, index=df.index)
    df_sorted = df.sort_values("game_date")

    for idx, row in df_sorted.iterrows():
        past = df_sorted[
            (df_sorted["game_date"] < row["game_date"])
            & (
                (
                    (df_sorted["home_team"] == row["home_team"])
                    & (df_sorted["away_team"] == row["away_team"])
                )
                | (
                    (df_sorted["home_team"] == row["away_team"])
                    & (df_sorted["away_team"] == row["home_team"])
                )
            )
        ]
        if len(past) >= 2 and "total" in past.columns:
            avg_total[idx] = past["total"].mean()
    return avg_total.fillna(df.get("total_line", pd.Series(np.nan, index=df.index)))


def add_division_flag(df: pd.DataFrame, division_map: dict[str, str]) -> pd.DataFrame:
    """Flag games within the same division."""
    df = df.copy()
    df["is_division_game"] = (
        df["home_team"].map(division_map) == df["away_team"].map(division_map)
    ).astype(int)
    return df
