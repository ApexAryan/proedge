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

    # Build opponent defensive allowed averages
    opp_allowed: dict[str, dict[str, float]] = {}  # team → {stat: rolling_allowed}

    for team in df["away_team"].unique():
        opp_allowed[team] = {}
        mask_home = df["home_team"] == team
        for col in stat_cols:
            away_col = f"away_{col}"
            if away_col in df.columns:
                allowed = df.loc[mask_home, away_col].shift(1).rolling(
                    opponent_window, min_periods=1
                ).mean()
                opp_allowed[team][col] = allowed

    # League averages per stat
    league_avg: dict[str, float] = {}
    for col in stat_cols:
        home_col, away_col = f"home_{col}", f"away_{col}"
        vals = []
        if home_col in df.columns:
            vals.append(df[home_col])
        if away_col in df.columns:
            vals.append(df[away_col])
        if vals:
            league_avg[col] = float(pd.concat(vals).mean())

    # Opponent defensive rating relative to league average
    for col in stat_cols:
        avg = league_avg.get(col, 1.0) or 1.0
        df[f"opp_def_{col}_ratio"] = 1.0  # default: league average
        for team in df["home_team"].unique():
            mask = df["home_team"] == team
            away_col = f"away_{col}"
            if away_col in df.columns:
                allowed_mean = df.loc[mask, away_col].shift(1).rolling(
                    opponent_window, min_periods=1
                ).mean()
                df.loc[mask, f"opp_def_{col}_ratio"] = allowed_mean / avg

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
