"""Rolling and EMA stat features — the core of the 200+ signal feature store."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [3, 5, 10, 20]
EMA_ALPHAS = [0.3, 0.5]


def add_rolling_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    team_col: str = "home_team",
    date_col: str = "game_date",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Rolling mean/std/max/min and EMA per team per stat column.
    Pre-allocates numpy arrays and fills by team, then concat-appends once
    to avoid DataFrame fragmentation.
    """
    df = df.sort_values(date_col).copy()
    n = len(df)
    full_index = df.index
    pos_map = {idx: pos for pos, idx in enumerate(full_index)}

    stat_cols_present = [c for c in stat_cols if c in df.columns]

    new_cols: dict[str, np.ndarray] = {}
    for col in stat_cols_present:
        for w in WINDOWS:
            for agg in ("mean", "std", "max", "min"):
                new_cols[f"{prefix}{col}_roll{w}_{agg}"] = np.full(n, np.nan)
        for alpha in EMA_ALPHAS:
            new_cols[f"{prefix}{col}_ema{int(alpha * 10)}"] = np.full(n, np.nan)

    for team in df[team_col].unique():
        mask = df[team_col] == team
        team_pos = [pos_map[i] for i in df.index[mask]]

        for col in stat_cols_present:
            series = df.loc[mask, col]

            for w in WINDOWS:
                rolled = series.shift(1).rolling(w, min_periods=1)
                new_cols[f"{prefix}{col}_roll{w}_mean"][team_pos] = rolled.mean().values
                new_cols[f"{prefix}{col}_roll{w}_std"][team_pos] = rolled.std().fillna(0).values
                new_cols[f"{prefix}{col}_roll{w}_max"][team_pos] = rolled.max().values
                new_cols[f"{prefix}{col}_roll{w}_min"][team_pos] = rolled.min().values

            for alpha in EMA_ALPHAS:
                ema = series.shift(1).ewm(alpha=alpha, adjust=False).mean()
                new_cols[f"{prefix}{col}_ema{int(alpha * 10)}"][team_pos] = ema.values

    return pd.concat([df, pd.DataFrame(new_cols, index=full_index)], axis=1)


def add_over_under_streak(df: pd.DataFrame, result_col: str = "result_over") -> pd.DataFrame:
    df = df.sort_values("game_date").copy()

    df["league_over_rate_10"] = df[result_col].shift(1).rolling(10, min_periods=1).mean()

    for team_col in ["home_team", "away_team"]:
        if team_col not in df.columns:
            continue
        streaks = np.zeros(len(df), dtype=int)
        results = df[result_col].values
        for team in df[team_col].unique():
            positions = df.index[df[team_col] == team].tolist()
            pos_list = [df.index.get_loc(p) for p in positions]
            for k in range(1, len(pos_list)):
                i, i_prev = pos_list[k], pos_list[k - 1]
                streaks[i] = (
                    (max(0, streaks[i_prev]) + 1)
                    if results[i_prev] == 1
                    else (min(0, streaks[i_prev]) - 1)
                )
        df[f"{team_col}_over_streak"] = streaks

    return df


def add_season_progress(df: pd.DataFrame, date_col: str = "game_date") -> pd.DataFrame:
    df = df.copy()
    df["season_progress"] = (
        df.groupby("season")[date_col].transform(
            lambda s: (
                (s - s.min()).dt.total_seconds()
                / max(float((s.max() - s.min()).total_seconds()), 1.0)
            )
        )
    ).astype(float)
    return df
