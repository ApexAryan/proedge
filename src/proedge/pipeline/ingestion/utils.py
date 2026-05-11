"""Shared helpers for ingestion fetchers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def compute_proxy_lines(
    df: pd.DataFrame,
    *,
    clip_lo: float,
    clip_hi: float,
    home_off_default: float,
    home_def_default: float,
    away_off_default: float,
    away_def_default: float,
    noise_std: float = 3.5,
    window: int = 20,
    home_advantage: float = 0.0,
) -> pd.DataFrame:
    """Rolling proxy bookmaker total shared across all sport fetchers.

    Blends each team's recent scoring and defensive averages, adds noise,
    and clips to sport-specific ranges. Clip bounds and defaults vary by sport:
      NFL: clip=[28,80], defaults≈23/21
      MLB: clip=[4,25],  defaults≈4.5
      NBA: clip=[180,280], defaults≈113/111
    """
    df = df.copy()
    team_scored: dict[str, list[float]] = defaultdict(list)
    team_allowed: dict[str, list[float]] = defaultdict(list)
    rng = np.random.default_rng(42)

    lines: list[float] = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        h_pts = float(row["home_score"])
        a_pts = float(row["away_score"])

        h_off = float(np.mean(team_scored[home][-window:])) if team_scored[home] else home_off_default
        h_def = float(np.mean(team_allowed[home][-window:])) if team_allowed[home] else home_def_default
        a_off = float(np.mean(team_scored[away][-window:])) if team_scored[away] else away_off_default
        a_def = float(np.mean(team_allowed[away][-window:])) if team_allowed[away] else away_def_default

        expected = (h_off + a_def) / 2.0 + (a_off + h_def) / 2.0 + home_advantage
        line = expected + float(rng.normal(0, noise_std))
        lines.append(float(np.clip(round(line * 2) / 2, clip_lo, clip_hi)))

        team_scored[home].append(h_pts)
        team_allowed[home].append(a_pts)
        team_scored[away].append(a_pts)
        team_allowed[away].append(h_pts)

    df["total_line"] = lines
    df["result_over"] = (df["total"] > df["total_line"]).astype(int)
    return df
