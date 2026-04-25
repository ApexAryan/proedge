"""Advanced derived features: pace/efficiency differentials, luck regression, schedule density."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def add_advanced_features(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    df = add_schedule_density(df)
    df = add_win_loss_streak(df)
    df = add_hot_shooting_streak(df)
    df = add_pace_efficiency_composites(df, sport)
    df = add_luck_regression(df, sport)
    df = add_situational_interactions(df, sport)
    return df


# ── Schedule density ──────────────────────────────────────────────────────────

def add_schedule_density(df: pd.DataFrame) -> pd.DataFrame:
    """Games played by each team in the last 3 / 5 days (beyond simple rest days)."""
    df = df.sort_values("game_date").copy()
    home_3d = np.zeros(len(df))
    away_3d = np.zeros(len(df))
    home_5d = np.zeros(len(df))
    away_5d = np.zeros(len(df))

    # team → sorted list of game dates (as timestamps)
    team_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)

    for pos, (_, row) in enumerate(df.iterrows()):
        dt = pd.Timestamp(row["game_date"])
        home, away = row["home_team"], row["away_team"]

        for team, h3, h5, idx in [(home, home_3d, home_5d, pos),
                                   (away, away_3d, away_5d, pos)]:
            dates = team_dates[team]
            h3[idx] = sum(1 for d in dates if (dt - d).days <= 3)
            h5[idx] = sum(1 for d in dates if (dt - d).days <= 5)

        team_dates[home].append(dt)
        team_dates[away].append(dt)

    df["home_games_3d"] = home_3d
    df["away_games_3d"] = away_3d
    df["home_games_5d"] = home_5d
    df["away_games_5d"] = away_5d
    df["schedule_density_diff"] = home_3d - away_3d   # negative = home more rested
    return df


# ── Win / loss streak ─────────────────────────────────────────────────────────

def add_win_loss_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team win/loss streak at the time of each game.
    Positive = win streak length, negative = loss streak length.
    Uses shift(1) so current game outcome is not included.
    """
    df = df.sort_values("game_date").copy()
    home_streak = np.zeros(len(df))
    away_streak = np.zeros(len(df))
    team_streak: dict[str, int] = {}

    for pos, (_, row) in enumerate(df.iterrows()):
        home, away = row["home_team"], row["away_team"]
        home_streak[pos] = team_streak.get(home, 0)
        away_streak[pos] = team_streak.get(away, 0)

        h_score = row.get("home_score", 0)
        a_score = row.get("away_score", 0)
        if h_score > a_score:
            team_streak[home] = max(1, team_streak.get(home, 0) + 1)
            team_streak[away] = min(-1, team_streak.get(away, 0) - 1)
        elif a_score > h_score:
            team_streak[home] = min(-1, team_streak.get(home, 0) - 1)
            team_streak[away] = max(1, team_streak.get(away, 0) + 1)

    df["home_win_streak"] = home_streak
    df["away_win_streak"] = away_streak
    df["win_streak_diff"] = home_streak - away_streak
    return df


# ── Hot shooting streak ───────────────────────────────────────────────────────

def add_hot_shooting_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag teams running significantly above their season-long shooting average
    (TS% 3-game rolling > 20-game rolling + 5pp) — regression candidates → Under.
    Only computed when trueShooting rolling columns exist (NBA).
    """
    for side in ("home", "away"):
        roll3 = f"{side}_trueShooting_roll3_mean"
        roll20 = f"{side}_trueShooting_roll20_mean"
        if roll3 in df.columns and roll20 in df.columns:
            df[f"{side}_hot_shooting"] = (
                (df[roll3] - df[roll20] > 0.05).astype(float)
            )
            df[f"{side}_cold_shooting"] = (
                (df[roll20] - df[roll3] > 0.05).astype(float)
            )
        else:
            df[f"{side}_hot_shooting"]  = 0.0
            df[f"{side}_cold_shooting"] = 0.0
    return df


# ── Pace & efficiency composites ──────────────────────────────────────────────

def add_pace_efficiency_composites(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Key over/under predictors:
    - projected_possessions: sum of both teams' pace (best total-score proxy)
    - ppp_sum: combined points-per-possession expectation
    - ts_diff: true-shooting differential
    - off_rating_sum: combined offensive rating
    """
    for w in (5, 10):
        hp = f"home_pace_roll{w}_mean"
        ap = f"away_pace_roll{w}_mean"
        if hp in df.columns and ap in df.columns:
            df[f"projected_possessions_roll{w}"] = df[hp] + df[ap]

        hppp = f"home_pointsPerPossession_roll{w}_mean"
        appp = f"away_pointsPerPossession_roll{w}_mean"
        if hppp in df.columns and appp in df.columns:
            df[f"ppp_sum_roll{w}"] = df[hppp] + df[appp]
            df[f"ppp_diff_roll{w}"] = df[hppp] - df[appp]

        hts = f"home_trueShooting_roll{w}_mean"
        ats = f"away_trueShooting_roll{w}_mean"
        if hts in df.columns and ats in df.columns:
            df[f"ts_sum_roll{w}"]  = df[hts] + df[ats]
            df[f"ts_diff_roll{w}"] = df[hts] - df[ats]

        hor = f"home_offensiveRating_roll{w}_mean"
        aor = f"away_offensiveRating_roll{w}_mean"
        hdr = f"home_defensiveRating_roll{w}_mean"
        adr = f"away_defensiveRating_roll{w}_mean"
        if all(c in df.columns for c in [hor, aor, hdr, adr]):
            # Expected total from rating matchup
            df[f"expected_total_roll{w}"] = (
                (df[hor] + df[adr]) / 2 + (df[aor] + df[hdr]) / 2
            )
            df[f"off_rating_sum_roll{w}"] = df[hor] + df[aor]
            df[f"def_rating_sum_roll{w}"] = df[hdr] + df[adr]

    # NFL: yards-per-play and tempo
    if sport == "nfl":
        for w in (3, 5):
            hypp = f"home_yardsPerPlay_roll{w}_mean"
            aypp = f"away_yardsPerPlay_roll{w}_mean"
            if hypp in df.columns and aypp in df.columns:
                df[f"ypp_sum_roll{w}"] = df[hypp] + df[aypp]

            hspp = f"home_secondsPerPlay_roll{w}_mean"
            aspp = f"away_secondsPerPlay_roll{w}_mean"
            if hspp in df.columns and aspp in df.columns:
                # High seconds-per-play = slow pace = Under signal
                df[f"tempo_sum_roll{w}"] = df[hspp] + df[aspp]

    # MLB: K/BB ratio sum predicts low-scoring games
    if sport == "mlb":
        for w in (5, 10):
            hkbb = f"home_kBbRatio_roll{w}_mean"
            akbb = f"away_kBbRatio_roll{w}_mean"
            if hkbb in df.columns and akbb in df.columns:
                df[f"kbb_sum_roll{w}"] = df[hkbb] + df[akbb]

    return df


# ── Luck / regression factor ──────────────────────────────────────────────────

def add_luck_regression(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Teams scoring well above what their shot profile predicts are 'running hot'
    and likely to regress → Under signal.

    For NBA: expected_pts ≈ shots × league_avg_efficiency
    Luck = actual_roll5_pts - expected_pts  (positive = overperforming)
    """
    if sport != "nba":
        df["home_luck_factor"] = 0.0
        df["away_luck_factor"] = 0.0
        return df

    needed = [
        "home_points_roll5_mean",
        "home_fieldGoalAttempts_roll5_mean", "home_fieldGoalPct_roll5_mean",
        "home_threePointAttempts_roll5_mean", "home_threePointPct_roll5_mean",
        "home_freeThrowAttempts_roll5_mean", "home_freeThrowPct_roll5_mean",
    ]
    if not all(c in df.columns for c in needed):
        df["home_luck_factor"] = 0.0
        df["away_luck_factor"] = 0.0
        return df

    for side in ("home", "away"):
        pts   = df.get(f"{side}_points_roll5_mean",            pd.Series(113.0, index=df.index))
        fga   = df.get(f"{side}_fieldGoalAttempts_roll5_mean", pd.Series(87.0,  index=df.index))
        fgp   = df.get(f"{side}_fieldGoalPct_roll5_mean",      pd.Series(0.47,  index=df.index))
        fg3a  = df.get(f"{side}_threePointAttempts_roll5_mean",pd.Series(35.0,  index=df.index))
        fg3p  = df.get(f"{side}_threePointPct_roll5_mean",     pd.Series(0.36,  index=df.index))
        fta   = df.get(f"{side}_freeThrowAttempts_roll5_mean", pd.Series(22.0,  index=df.index))
        ftp   = df.get(f"{side}_freeThrowPct_roll5_mean",      pd.Series(0.78,  index=df.index))

        expected = 2 * fga * fgp + 1 * fg3a * fg3p + fta * ftp
        df[f"{side}_luck_factor"] = pts - expected

    return df


# ── Situational interactions ──────────────────────────────────────────────────

def add_situational_interactions(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Cross-feature interactions for situational factors already present in the df
    as constants during training (will vary at inference time).
    """
    # Altitude boost: thinner air → more scoring, especially in NBA/MLB
    if "altitude_feet" in df.columns:
        df["altitude_boost"] = df["altitude_feet"] / 5280.0  # 0 at sea level, 1.0 at Denver

    # Dome removes weather variance → typically higher-scoring
    if "is_dome" in df.columns:
        df["dome_flag"] = df["is_dome"]

    # Wind impact (NFL/MLB: high wind = under)
    if "wind_speed_mph" in df.columns:
        df["wind_under_signal"] = (df["wind_speed_mph"] > 15).astype(float)
        df["wind_severity"]     = df["wind_speed_mph"] / 30.0   # normalised 0–1+

    # Temperature extremes
    if "temperature_f" in df.columns:
        df["cold_game"] = (df["temperature_f"] < 40).astype(float)
        df["hot_game"]  = (df["temperature_f"] > 85).astype(float)

    # Sharp money direction (positive = sharp on over, negative = sharp on under)
    if "sharp_over_pct" in df.columns and "public_over_pct" in df.columns:
        df["sharp_vs_public"] = df["sharp_over_pct"] - df["public_over_pct"]

    # Line movement magnitude (big moves = strong information signal)
    if "line_movement" in df.columns:
        df["line_move_magnitude"] = df["line_movement"].abs()
        df["line_move_direction"] = np.sign(df["line_movement"])

    # Injury adjustment: each key player out ≈ −3 pts for NBA
    if "home_key_players_out" in df.columns:
        df["injury_pts_impact"] = (
            df["home_key_players_out"] - df["away_key_players_out"]
        ) * -3.0   # negative when home is more injured → Under for home, closer game

    return df
