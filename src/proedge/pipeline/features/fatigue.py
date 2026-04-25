"""Rest and travel fatigue indices."""
from __future__ import annotations

import numpy as np
import pandas as pd


# Approximate stadium coordinates (lat, lon) — used for travel distance
VENUE_COORDS: dict[str, tuple[float, float]] = {
    # NFL
    "ARI": (33.5277, -112.2626), "ATL": (33.7555, -84.4010), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LV": (36.0909, -115.1833), "LAC": (33.8644, -118.2611),
    "LAR": (33.8644, -118.2611), "MIA": (25.9580, -80.2389), "MIN": (44.9740, -93.2571),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SF": (37.4032, -121.9698), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645),
    # NBA (arenas — same city approximations)
    "BOS": (42.3662, -71.0621), "BKN": (40.6826, -73.9754), "GSW": (37.7680, -122.3877),
    "LAL": (34.0430, -118.2673), "MIL": (43.0450, -87.9170), "PHX": (33.4457, -112.0712),
    "OKC": (35.4634, -97.5152), "POR": (45.5316, -122.6668), "SAC": (38.5802, -121.4996),
    "SAS": (29.4270, -98.4375), "TOR": (43.6435, -79.3791), "UTA": (40.7683, -111.9011),
    # MLB
    "CHC": (41.9484, -87.6553), "CWS": (41.8299, -87.6338), "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262), "OAK": (37.7516, -122.2005), "WSH": (38.8730, -77.0074),
}


def add_fatigue_features(
    df: pd.DataFrame,
    date_col: str = "game_date",
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    df = df.sort_values(date_col).copy()

    # Rest days since last game (per team, per side)
    for side, team_col in [("home", home_col), ("away", away_col)]:
        rest_col = f"{side}_rest_days"
        b2b_col = f"{side}_back_to_back"
        games_7d_col = f"{side}_games_7d"
        games_14d_col = f"{side}_games_14d"
        dist_col = f"{side}_travel_km"

        df[rest_col] = np.nan
        df[b2b_col] = 0
        df[games_7d_col] = 0
        df[games_14d_col] = 0
        df[dist_col] = 0.0

        for team in df[team_col].unique():
            mask = df[team_col] == team
            team_df = df[mask].sort_values(date_col)

            dates = team_df[date_col].values
            venues = team_df.get(f"{side}_venue", team_df[team_col]).values

            rest_days = np.full(len(dates), np.nan)
            back_to_back = np.zeros(len(dates), dtype=int)
            games_7 = np.zeros(len(dates), dtype=int)
            games_14 = np.zeros(len(dates), dtype=int)
            travel_km = np.zeros(len(dates))

            for i in range(1, len(dates)):
                delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
                rest_days[i] = delta
                back_to_back[i] = int(delta <= 1)

                cutoff_7 = pd.Timestamp(dates[i]) - pd.Timedelta(days=7)
                cutoff_14 = pd.Timestamp(dates[i]) - pd.Timedelta(days=14)
                games_7[i] = sum(
                    1 for d in dates[:i] if pd.Timestamp(d) >= cutoff_7
                )
                games_14[i] = sum(
                    1 for d in dates[:i] if pd.Timestamp(d) >= cutoff_14
                )

                prev_venue = str(venues[i - 1]) if i > 0 else team
                curr_venue = str(venues[i])
                travel_km[i] = _haversine_km(
                    VENUE_COORDS.get(prev_venue, VENUE_COORDS.get(team, (0, 0))),
                    VENUE_COORDS.get(curr_venue, VENUE_COORDS.get(team, (0, 0))),
                )

            df.loc[mask, rest_col] = rest_days
            df.loc[mask, b2b_col] = back_to_back
            df.loc[mask, games_7d_col] = games_7
            df.loc[mask, games_14d_col] = games_14
            df.loc[mask, dist_col] = travel_km

    df["home_advantage"] = 1  # constant; model learns the coefficient
    df["home_rest_advantage"] = df["home_rest_days"].fillna(7) - df["away_rest_days"].fillna(7)

    return df.fillna({"home_rest_days": 7, "away_rest_days": 7})


def _haversine_km(
    coord1: tuple[float, float], coord2: tuple[float, float]
) -> float:
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))
