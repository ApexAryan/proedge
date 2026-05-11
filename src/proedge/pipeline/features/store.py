"""Feature store: orchestrates all feature engineering into a versioned, cached output."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from proedge.pipeline.features.advanced import add_advanced_features
from proedge.pipeline.features.fatigue import add_fatigue_features
from proedge.pipeline.features.matchup import add_matchup_features
from proedge.pipeline.features.rolling import (
    add_over_under_streak,
    add_rolling_features,
    add_season_progress,
)
from proedge.pipeline.ingestion.stats import STAT_KEYS

logger = logging.getLogger(__name__)

# Features excluded from model input
_DROP_COLS = {
    "game_id", "sport", "game_date", "home_team", "away_team",
    "home_score", "away_score", "total", "result_over", "venue",
    "season", "external_id",
}


class FeatureStore:
    """
    Computes the full 200+ feature matrix from raw historical game data.
    Caches to disk keyed by a hash of the input shape and date range.
    """

    def __init__(self, cache_dir: str = "./data/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute(self, df: pd.DataFrame, sport: str, use_cache: bool = True) -> pd.DataFrame:
        cache_key = self._cache_key(df, sport)
        cache_path = self.cache_dir / f"{sport}_{cache_key}.parquet"

        if use_cache and cache_path.exists():
            logger.info("Loading features from cache: %s", cache_path)
            try:
                from proedge.monitoring.metrics import FEATURE_CACHE_HITS
                FEATURE_CACHE_HITS.labels(result="hit").inc()
            except Exception:
                pass
            return pd.read_parquet(cache_path)

        try:
            from proedge.monitoring.metrics import FEATURE_CACHE_HITS
            FEATURE_CACHE_HITS.labels(result="miss").inc()
        except Exception:
            pass
        logger.info("Computing feature matrix for %s (%d games)", sport, len(df))
        features = self._build(df, sport)

        features.to_parquet(cache_path, index=False)
        logger.info(
            "Feature matrix: %d rows × %d columns — saved to %s",
            len(features), features.shape[1], cache_path,
        )
        return features

    def _build(self, df: pd.DataFrame, sport: str) -> pd.DataFrame:
        stat_cols = STAT_KEYS.get(sport, [])

        home_stats = [f"home_{s}" for s in stat_cols if f"home_{s}" in df.columns]
        away_stats = [f"away_{s}" for s in stat_cols if f"away_{s}" in df.columns]

        # Rolling features (shift(1) applied inside — no current-game leakage)
        df = add_rolling_features(df, home_stats, team_col="home_team", prefix="")
        df = add_rolling_features(df, away_stats, team_col="away_team", prefix="")

        # Matchup features (also uses shift(1) internally — safe)
        df = add_matchup_features(df, stat_cols)

        # Drop raw per-game stat columns — they contain the current game's outcome
        # (e.g. home_points == actual score) which leaks the label at training time.
        # All signal from these columns is already captured in the rolled versions.
        raw_stat_cols = [c for c in home_stats + away_stats if c in df.columns]
        df = df.drop(columns=raw_stat_cols, errors="ignore")

        # Fatigue / rest / travel
        df = add_fatigue_features(df)

        # Streaks and season context
        df = add_over_under_streak(df)
        df = add_season_progress(df)

        # Advanced: pace composites, luck regression, schedule density, situational
        df = add_advanced_features(df, sport)

        # Ratio features from rolling means (raw stats already dropped)
        df = self._add_ratio_features(df, stat_cols)

        # Total line is a direct model input
        if "total_line" not in df.columns:
            df["total_line"] = np.nan

        return df

    def _add_ratio_features(self, df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
        """Rolling mean differentials and ratios — raw stat cols are already dropped."""
        new_cols: dict[str, pd.Series] = {}
        for w in [3, 5, 10]:
            for stat in stat_cols:
                h_roll = f"home_{stat}_roll{w}_mean"
                a_roll = f"away_{stat}_roll{w}_mean"
                if h_roll in df.columns and a_roll in df.columns:
                    h = df[h_roll]
                    a = df[a_roll]
                    new_cols[f"roll{w}_diff_{stat}"] = h - a
                    denom = (h + a).replace(0, np.nan)
                    new_cols[f"roll{w}_ratio_{stat}"] = h / denom
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in _DROP_COLS]

    def _cache_key(self, df: pd.DataFrame, sport: str) -> str:
        sig = f"{sport}_{len(df)}_{df['game_date'].min()}_{df['game_date'].max()}"
        return hashlib.md5(sig.encode()).hexdigest()[:8]
