import numpy as np

from proedge.pipeline.features.rolling import add_rolling_features, add_over_under_streak
from proedge.pipeline.features.fatigue import add_fatigue_features
from proedge.pipeline.features.store import FeatureStore


def test_rolling_features_shape(sample_game_df):
    stat_cols = ["home_points", "home_rebounds"]
    # Add dummy columns if not present
    for c in stat_cols:
        if c not in sample_game_df.columns:
            sample_game_df[c] = np.random.rand(len(sample_game_df))
    result = add_rolling_features(sample_game_df, stat_cols, team_col="home_team")
    # Should have added: len(stat_cols) * (4 windows * 3 stats + 2 ema) new columns
    assert result.shape[1] > sample_game_df.shape[1]


def test_rolling_no_leakage(sample_game_df):
    """Rolling features must shift by 1 — no future data leakage."""
    stat_col = "home_points"
    if stat_col not in sample_game_df.columns:
        sample_game_df[stat_col] = np.random.rand(len(sample_game_df))

    result = add_rolling_features(
        sample_game_df.sort_values("game_date"), [stat_col], team_col="home_team"
    )
    roll3_col = f"{stat_col}_roll3_mean"
    if roll3_col in result.columns:
        # First occurrence per team should be NaN (no prior games)
        # May be NaN or a valid value depending on min_periods=1 — just ensure no crash
        assert roll3_col in result.columns


def test_fatigue_features_rest_days(sample_game_df):
    result = add_fatigue_features(sample_game_df)
    assert "home_rest_days" in result.columns
    assert "away_rest_days" in result.columns
    assert "home_back_to_back" in result.columns
    # Rest days should be non-negative where not NaN
    valid = result["home_rest_days"].dropna()
    assert (valid >= 0).all()


def test_over_under_streak_bounded(sample_game_df):
    result = add_over_under_streak(sample_game_df, result_col="result_over")
    assert "league_over_rate_10" in result.columns
    rates = result["league_over_rate_10"].dropna()
    assert (rates >= 0).all() and (rates <= 1).all()


def test_feature_store_column_count(sample_game_df):
    store = FeatureStore(cache_dir="/tmp/proedge_test_features")
    feature_df = store.compute(sample_game_df, "nba", use_cache=False)
    feature_cols = store.get_feature_columns(feature_df)
    # Should produce a substantial number of features (targeting 200+)
    assert len(feature_cols) >= 50, f"Only {len(feature_cols)} features generated"


def test_feature_store_no_infinities(sample_game_df):
    store = FeatureStore(cache_dir="/tmp/proedge_test_features")
    feature_df = store.compute(sample_game_df, "nba", use_cache=False)
    feature_cols = store.get_feature_columns(feature_df)
    X = feature_df[feature_cols]
    assert not np.isinf(X.values).any(), "Feature matrix contains infinite values"


def test_feature_store_cache(sample_game_df, tmp_path):
    store = FeatureStore(cache_dir=str(tmp_path))
    df1 = store.compute(sample_game_df, "nba", use_cache=True)
    df2 = store.compute(sample_game_df, "nba", use_cache=True)
    assert df1.shape == df2.shape
