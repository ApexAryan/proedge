import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://proedge:proedge@localhost:5432/proedge_test")
os.environ.setdefault("MODEL_REGISTRY_PATH", "/tmp/proedge_test_models")

from proedge.pipeline.ingestion.stats import STAT_KEYS


@pytest.fixture
def sample_game_df():
    """Minimal realistic game DataFrame for feature testing."""
    rng = np.random.default_rng(42)
    n = 200
    teams = ["BOS", "LAL", "GSW", "MIA", "CHI"]
    sport = "nba"
    stat_cols = STAT_KEYS[sport]

    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        total = float(rng.normal(224, 18))
        line = total + rng.normal(0, 2)
        rows.append({
            "game_id": f"test_{i:04d}",
            "sport": sport,
            "season": 2024 if i < 100 else 2025,
            "game_date": datetime(2024, 1, 1) + timedelta(days=i),
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
        })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@pytest.fixture
def sample_feature_matrix(sample_game_df):
    """Pre-computed feature matrix for model tests."""
    from proedge.pipeline.features.store import FeatureStore
    store = FeatureStore(cache_dir="/tmp/proedge_test_features")
    return store.compute(sample_game_df, "nba", use_cache=False)


@pytest.fixture
def trained_model(sample_feature_matrix):
    """Quickly trained model for API / integration tests."""
    from proedge.pipeline.features.store import FeatureStore
    from proedge.pipeline.models.ensemble import OverUnderEnsemble

    store = FeatureStore(cache_dir="/tmp/proedge_test_features")
    feature_cols = store.get_feature_columns(sample_feature_matrix)

    X = sample_feature_matrix[feature_cols].fillna(0)
    y = sample_feature_matrix["result_over"].astype(int)

    n = len(X)
    X_tr, y_tr = X.iloc[: int(n * 0.7)], y.iloc[: int(n * 0.7)]
    X_val, y_val = X.iloc[int(n * 0.7) : int(n * 0.85)], y.iloc[int(n * 0.7) : int(n * 0.85)]

    model = OverUnderEnsemble(xgb_weight=0.5, lgb_weight=0.5)
    model.fit(X_tr, y_tr, X_val, y_val)
    return model, feature_cols
