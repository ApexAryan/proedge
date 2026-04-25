# Databricks notebook source
# MAGIC %md
# MAGIC # ProEdge — 03: Distributed Model Training
# MAGIC Reads silver feature store, trains XGBoost + LightGBM ensemble on each sport,
# MAGIC evaluates on holdout set, and registers to Azure ML model registry.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ProEdge-Training").getOrCreate()

SPORTS = ["nfl", "nba", "mlb"]
SILVER_PATH = "dbfs:/proedge/silver"
HOLDOUT_FRAC = 0.15
VAL_FRAC = 0.15
EXPERIMENT_NAME = "/proedge/over-under-ensemble"

mlflow.set_experiment(EXPERIMENT_NAME)

# Column groups to exclude from features
NON_FEATURE_COLS = {
    "game_id", "sport", "game_date", "home_team", "away_team", "season",
    "home_score", "away_score", "total", "result_over", "venue", "external_id",
}

# COMMAND ----------
# MAGIC %md ## Training Loop

results = {}

for sport in SPORTS:
    print(f"\n{'='*60}")
    print(f"Training {sport.upper()} model")
    print(f"{'='*60}")

    try:
        df = spark.read.format("delta").load(f"{SILVER_PATH}/features/{sport}")
    except Exception as e:
        print(f"  No feature data for {sport}: {e}")
        continue

    # Convert to Pandas for sklearn/xgb/lgb
    pdf = df.toPandas().sort_values("game_date").reset_index(drop=True)
    print(f"  Total games: {len(pdf)}")

    feature_cols = [c for c in pdf.select_dtypes(include=[np.number]).columns
                    if c not in NON_FEATURE_COLS]
    print(f"  Features: {len(feature_cols)}")

    X = pdf[feature_cols].fillna(0)
    y = pdf["result_over"].astype(int)

    n = len(X)
    holdout_start = int(n * (1 - HOLDOUT_FRAC))
    val_start = int(holdout_start * (1 - VAL_FRAC))

    X_train, y_train = X.iloc[:val_start], y.iloc[:val_start]
    X_val, y_val = X.iloc[val_start:holdout_start], y.iloc[val_start:holdout_start]
    X_test, y_test = X.iloc[holdout_start:], y.iloc[holdout_start:]

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, holdout={len(X_test)}")

    with mlflow.start_run(run_name=f"{sport}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("sport", sport)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_games", len(X_train))
        mlflow.log_param("holdout_games", len(X_test))

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", early_stopping_rounds=30,
            random_state=42, verbosity=0,
        )
        xgb_model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )

        # --- LightGBM ---
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=-1,
        )
        lgb_model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        # --- Calibration ---
        xgb_val_prob = xgb_model.predict_proba(X_val.values)[:, 1]
        lgb_val_prob = lgb_model.predict_proba(X_val.values)[:, 1]
        ensemble_val = 0.5 * xgb_val_prob + 0.5 * lgb_val_prob

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(ensemble_val, y_val.values)

        # --- Holdout evaluation ---
        xgb_test_prob = xgb_model.predict_proba(X_test.values)[:, 1]
        lgb_test_prob = lgb_model.predict_proba(X_test.values)[:, 1]
        raw_test = 0.5 * xgb_test_prob + 0.5 * lgb_test_prob
        cal_test = calibrator.predict(raw_test)

        preds_binary = (cal_test >= 0.5).astype(int)
        accuracy = float((preds_binary == y_test.values).mean())
        auc = float(roc_auc_score(y_test.values, cal_test))
        ll = float(log_loss(y_test.values, cal_test))
        brier = float(brier_score_loss(y_test.values, cal_test))
        lift = (accuracy - 0.50) / 0.50 * 100

        print(f"  Holdout: accuracy={accuracy:.4f} | AUC={auc:.4f} | LogLoss={ll:.4f}")
        print(f"  Directional lift over baseline: +{lift:.1f}%")

        mlflow.log_metrics({
            "accuracy": accuracy,
            "auc": auc,
            "log_loss": ll,
            "brier_score": brier,
            "lift_pct": lift,
        })

        # --- Log models ---
        mlflow.sklearn.log_model(xgb_model, f"xgb_{sport}")
        mlflow.sklearn.log_model(lgb_model, f"lgb_{sport}")
        mlflow.sklearn.log_model(calibrator, f"calibrator_{sport}")
        mlflow.log_param("feature_names", ",".join(feature_cols[:50]))  # truncated

        # --- Feature importance ---
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "xgb_importance": xgb_model.feature_importances_,
            "lgb_importance": lgb_model.feature_importances_,
        }).sort_values("xgb_importance", ascending=False)
        imp_df.to_csv(f"/tmp/{sport}_feature_importance.csv", index=False)
        mlflow.log_artifact(f"/tmp/{sport}_feature_importance.csv")

        results[sport] = {
            "accuracy": accuracy, "auc": auc,
            "log_loss": ll, "brier": brier, "lift_pct": lift,
        }

# COMMAND ----------
# MAGIC %md ## Summary

summary = pd.DataFrame(results).T
print("\n=== Training Summary ===")
print(summary.to_string())
display(summary)
