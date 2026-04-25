# Databricks notebook source
# MAGIC %md
# MAGIC # ProEdge — 04: Drift Detection & Automated Retraining
# MAGIC Scheduled daily. Computes PSI against training reference distribution.
# MAGIC Triggers retraining job if drift threshold is exceeded.

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("ProEdge-DriftDetection").getOrCreate()

SPORTS = ["nfl", "nba", "mlb"]
SILVER_PATH = "dbfs:/proedge/silver"
DRIFT_PATH = "dbfs:/proedge/drift"
PSI_THRESHOLD = 0.25
TOP_K_FEATURES = 20
LOOKBACK_DAYS = 30  # current window
REFERENCE_DAYS = 180  # reference window

NON_FEATURE_COLS = {
    "game_id", "sport", "game_date", "home_team", "away_team", "season",
    "home_score", "away_score", "total", "result_over", "venue",
}

# COMMAND ----------
# MAGIC %md ## PSI Computation

def compute_psi_spark(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index between reference and current distributions."""
    bins = np.percentile(reference.dropna(), np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts = np.histogram(reference.dropna(), bins=bins)[0]
    cur_counts = np.histogram(current.dropna(), bins=bins)[0]

    ref_pct = ref_counts / max(len(reference.dropna()), 1)
    cur_pct = cur_counts / max(len(current.dropna()), 1)

    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

# COMMAND ----------
# MAGIC %md ## Main Drift Check Loop

drift_results = {}
retrain_sports = []
today = datetime.utcnow()

for sport in SPORTS:
    print(f"\n=== Drift check: {sport.upper()} ===")

    try:
        df = spark.read.format("delta").load(f"{SILVER_PATH}/features/{sport}")
    except Exception:
        print(f"  No feature data — skipping")
        continue

    df = df.withColumn("game_date", F.to_timestamp("game_date"))
    pdf = df.toPandas()
    pdf["game_date"] = pd.to_datetime(pdf["game_date"])

    feature_cols = [c for c in pdf.select_dtypes(include=[np.number]).columns
                    if c not in NON_FEATURE_COLS]

    cutoff_ref_end = today - timedelta(days=LOOKBACK_DAYS)
    cutoff_ref_start = today - timedelta(days=LOOKBACK_DAYS + REFERENCE_DAYS)

    ref_mask = (pdf["game_date"] >= cutoff_ref_start) & (pdf["game_date"] < cutoff_ref_end)
    cur_mask = pdf["game_date"] >= (today - timedelta(days=LOOKBACK_DAYS))

    pdf_ref = pdf[ref_mask]
    pdf_cur = pdf[cur_mask]

    if len(pdf_ref) < 10 or len(pdf_cur) < 5:
        print(f"  Insufficient data: ref={len(pdf_ref)}, current={len(pdf_cur)}")
        continue

    print(f"  Reference: {len(pdf_ref)} games | Current: {len(pdf_cur)} games")

    # Use top-K features by variance in reference set (proxy for importance)
    variances = pdf_ref[feature_cols].var().sort_values(ascending=False)
    top_features = list(variances.head(TOP_K_FEATURES).index)

    sport_psi: dict[str, float] = {}
    drifted_features: list[str] = []

    for feat in top_features:
        psi = compute_psi_spark(pdf_ref[feat], pdf_cur[feat])
        sport_psi[feat] = round(psi, 4)
        if psi >= PSI_THRESHOLD:
            drifted_features.append(feat)
            print(f"  DRIFT DETECTED: {feat} PSI={psi:.4f}")

    drift_results[sport] = {
        "features_checked": len(top_features),
        "features_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "max_psi": max(sport_psi.values()) if sport_psi else 0.0,
        "psi_scores": sport_psi,
        "retrain_triggered": len(drifted_features) > 0,
    }

    if len(drifted_features) > 0:
        retrain_sports.append(sport)
        print(f"  → Retraining triggered for {sport} ({len(drifted_features)} features drifted)")
    else:
        print(f"  → No significant drift detected (max PSI={max(sport_psi.values(), default=0):.4f})")

# COMMAND ----------
# MAGIC %md ## Log Drift Report to Delta

report_rows = []
for sport, report in drift_results.items():
    for feat, psi in report.get("psi_scores", {}).items():
        report_rows.append({
            "check_date": today,
            "sport": sport,
            "feature": feat,
            "psi": psi,
            "drifted": psi >= PSI_THRESHOLD,
            "retrain_triggered": report["retrain_triggered"],
        })

if report_rows:
    report_df = spark.createDataFrame(pd.DataFrame(report_rows))
    (
        report_df.write.format("delta")
        .mode("append")
        .save(f"{DRIFT_PATH}/reports")
    )

print(f"\nDrift check complete. Retraining queued for: {retrain_sports or 'none'}")

# COMMAND ----------
# MAGIC %md ## Trigger Retraining Jobs via Databricks Jobs API

if retrain_sports:
    import requests

    DATABRICKS_HOST = dbutils.secrets.get(scope="proedge", key="databricks_host")  # noqa: F821
    DATABRICKS_TOKEN = dbutils.secrets.get(scope="proedge", key="databricks_token")  # noqa: F821
    TRAINING_JOB_ID = dbutils.secrets.get(scope="proedge", key="training_job_id")  # noqa: F821

    for sport in retrain_sports:
        payload = {
            "job_id": int(TRAINING_JOB_ID),
            "notebook_params": {"sport": sport, "trigger": "drift_detection"},
        }
        resp = requests.post(
            f"{DATABRICKS_HOST}/api/2.1/jobs/run-now",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload,
            timeout=30,
        )
        if resp.ok:
            run_id = resp.json().get("run_id")
            print(f"  [{sport}] Retraining job submitted: run_id={run_id}")
        else:
            print(f"  [{sport}] Failed to submit retraining: {resp.text}")
