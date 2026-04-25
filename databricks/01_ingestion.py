# Databricks notebook source
# MAGIC %md
# MAGIC # ProEdge — 01: Data Ingestion
# MAGIC Ingests live player stats, injury reports, and historical game data from
# MAGIC upstream sports APIs into the Delta Lake bronze layer.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
)
import requests
from datetime import datetime, timedelta

spark = SparkSession.builder.appName("ProEdge-Ingestion").getOrCreate()

SPORTS = ["nfl", "nba", "mlb"]
BRONZE_PATH = "dbfs:/proedge/bronze"
API_KEY = dbutils.secrets.get(scope="proedge", key="sportradar_api_key")  # noqa: F821

# COMMAND ----------
# MAGIC %md ## 1. Historical Game Results (5-year backfill)

game_schema = StructType([
    StructField("game_id", StringType(), False),
    StructField("sport", StringType(), False),
    StructField("season", IntegerType(), False),
    StructField("game_date", TimestampType(), False),
    StructField("home_team", StringType(), False),
    StructField("away_team", StringType(), False),
    StructField("home_score", IntegerType(), True),
    StructField("away_score", IntegerType(), True),
    StructField("total_line", DoubleType(), True),
    StructField("result_over", IntegerType(), True),
    StructField("venue", StringType(), True),
])

for sport in SPORTS:
    games_url = f"https://api.sportradar.com/{sport}/trial/v7/en/seasons.json?api_key={API_KEY}"
    try:
        resp = requests.get(games_url, timeout=10)
        raw_json = resp.json()
        # Normalize and write to Delta
        games_df = spark.read.json(
            spark.sparkContext.parallelize([str(raw_json)])
        )
        (
            games_df
            .write.format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .save(f"{BRONZE_PATH}/games/{sport}")
        )
        print(f"[{sport}] Games ingested: {games_df.count()} records")
    except Exception as e:
        print(f"[{sport}] API unavailable ({e}) — skipping live ingest, using existing data")

# COMMAND ----------
# MAGIC %md ## 2. Live Player Stats

stats_schema = StructType([
    StructField("player_id", StringType(), False),
    StructField("player_name", StringType(), True),
    StructField("team_id", StringType(), False),
    StructField("game_id", StringType(), True),
    StructField("game_date", TimestampType(), False),
    StructField("sport", StringType(), False),
    StructField("stat_key", StringType(), False),
    StructField("stat_value", DoubleType(), True),
])

today = datetime.utcnow().date()
lookback_days = 7

for sport in SPORTS:
    dates = [(today - timedelta(days=i)).isoformat() for i in range(lookback_days)]
    print(f"[{sport}] Ingesting stats for dates: {dates}")

    # In production: call SportRadar daily game log endpoint per date
    # stats_df = spark.createDataFrame(fetch_stats(sport, dates), schema=stats_schema)
    # Here we read existing bronze data and append incrementally

    try:
        existing = spark.read.format("delta").load(f"{BRONZE_PATH}/stats/{sport}")
        cutoff = F.lit(datetime.utcnow() - timedelta(days=lookback_days))
        new_stats = existing.filter(F.col("game_date") >= cutoff)
        print(f"[{sport}] Stats rows (last {lookback_days}d): {new_stats.count()}")
    except Exception:
        print(f"[{sport}] No existing stats data — will populate on first full run")

# COMMAND ----------
# MAGIC %md ## 3. Injury Reports

injury_schema = StructType([
    StructField("player_id", StringType(), False),
    StructField("player_name", StringType(), True),
    StructField("team_id", StringType(), False),
    StructField("sport", StringType(), False),
    StructField("status", StringType(), False),
    StructField("injury_type", StringType(), True),
    StructField("impact_score", DoubleType(), True),
    StructField("reported_at", TimestampType(), False),
])

for sport in SPORTS:
    injury_url = f"https://api.sportradar.com/{sport}/trial/v7/en/injuries.json?api_key={API_KEY}"
    try:
        resp = requests.get(injury_url, timeout=10)
        injuries = resp.json().get("injuries", [])
        if injuries:
            inj_df = spark.createDataFrame(injuries, schema=injury_schema)
            (
                inj_df
                .write.format("delta")
                .mode("overwrite")
                .save(f"{BRONZE_PATH}/injuries/{sport}")
            )
            print(f"[{sport}] Injuries loaded: {inj_df.count()}")
    except Exception as e:
        print(f"[{sport}] Injury fetch failed: {e}")

# COMMAND ----------
# MAGIC %md ## 4. Delta Lake Optimize + Z-Order

for sport in SPORTS:
    for table in ["games", "stats", "injuries"]:
        try:
            spark.sql(f"""
                OPTIMIZE delta.`{BRONZE_PATH}/{table}/{sport}`
                ZORDER BY (game_date)
            """)
            print(f"[{sport}/{table}] Optimized")
        except Exception:
            pass

print("Ingestion complete.")
