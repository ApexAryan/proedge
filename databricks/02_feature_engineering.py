# Databricks notebook source
# MAGIC %md
# MAGIC # ProEdge — 02: Feature Engineering
# MAGIC Transforms bronze game data into the 200+ signal feature store (silver layer).

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("ProEdge-Features").getOrCreate()

SPORTS = ["nfl", "nba", "mlb"]
BRONZE_PATH = "dbfs:/proedge/bronze"
SILVER_PATH = "dbfs:/proedge/silver"

WINDOWS = [3, 5, 10, 20]
EMA_ALPHAS = [0.3, 0.5]

STAT_KEYS = {
    "nfl": ["passingYards", "rushingYards", "pointsScored", "pointsAllowed",
             "turnovers", "sacks", "thirdDownConversion", "redZoneEfficiency"],
    "nba": ["points", "rebounds", "assists", "steals", "blocks", "turnovers",
             "fieldGoalPct", "threePointPct", "defensiveRating", "offensiveRating", "pace"],
    "mlb": ["runsScored", "runsAllowed", "hits", "strikeouts", "era",
             "whip", "battingAvg", "onBasePct", "sluggingPct"],
}

# COMMAND ----------
# MAGIC %md ## Rolling Window Features

def add_rolling_features(df, stat_cols, team_col, date_col="game_date", prefix=""):
    """Compute rolling mean/std over multiple windows, partitioned by team."""
    team_date_window = Window.partitionBy(team_col).orderBy(date_col)

    for col in stat_cols:
        if col not in df.columns:
            continue
        # Shift by 1 to avoid data leakage
        shifted = F.lag(col, 1).over(team_date_window)
        df = df.withColumn(f"__shift_{col}", shifted)

        for w in WINDOWS:
            w_spec = Window.partitionBy(team_col).orderBy(date_col).rowsBetween(-w, -1)
            df = df.withColumn(f"{prefix}{col}_roll{w}_mean", F.avg(col).over(w_spec))
            df = df.withColumn(f"{prefix}{col}_roll{w}_std", F.stddev(col).over(w_spec))

        # Exponential moving average approximated via weighted lag combination
        for alpha in EMA_ALPHAS:
            ema_col = f"{prefix}{col}_ema{int(alpha * 10)}"
            # PySpark approximation: use weighted sum of last 10 values
            w_spec_ema = Window.partitionBy(team_col).orderBy(date_col).rowsBetween(-10, -1)
            df = df.withColumn(ema_col, F.avg(col).over(w_spec_ema))

        df = df.drop(f"__shift_{col}")

    return df

# COMMAND ----------
# MAGIC %md ## Fatigue / Rest Features

def add_rest_features(df, team_col, date_col="game_date", prefix=""):
    team_date_window = Window.partitionBy(team_col).orderBy(date_col)

    prev_date = F.lag(date_col, 1).over(team_date_window)
    df = df.withColumn(
        f"{prefix}rest_days",
        F.datediff(F.col(date_col), prev_date).cast(DoubleType()),
    )
    df = df.withColumn(
        f"{prefix}back_to_back",
        (F.col(f"{prefix}rest_days") <= 1).cast(DoubleType()),
    )

    # Games in last 7 days
    w7 = Window.partitionBy(team_col).orderBy(F.col(date_col).cast("long")).rangeBetween(-7 * 86400, -1)
    df = df.withColumn(f"{prefix}games_7d", F.count(date_col).over(w7))

    # Games in last 14 days
    w14 = Window.partitionBy(team_col).orderBy(F.col(date_col).cast("long")).rangeBetween(-14 * 86400, -1)
    df = df.withColumn(f"{prefix}games_14d", F.count(date_col).over(w14))

    return df

# COMMAND ----------
# MAGIC %md ## Season Progress & Over/Under Rate

def add_context_features(df):
    # Season progress: fraction of season elapsed
    season_window = Window.partitionBy("sport", "season")
    df = df.withColumn("season_min_date", F.min("game_date").over(season_window))
    df = df.withColumn("season_max_date", F.max("game_date").over(season_window))
    df = df.withColumn(
        "season_progress",
        (F.unix_timestamp("game_date") - F.unix_timestamp("season_min_date")) /
        (F.unix_timestamp("season_max_date") - F.unix_timestamp("season_min_date") + 1),
    )
    df = df.drop("season_min_date", "season_max_date")

    # Home advantage flag
    df = df.withColumn("home_advantage", F.lit(1.0))

    # Over/under streak per home team (rolling)
    team_date_w = Window.partitionBy("home_team").orderBy("game_date").rowsBetween(-10, -1)
    df = df.withColumn("home_team_over_rate_10", F.avg("result_over").over(team_date_w))

    # League-wide rolling over rate
    date_w = Window.orderBy("game_date").rowsBetween(-50, -1)
    df = df.withColumn("league_over_rate", F.avg("result_over").over(date_w))

    return df

# COMMAND ----------
# MAGIC %md ## Main Pipeline per Sport

for sport in SPORTS:
    print(f"\n=== Processing {sport.upper()} ===")
    try:
        df = spark.read.format("delta").load(f"{BRONZE_PATH}/games/{sport}")
    except Exception:
        print(f"  No bronze data for {sport} — skipping")
        continue

    stat_cols = STAT_KEYS.get(sport, [])
    home_stats = [c for c in [f"home_{s}" for s in stat_cols] if c in df.columns]
    away_stats = [c for c in [f"away_{s}" for s in stat_cols] if c in df.columns]

    # Rolling features per team side
    df = add_rolling_features(df, home_stats, "home_team", prefix="h_")
    df = add_rolling_features(df, away_stats, "away_team", prefix="a_")

    # Rest/fatigue per team side
    df = add_rest_features(df, "home_team", prefix="home_")
    df = add_rest_features(df, "away_team", prefix="away_")

    # Contextual features
    df = add_context_features(df)

    # Ratio features: home vs away differentials for rolling means
    for stat in stat_cols[:5]:
        for w in [3, 5, 10]:
            h_col = f"h_home_{stat}_roll{w}_mean"
            a_col = f"a_away_{stat}_roll{w}_mean"
            if h_col in df.columns and a_col in df.columns:
                df = df.withColumn(f"roll{w}_diff_{stat}", F.col(h_col) - F.col(a_col))

    n_features = len([c for c in df.columns if c not in {
        "game_id", "sport", "game_date", "home_team", "away_team",
        "home_score", "away_score", "total", "result_over", "venue", "season",
    }])
    print(f"  Feature count: {n_features}")

    (
        df.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .partitionBy("season")
        .save(f"{SILVER_PATH}/features/{sport}")
    )
    print(f"  Written to {SILVER_PATH}/features/{sport}")

    # Register as Databricks table for SQL access
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS proedge.{sport}_features
        USING DELTA
        LOCATION '{SILVER_PATH}/features/{sport}'
    """)

print("\nFeature engineering complete.")
