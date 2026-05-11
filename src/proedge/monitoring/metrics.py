"""Prometheus metrics for API latency, model performance, and prediction confidence."""

from prometheus_client import Counter, Gauge, Histogram

# ── API metrics ───────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "proedge_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "proedge_api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# ── Prediction metrics ────────────────────────────────────────────────────────

PREDICTION_COUNT = Counter(
    "proedge_predictions_total",
    "Total predictions generated",
    ["sport", "direction"],
)

PREDICTION_CONFIDENCE = Histogram(
    "proedge_prediction_confidence",
    "Distribution of prediction confidence scores",
    ["sport"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTION_PROB_OVER = Histogram(
    "proedge_prediction_prob_over",
    "Distribution of over-probabilities",
    ["sport"],
    buckets=[i / 10 for i in range(11)],
)

# ── Model performance metrics ─────────────────────────────────────────────────

MODEL_ACCURACY = Gauge(
    "proedge_model_accuracy",
    "Live rolling accuracy of the active model",
    ["sport", "model_version"],
)

MODEL_LOG_LOSS = Gauge(
    "proedge_model_log_loss",
    "Live rolling log loss",
    ["sport", "model_version"],
)

MODEL_BRIER_SCORE = Gauge(
    "proedge_model_brier_score",
    "Live rolling Brier score",
    ["sport", "model_version"],
)

DRIFT_PSI = Gauge(
    "proedge_drift_psi",
    "Latest PSI score for the top feature",
    ["sport", "feature"],
)

RETRAIN_COUNTER = Counter(
    "proedge_model_retrains_total",
    "Number of automated retraining runs",
    ["sport", "trigger_reason"],
)

# ── System metrics ────────────────────────────────────────────────────────────

ACTIVE_MODEL_VERSION = Gauge(
    "proedge_active_model_info",
    "Currently active model version (label only)",
    ["sport", "version"],
)

# ── Data quality metrics ──────────────────────────────────────────────────────

SYNTHETIC_DATA_TOTAL = Counter(
    "proedge_synthetic_data_total",
    "Number of times synthetic data was used for model training (real data unavailable)",
    ["sport"],
)

DATA_FETCH_ERRORS = Counter(
    "proedge_data_fetch_errors_total",
    "Data fetcher failures by sport and source",
    ["sport", "source"],
)

FEATURE_CACHE_HITS = Counter(
    "proedge_feature_cache_total",
    "Feature cache hits and misses",
    ["result"],  # "hit" or "miss"
)

INFERENCE_FEATURE_MISSING = Counter(
    "proedge_inference_feature_missing_total",
    "Features present in training but missing at inference time",
    ["sport"],
)
