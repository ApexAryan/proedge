# ProEdge Analytics

An adaptive over/under prediction engine for NFL, NBA, and MLB games. ProEdge uses an XGBoost + LightGBM ensemble trained on 200+ engineered features to forecast whether a game total will go over or under the posted betting line, with calibrated probabilities and confidence intervals.

---

## What It Does

- Fetches historical and live game data from ESPN, NBA Stats API, PrizePicks, and The Odds API
- Engineers 200+ features covering rolling team stats, pace/efficiency, injury impact, weather, park factors, and market signals
- Trains an ensemble model per sport and auto-retrains when data drift is detected
- Serves predictions via a REST API with live odds and injury enrichment baked in
- Tracks prediction accuracy and closing line value (CLV) as games settle
- Exports Prometheus metrics and ships a Grafana dashboard

---

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL 16, SQLAlchemy 2.0 (async), Alembic |
| ML | XGBoost, LightGBM, scikit-learn, joblib |
| Data | Pandas, NumPy, PyArrow (Parquet), SciPy |
| Data Sources | ESPN API, nba_api, PrizePicks API, The Odds API |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker Compose, Python 3.11+ |

---

## Architecture

```
External Data Sources
  ESPN API  →  scores, injuries, rosters
  NBA Stats →  detailed game logs
  Odds API  →  opening/closing lines
  PrizePicks→  player props and game totals
         ↓
   Data Ingestion Layer
  daily_updater.py runs at 6 AM UTC
  appends completed games to Parquet store
         ↓
   Feature Engineering (200+ features)
  rolling team stats, pace, efficiency,
  injury impact, weather, park factors,
  market signals, schedule context
         ↓
   ML Ensemble (XGBoost 50% + LightGBM 50%)
  time-aware train/val/holdout split
  isotonic calibration
  conformal prediction intervals
         ↓
   FastAPI REST layer
  POST /predictions  →  enriched predictions
  GET  /lines        →  live PrizePicks board
  POST /backtest     →  walk-forward CV
  GET  /metrics      →  Prometheus metrics
         ↓
   PostgreSQL  +  Prometheus  +  Grafana
  prediction settlement, CLV, rolling accuracy
```

---

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- (Optional) The Odds API key for live betting lines
- (Optional) Slack/Discord webhook URL for alerts

---

## Quick Start

**1. Clone and configure**
```bash
git clone <repo-url>
cd proedge
cp .env.example .env
# Edit .env with your database password, API keys, etc.
```

**2. Start the full stack**
```bash
docker compose up -d
```

This starts PostgreSQL, runs Alembic migrations, launches the API on port 8001, Prometheus on 9091, and Grafana on 3001.

**3. Train models**
```bash
make train-all
```

**4. Verify**
```bash
curl http://localhost:8001/health
```

---

## Local Development (without Docker)

```bash
make install       # pip install -e ".[dev]"
make env-setup     # copy .env.example to .env
make migrate       # run database migrations
make dev           # start uvicorn with hot reload on port 8000
```

```bash
make train-nfl     # train NFL model only
make train-nba     # train NBA model only
make train-mlb     # train MLB model only
make test          # run test suite
```

---

## API Reference

### Predict a game total

```
POST /predictions
```

```json
{
  "home_team": "Boston Celtics",
  "away_team": "Golden State Warriors",
  "game_date": "2025-01-15",
  "sport": "nba",
  "total_line": 224.5
}
```

Response:

```json
{
  "prediction_id": "uuid",
  "prob_over": 0.61,
  "prob_under": 0.39,
  "ci_lower": 218.0,
  "ci_upper": 231.0,
  "confidence": 0.72,
  "latency_ms": 43
}
```

The endpoint automatically enriches the request with live odds (from The Odds API) and injury counts (from ESPN) before running inference. These are best-effort — if a fetch fails, caller-supplied values are used as fallback.

### Settle a prediction

```
POST /predictions/{prediction_id}/settle
```

```json
{
  "actual_total": 229,
  "closing_line": 225.5
}
```

Records whether the prediction was correct and computes CLV (positive CLV means the model beat the closing line).

### Other endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB status, models loaded, uptime |
| GET | `/models/performance` | Accuracy, log loss, Brier score per model |
| GET | `/models/accuracy/live` | Rolling accuracy from settled predictions |
| POST | `/training/update/{sport}` | Fetch yesterday's games and optionally retrain |
| POST | `/backtest/{sport}` | Walk-forward cross-validation with ROI and Sharpe ratio |
| GET | `/lines/prizepicks/{sport}` | Live PrizePicks player props and game lines |
| GET | `/metrics` | Prometheus metrics endpoint |

---

## Feature Groups

**Group A — Rolling Team Stats**
3, 5, and 10-game rolling means and standard deviations for all sport-specific stats (passing yards, points, shooting percentage, pace, ratings). All features use `.shift(1)` to prevent look-ahead bias.

**Group B — Advanced Derived Features**
Schedule density (games in last 3/5 days), win/loss streaks, pace and efficiency composites, projected possessions, true shooting differentials, offensive/defensive rating matchups, luck regression signals, MLB park factors (Coors Field +15%, Oracle Park -8%, etc.), scoring volatility.

**Group C — Situational Context**
Weather (wind speed, temperature, dome flag) for NFL and MLB, altitude (Denver = 5280 ft), playoff flag, home/away rest days.

**Group D — Market Signals**
Line movement (closing minus opening), sharp vs. public sentiment splits, referee/umpire effect adjustments.

**Group E — Injury Impact**
Count of key players out from ESPN rosters, distinguishes "out/IR" from coach's-decision DNPs, impact score per absent player.

---

## ML Pipeline

**Ensemble:** XGBoost (50%) + LightGBM (50%) binary classifiers, blended by weighted average of probability outputs.

**Training split:** Time-ordered 70% train / 15% validation / 15% holdout. No shuffling — respects temporal order.

**Calibration:** Isotonic regression maps raw model probabilities to empirical frequencies.

**Confidence intervals:** Conformal prediction produces calibrated 80% CI around each prediction.

**Hyperparameter selection:** Automatically switches to aggressive regularization (max_depth=4, high L1/L2) for datasets under 2,000 games, standard regularization otherwise.

**Auto-retraining:** The daily updater triggers a retrain after 30+ new games accumulate, or immediately when Population Stability Index (PSI) ≥ 0.25 on any of the top-20 features.

**Model registry:** Each version saved to `models/<sport>/<version>/model.joblib` + `meta.json` with a `latest` symlink for live loading.

---

## Database Schema

| Table | Key Columns |
|---|---|
| `games` | game_id, sport, home_team, away_team, game_date, home_score, away_score, total_line, result_over, venue |
| `predictions` | prediction_id, game_id, model_version, prob_over, prob_under, confidence, ci_lower, ci_upper, features_snapshot (JSONB), is_correct, clv |
| `model_runs` | version, sport, accuracy, log_loss, brier_score, training_games, feature_count, xgb_weight, lgb_weight, is_active |
| `player_stats` | player_id, player_name, team_id, game_id, sport, game_date, stats (JSONB) |
| `injury_reports` | player_id, team_id, sport, status, impact_score, reported_at |

---

## Monitoring

**Prometheus metrics (port 9091):**
- `proedge_api_request_duration_seconds` — latency histogram by endpoint
- `proedge_predictions_total` — prediction count by sport and direction
- `proedge_prediction_confidence` — confidence score distribution
- `proedge_model_accuracy` — live rolling accuracy
- `proedge_drift_psi` — PSI per feature (triggers retrain at ≥ 0.25)
- `proedge_model_retrains_total` — retrain count by sport and reason

**Grafana** (port 3001): pre-built dashboard at `grafana/dashboards/`.

**Alerts:** Configure `ALERT_WEBHOOK_URL` (Slack or Discord) and `ALERT_CONFIDENCE_THRESHOLD` in `.env` to receive webhook notifications when high-confidence predictions are generated.

---

## Configuration

All configuration is via environment variables or `.env`. Key settings:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | required | PostgreSQL connection string |
| `API_KEY` | `""` | Bearer token for API auth (empty = no auth) |
| `ODDS_API_KEY` | `""` | The Odds API key for live lines |
| `ALERT_WEBHOOK_URL` | `""` | Slack/Discord webhook for alerts |
| `ALERT_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence to trigger alert |
| `MODEL_DIR` | `models/` | Where trained models are saved |
| `DATA_DIR` | `data/` | Where Parquet data files are stored |

See `.env.example` for the full list.

---

## Tests

```bash
make test
```

Test files:
- `tests/test_api.py` — HTTP endpoint integration tests
- `tests/test_models.py` — model training and inference
- `tests/test_features.py` — feature engineering
- `tests/test_backtester.py` — walk-forward backtesting logic

---

## Project Structure

```
src/proedge/
├── api/
│   ├── main.py                  # FastAPI app, lifespan, daily update loop
│   ├── schemas.py               # Pydantic request/response models
│   ├── middleware/              # Auth, logging middleware
│   └── routers/
│       ├── predictions.py       # POST /predictions
│       ├── performance.py       # GET /models/performance, /accuracy/live
│       ├── health.py            # GET /health
│       ├── training.py          # POST /training/update/{sport}
│       ├── backtest.py          # POST /backtest/{sport}
│       └── lines.py             # GET /lines/prizepicks/{sport}
├── db/
│   ├── models.py                # SQLAlchemy ORM models
│   ├── repositories.py          # Data access layer
│   └── session.py               # Async DB session
├── pipeline/
│   ├── features/
│   │   ├── advanced.py          # 200+ feature computations
│   │   └── store.py             # FeatureStore with disk caching
│   ├── ingestion/
│   │   ├── historical.py        # Historical Parquet loader
│   │   ├── stats.py             # ESPN live stats fetcher
│   │   ├── injuries.py          # ESPN injury report parser
│   │   ├── daily_updater.py     # Nightly game append + retrain trigger
│   │   ├── espn_nfl_fetcher.py  # NFL-specific ESPN integration
│   │   ├── mlb_stats_fetcher.py # MLB stats
│   │   ├── odds_fetcher.py      # The Odds API (15-min cache)
│   │   └── prizepicks_fetcher.py# PrizePicks board
│   ├── training/
│   │   └── trainer.py           # train(), check_and_retrain()
│   ├── models/
│   │   ├── ensemble.py          # OverUnderEnsemble (XGB + LGBM)
│   │   ├── drift.py             # DriftDetector (PSI + KS test)
│   │   └── registry.py          # ModelRegistry (versioning, save/load)
│   └── backtesting/
│       └── backtester.py        # Walk-forward CV, ROI, Sharpe, drawdown
├── monitoring/
│   ├── metrics.py               # Prometheus metric definitions
│   └── alerts.py                # Webhook alert sender
└── config.py                    # Pydantic Settings
```
