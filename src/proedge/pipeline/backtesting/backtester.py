"""Walk-forward backtesting — replay historical predictions to measure edge and ROI."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from proedge.pipeline.features.store import FeatureStore
from proedge.pipeline.models.ensemble import OverUnderEnsemble

logger = logging.getLogger(__name__)

# Payoff ratio at standard -110 juice: win $90.91 on a $100 bet
_JUICE_PAYOFF = 100 / 110  # ≈ 0.909 1
_BET_SIZE = 100.0  # flat-bet unit in dollars
_KELLY_CAP = 0.25  # quarter-Kelly cap


@dataclass
class FoldResult:
    fold: int
    start_date: str
    end_date: str
    n_games: int
    accuracy: float
    auc: float
    log_loss: float
    brier_score: float
    roi_flat: float  # ROI assuming $100 flat bet at -110 juice per game
    roi_kelly: float  # ROI using (capped) Kelly fraction sizing
    edge_mean: float  # mean |prob_over - 0.5| (average model confidence)
    high_conf_accuracy: float  # accuracy on games where confidence >= 0.60


@dataclass
class BacktestResult:
    sport: str
    n_folds: int
    min_confidence: float
    total_games: int
    total_bets: int  # games where confidence >= min_confidence
    overall_accuracy: float
    overall_auc: float
    overall_roi_flat: float
    overall_roi_kelly: float
    sharpe_ratio: float  # of per-game returns (annualised × √252)
    max_drawdown: float  # max peak-to-trough decline in cumulative P&L
    status: str = "complete"  # "complete" | "no_folds_completed" | "insufficient_data"
    folds: list[FoldResult] = field(default_factory=list)
    calibration: dict[str, list[float]] = field(default_factory=dict)
    # {"bin_midpoint": [...], "actual_freq": [...], "predicted_prob": [...]}


class Backtester:
    """
    Walk-forward (expanding-window) backtester for the ProEdge over/under model.

    For fold k the training set is all games *before* the fold window and the
    test set is the fold window itself.  Features are computed on the full
    sorted dataset once (rolling features use shift(1), so there is no
    look-ahead leakage), then the X/y matrices are split by date for each fold.
    """

    def __init__(self, sport: str, data_dir: str = "./data") -> None:
        self.sport = sport.lower()
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_folds: int = 5,
        min_train_games: int = 500,
        min_confidence: float = 0.0,
    ) -> BacktestResult:
        """Time-series walk-forward cross-validation.

        Splits historical data into ``n_folds`` equal time periods.
        For fold k: train on folds 0..k-1, predict on fold k.
        Requires at least ``min_train_games`` in training set before the first
        prediction fold is attempted.
        """
        logger.info(
            "=== Backtester | sport=%s | folds=%d | min_conf=%.2f ===",
            self.sport,
            n_folds,
            min_confidence,
        )

        # 1. Load and sort historical data
        df = self._load_data()
        logger.info("Loaded %d historical games for %s", len(df), self.sport)

        # 2. Compute features on the FULL dataset once.
        #    Rolling features inside FeatureStore use shift(1), so there is no
        #    leakage from future games into any row's feature vector.
        store = FeatureStore()
        feature_df = store.compute(df, self.sport, use_cache=False)
        feature_cols = store.get_feature_columns(feature_df)

        X_all = feature_df[feature_cols].fillna(0)
        y_all = feature_df["result_over"].astype(int)
        dates_all = feature_df["game_date"]  # still aligned to feature_df index

        # 3. Split into n_folds equal-size time windows
        fold_edges = self._fold_edges(dates_all, n_folds)
        logger.info("Fold boundaries: %s", fold_edges)

        fold_results: list[FoldResult] = []
        all_probs: list[float] = []
        all_labels: list[int] = []
        all_per_game_returns: list[float] = []  # for Sharpe / drawdown

        # 4. Walk-forward loop — fold 0 is pure training, start predicting at fold 1
        for k in range(1, n_folds):
            fold_start = fold_edges[k]
            fold_end = fold_edges[k + 1]

            train_mask = dates_all < fold_start
            test_mask = (dates_all >= fold_start) & (dates_all < fold_end)

            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            X_test = X_all[test_mask]
            y_test = y_all[test_mask]

            n_train = len(X_train)
            n_test = len(X_test)

            if n_train < min_train_games:
                logger.info(
                    "Fold %d skipped: only %d training games (need %d)",
                    k,
                    n_train,
                    min_train_games,
                )
                continue

            if n_test == 0:
                logger.info("Fold %d skipped: no test games", k)
                continue

            logger.info(
                "Fold %d | train=%d | test=%d | %s → %s",
                k,
                n_train,
                n_test,
                fold_start,
                fold_end,
            )

            # 5. Train on a 85/15 train/val split within the training window
            val_start_idx = int(n_train * 0.85)
            X_tr = X_train.iloc[:val_start_idx]
            y_tr = y_train.iloc[:val_start_idx]
            X_val = X_train.iloc[val_start_idx:]
            y_val = y_train.iloc[val_start_idx:]

            model = OverUnderEnsemble(training_games=len(X_tr))
            model.fit(X_tr, y_tr, X_val, y_val)

            # 6. Predict
            probs = model.predict_proba(X_test)
            labels = y_test.values

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

            # 7. Fold metrics
            preds = (probs >= 0.5).astype(int)
            accuracy = float((preds == labels).mean())
            auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5
            fold_log_loss = float(log_loss(labels, probs))
            fold_brier = float(brier_score_loss(labels, probs))
            edge_mean = float(np.abs(probs - 0.5).mean())

            conf_mask = np.abs(probs - 0.5) * 2 >= 0.60
            high_conf_acc = (
                float((preds[conf_mask] == labels[conf_mask]).mean())
                if conf_mask.any()
                else float("nan")
            )

            # 8. Betting simulation
            conf_scores = np.abs(probs - 0.5) * 2  # 0=coin-flip, 1=certain
            bet_mask = conf_scores >= min_confidence

            flat_returns, kelly_returns = self._simulate_betting(probs[bet_mask], labels[bet_mask])
            all_per_game_returns.extend(flat_returns)

            roi_flat = (
                float(np.sum(flat_returns) / (np.sum(bet_mask) * _BET_SIZE))
                if bet_mask.any()
                else 0.0
            )
            roi_kelly = (
                float(np.sum(kelly_returns) / (np.sum(bet_mask) * _BET_SIZE))
                if bet_mask.any()
                else 0.0
            )

            fold_results.append(
                FoldResult(
                    fold=k,
                    start_date=str(fold_start),
                    end_date=str(fold_end),
                    n_games=n_test,
                    accuracy=accuracy,
                    auc=auc,
                    log_loss=fold_log_loss,
                    brier_score=fold_brier,
                    roi_flat=roi_flat,
                    roi_kelly=roi_kelly,
                    edge_mean=edge_mean,
                    high_conf_accuracy=high_conf_acc,
                )
            )
            logger.info(
                "Fold %d done — acc=%.3f | AUC=%.3f | ROI_flat=%.2f%%",
                k,
                accuracy,
                auc,
                roi_flat * 100,
            )

        if not fold_results:
            logger.warning("No folds produced results — check min_train_games or data size.")
            return self._empty_result(n_folds, min_confidence, status="no_folds_completed")

        # 9. Aggregate metrics across all folds
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        all_preds_arr = (all_probs_arr >= 0.5).astype(int)

        overall_accuracy = float((all_preds_arr == all_labels_arr).mean())
        overall_auc = (
            float(roc_auc_score(all_labels_arr, all_probs_arr))
            if len(np.unique(all_labels_arr)) > 1
            else 0.5
        )

        conf_all = np.abs(all_probs_arr - 0.5) * 2
        bet_all = conf_all >= min_confidence
        flat_all, kelly_all = self._simulate_betting(
            all_probs_arr[bet_all], all_labels_arr[bet_all]
        )
        total_bets = int(bet_all.sum())
        overall_roi_flat = (
            float(np.sum(flat_all) / (total_bets * _BET_SIZE)) if total_bets > 0 else 0.0
        )
        overall_roi_kelly = (
            float(np.sum(kelly_all) / (total_bets * _BET_SIZE)) if total_bets > 0 else 0.0
        )

        # 10. Sharpe and drawdown on per-game flat returns
        returns_arr = np.array(all_per_game_returns) / _BET_SIZE  # normalise to fractions
        sharpe = self._sharpe(returns_arr)
        max_dd = self._max_drawdown(flat_all)

        # 11. Calibration
        calibration = self._calibration(all_probs_arr, all_labels_arr)

        result = BacktestResult(
            sport=self.sport,
            n_folds=n_folds,
            min_confidence=min_confidence,
            total_games=len(all_labels),
            total_bets=total_bets,
            overall_accuracy=overall_accuracy,
            overall_auc=overall_auc,
            overall_roi_flat=overall_roi_flat,
            overall_roi_kelly=overall_roi_kelly,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            folds=fold_results,
            calibration=calibration,
        )

        logger.info(
            "Backtest complete — acc=%.3f | AUC=%.3f | ROI_flat=%.2f%% | Sharpe=%.2f | MaxDD=%.2f%%",
            overall_accuracy,
            overall_auc,
            overall_roi_flat * 100,
            sharpe,
            max_dd * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        path = self.data_dir / f"{self.sport}_historical.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Historical data not found: {path}. Run the ingestion pipeline first."
            )
        df = pd.read_parquet(path)
        df = df.dropna(subset=["result_over"])
        df = df.sort_values("game_date").reset_index(drop=True)
        return df

    def _fold_edges(self, dates: pd.Series, n_folds: int) -> list[Any]:
        """Return n_folds+1 boundary dates that divide the series into equal time windows."""
        unique_dates = dates.sort_values().unique()
        indices = np.linspace(0, len(unique_dates) - 1, n_folds + 1, dtype=int)
        return [unique_dates[i] for i in indices]

    def _simulate_betting(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """Return (flat_returns, kelly_returns) in dollars for each bet.

        Flat bet: $100 on every game selected by the caller.
        Kelly bet: Kelly fraction × $100, capped at quarter-Kelly.
        """
        flat_returns: list[float] = []
        kelly_returns: list[float] = []

        for p, label in zip(probs, labels):
            direction_over = p >= 0.5
            p_bet = p if direction_over else (1.0 - p)
            won = int(label) == int(direction_over)

            # Flat bet
            flat_returns.append(_BET_SIZE * _JUICE_PAYOFF if won else -_BET_SIZE)

            # Kelly fraction: f = (p*(b+1) - 1) / b, b = _JUICE_PAYOFF
            b = _JUICE_PAYOFF
            f = (p_bet * (b + 1) - 1) / b
            f = max(0.0, min(f, _KELLY_CAP))
            kelly_stake = f * _BET_SIZE
            kelly_returns.append(kelly_stake * _JUICE_PAYOFF if won else -kelly_stake)

        return flat_returns, kelly_returns

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Annualised Sharpe ratio (√252 scaling, per-game returns)."""
        if len(returns) < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(returns) / std * math.sqrt(252))

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        """Maximum peak-to-trough decline in cumulative P&L (as a fraction)."""
        if not returns:
            return 0.0
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / (np.abs(peak) + 1e-9)
        return float(drawdowns.max())

    @staticmethod
    def _calibration(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> dict[str, list[float]]:
        """Bin predictions into equal-width bins and compute actual win rate per bin."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_midpoints: list[float] = []
        actual_freqs: list[float] = []
        pred_probs: list[float] = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if i == n_bins - 1:  # include right edge for the last bin
                mask = (probs >= lo) & (probs <= hi)
            bin_midpoints.append(round(float((lo + hi) / 2), 3))
            if mask.any():
                actual_freqs.append(round(float(labels[mask].mean()), 4))
                pred_probs.append(round(float(probs[mask].mean()), 4))
            else:
                actual_freqs.append(None)
                pred_probs.append(None)

        return {
            "bin_midpoint": bin_midpoints,
            "actual_freq": actual_freqs,
            "predicted_prob": pred_probs,
        }

    def _empty_result(
        self, n_folds: int, min_confidence: float, status: str = "insufficient_data"
    ) -> BacktestResult:
        return BacktestResult(
            sport=self.sport,
            n_folds=n_folds,
            min_confidence=min_confidence,
            total_games=0,
            total_bets=0,
            overall_accuracy=0.0,
            overall_auc=0.5,
            overall_roi_flat=0.0,
            overall_roi_kelly=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            status=status,
            folds=[],
            calibration={},
        )
