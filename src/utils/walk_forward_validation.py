"""Walk-forward validation for time-series models.

Implements expanding-window walk-forward validation with:
- Configurable purge gap to prevent train/test leakage at boundaries
- Per-fold metrics tracking with variance analysis
- Support for per-fold scaler fitting
- Integration with SMOTE for class imbalance (optional)
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Expanding-window walk-forward validator for time-series data.

    Generates chronological train/test splits where:
    - Training always starts at index 0 (expanding window)
    - An optional purge gap separates train and test to avoid leakage
    - Test window slides forward by step_size after each fold
    """

    def __init__(
        self,
        window_size: int = 252,
        step_size: int = 21,
        min_train_size: int = 1260,
        purge_gap: int = 0,
    ):
        """
        Args:
            window_size: Number of samples in each test window (~1 year)
            step_size: How far to advance the test window each fold (~1 month)
            min_train_size: Minimum training samples before first split (~5 years)
            purge_gap: Samples to skip between train end and test start
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.purge_gap = purge_gap

    def split(
        self,
        n_samples: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding-window train/test splits.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        splits = []
        train_end = self.min_train_size

        while train_end + self.purge_gap + self.window_size <= n_samples:
            train_idx = np.arange(0, train_end)
            test_start = train_end + self.purge_gap
            test_end = min(test_start + self.window_size, n_samples)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))
            train_end += self.step_size

        logger.info(
            f"Generated {len(splits)} walk-forward splits "
            f"(min_train={self.min_train_size}, "
            f"window={self.window_size}, "
            f"step={self.step_size}, "
            f"purge={self.purge_gap})"
        )
        return splits

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        feature_pipeline=None,
        use_smote: bool = False,
    ) -> Dict:
        """Run full walk-forward validation.

        For each fold:
        1. Split into train/test chronologically
        2. Optionally fit feature pipeline per fold (proper scaling)
        3. Optionally apply SMOTE to training fold
        4. Train a fresh model
        5. Evaluate on test fold

        Args:
            X: Features DataFrame (raw or pre-computed indicators).
            y: Labels Series aligned with X.
            model_factory: Callable returning a fresh, untrained model.
                Model must have train(X, y) and predict_proba(X) methods.
            feature_pipeline: Optional FeaturePipeline for per-fold fitting.
                If provided, X should be RAW data (not yet scaled).
            use_smote: Whether to apply SMOTE oversampling per fold.

        Returns:
            Dict with 'fold_metrics' (list of per-fold dicts) and
            'aggregate' (mean/std across folds).
        """
        splits = self.split(len(X))
        if not splits:
            raise ValueError(
                f"No splits generated. Data has {len(X)} samples, "
                f"need at least {self.min_train_size + self.purge_gap + self.window_size}."
            )

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            # Per-fold feature pipeline fitting
            if feature_pipeline is not None:
                X_train = feature_pipeline.fit_transform(X_train)
                X_test = feature_pipeline.transform(X_test)

            # Per-fold SMOTE (only if positive samples exist)
            if use_smote and y_train.sum() > 5:
                try:
                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE(random_state=42)
                    X_train_arr, y_train_arr = sm.fit_resample(
                        X_train, y_train
                    )
                    X_train = pd.DataFrame(
                        X_train_arr, columns=X_train.columns
                    )
                    y_train = pd.Series(y_train_arr)
                except ImportError:
                    logger.warning(
                        "imbalanced-learn not installed; skipping SMOTE"
                    )

            # Train fresh model
            model = model_factory()
            model.train(X_train, y_train)

            # Evaluate
            metrics = _evaluate_fold(model, X_test, y_test, fold_idx)
            metrics['train_size'] = len(train_idx)
            metrics['test_size'] = len(test_idx)
            metrics['train_positive_rate'] = float(y_train.mean())
            metrics['test_positive_rate'] = float(y_test.mean())
            fold_metrics.append(metrics)

            if fold_idx % 10 == 0:
                logger.info(
                    f"  Fold {fold_idx}/{len(splits)}: "
                    f"AUC={metrics.get('auc', 0):.3f}, "
                    f"Recall={metrics.get('recall', 0):.3f}"
                )

        aggregate = _aggregate_metrics(fold_metrics)

        logger.info(
            f"Walk-forward complete: {len(splits)} folds, "
            f"Mean AUC={aggregate['auc_mean']:.3f} "
            f"(±{aggregate['auc_std']:.3f}), "
            f"Mean Recall={aggregate['recall_mean']:.3f} "
            f"(±{aggregate['recall_std']:.3f})"
        )

        return {
            'fold_metrics': fold_metrics,
            'aggregate': aggregate,
            'n_folds': len(splits),
        }


def _evaluate_fold(
    model, X_test: pd.DataFrame, y_test: pd.Series, fold_idx: int
) -> Dict:
    """Evaluate a model on a single fold.

    Returns dict with standard classification metrics.
    Handles edge cases (no positive samples, constant predictions).
    """
    metrics = {'fold': fold_idx}

    try:
        y_proba = model.predict_proba(X_test)
        if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    except Exception as e:
        logger.warning(f"Fold {fold_idx}: predict failed: {e}")
        return metrics

    y_true = np.asarray(y_test)

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    metrics['recall'] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))

    # AUC requires both classes present
    if len(np.unique(y_true)) >= 2:
        metrics['auc'] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics['auc'] = np.nan

    return metrics


def _aggregate_metrics(fold_metrics: List[Dict]) -> Dict:
    """Compute mean and std of metrics across folds."""
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    agg = {}

    for key in metric_keys:
        values = [
            m[key] for m in fold_metrics
            if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))
        ]
        if values:
            agg[f'{key}_mean'] = float(np.mean(values))
            agg[f'{key}_std'] = float(np.std(values))
        else:
            agg[f'{key}_mean'] = 0.0
            agg[f'{key}_std'] = 0.0

    return agg
