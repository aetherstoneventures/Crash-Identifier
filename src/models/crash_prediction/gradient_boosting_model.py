"""
Gradient Boosting Crash Prediction Model

Implements Gradient Boosting for crash prediction.
"""

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from src.models.crash_prediction.base_model import BaseCrashModel
from src.utils.config import RANDOM_STATE

logger = logging.getLogger(__name__)


class GradientBoostingCrashModel(BaseCrashModel):
    """Gradient Boosting for crash prediction."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, subsample: float = 0.8):
        """
        Initialize Gradient Boosting model.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate (shrinkage)
            max_depth: Maximum tree depth
            subsample: Fraction of samples for fitting
        """
        super().__init__('GradientBoosting')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=RANDOM_STATE,
            verbose=0
        )
        self.feature_importance = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              hyperparameter_tuning: bool = False) -> Dict:
        """
        Train Gradient Boosting model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparameter_tuning: Whether to perform grid search

        Returns:
            Dictionary with training metrics
        """
        try:
            # Handle NaN values
            X_train = np.nan_to_num(X_train, nan=0.0)
            if X_val is not None:
                X_val = np.nan_to_num(X_val, nan=0.0)

            if hyperparameter_tuning and X_val is not None and y_val is not None:
                # Grid search for optimal hyperparameters
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                }
                grid_search = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.logger.info(f"Best GB params: {grid_search.best_params_}")
            else:
                # Train with current parameters
                self.model.fit(X_train, y_train)

            # Extract feature importance
            self.feature_importance = self.model.feature_importances_

            self.is_trained = True

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                metrics = self.evaluate(X_val, y_val)
                self.logger.info(f"GB validation metrics: {metrics}")
                return metrics

            return {'status': 'trained'}

        except Exception as e:
            self.logger.error(f"Gradient Boosting training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict crash probability.

        Args:
            X: Features

        Returns:
            Crash probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = np.nan_to_num(X, nan=0.0)
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of crash (class 1)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance is None:
            raise ValueError("Model must be trained before getting feature importance")
        return self.feature_importance

