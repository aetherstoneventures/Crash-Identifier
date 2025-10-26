"""
Support Vector Machine (SVM) Crash Prediction Model

Implements SVM with RBF kernel for crash prediction.
"""

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.models.crash_prediction.base_model import BaseCrashModel
from src.utils.config import RANDOM_STATE

logger = logging.getLogger(__name__)


class SVMCrashModel(BaseCrashModel):
    """Support Vector Machine for crash prediction."""

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        Initialize SVM model.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        super().__init__('SVM')
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=RANDOM_STATE,
            verbose=0
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              hyperparameter_tuning: bool = False) -> Dict:
        """
        Train SVM model.

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
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
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
                self.logger.info(f"Best SVM params: {grid_search.best_params_}")
            else:
                # Train with current parameters
                self.model.fit(X_train, y_train)

            self.is_trained = True

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                metrics = self.evaluate(X_val, y_val)
                self.logger.info(f"SVM validation metrics: {metrics}")
                return metrics

            return {'status': 'trained'}

        except Exception as e:
            self.logger.error(f"SVM training failed: {e}")
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

