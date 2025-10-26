"""Base class for bottom prediction models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseBottomModel(ABC):
    """Abstract base class for bottom prediction models."""

    def __init__(self, name: str):
        """
        Initialize base bottom model.

        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (days to bottom)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predicted days to bottom
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 tolerance_days: int = 30) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels (days to bottom)
            tolerance_days: Tolerance for accuracy calculation (default: 30)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Handle NaN values
        X_test = np.nan_to_num(X_test, nan=0.0)

        y_pred = self.predict(X_test)

        # Handle case where predictions are empty
        if len(y_pred) == 0:
            return {
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'accuracy': 0.0,
                'mean_error_days': np.nan,
            }

        # Ensure y_test and y_pred have same length
        min_len = min(len(y_test), len(y_pred))
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]

        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Accuracy within tolerance
        abs_errors = np.abs(y_test - y_pred)
        accuracy = np.mean(abs_errors <= tolerance_days)

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,  # % within tolerance
            'mean_error_days': np.mean(y_test - y_pred),
        }

        self.metrics = metrics
        return metrics

    def get_metrics(self) -> Dict:
        """Get last evaluation metrics."""
        return self.metrics

