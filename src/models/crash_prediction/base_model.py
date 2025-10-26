"""
Base Crash Prediction Model

Abstract base class for all crash prediction models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)

logger = logging.getLogger(__name__)


class BaseCrashModel(ABC):
    """Abstract base class for crash prediction models."""

    def __init__(self, name: str):
        """
        Initialize base model.

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
            y_train: Training labels
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
            Predicted labels (0 or 1)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict crash probability.

        Args:
            X: Features

        Returns:
            Crash probability (0-1)
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Handle NaN values
        X_test = np.nan_to_num(X_test, nan=0.0)

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba),
        }

        # Calculate false alarm rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        self.metrics = metrics
        return metrics

    def get_metrics(self) -> Dict:
        """Get model metrics."""
        return self.metrics.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(trained={self.is_trained})"

