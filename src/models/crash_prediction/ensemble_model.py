"""
Ensemble Crash Prediction Model

Combines multiple models with weighted voting for improved predictions.
"""

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.crash_prediction.base_model import BaseCrashModel

logger = logging.getLogger(__name__)


class EnsembleCrashModel(BaseCrashModel):
    """Ensemble model combining multiple crash prediction models."""

    def __init__(self):
        """Initialize ensemble model."""
        super().__init__('Ensemble')
        self.models = {}
        self.weights = {}
        self.model = None  # Ensemble doesn't have a single model

    def add_model(self, name: str, model: BaseCrashModel, weight: float = 1.0):
        """
        Add model to ensemble.

        Args:
            name: Model name
            model: Trained model instance
            weight: Model weight in ensemble
        """
        if not model.is_trained:
            raise ValueError(f"Model {name} must be trained before adding to ensemble")
        self.models[name] = model
        self.weights[name] = weight
        self.logger.info(f"Added {name} to ensemble with weight {weight}")

    def calculate_optimal_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Calculate optimal weights based on validation AUC.

        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        aucs = {}
        for name, model in self.models.items():
            try:
                pred = model.predict_proba(X_val)
                auc = roc_auc_score(y_val, pred)
                aucs[name] = auc
                self.logger.info(f"{name} validation AUC: {auc:.4f}")
            except Exception as e:
                self.logger.error(f"Error calculating AUC for {name}: {e}")
                aucs[name] = 0.5

        # Normalize weights by AUC
        total_auc = sum(aucs.values())
        if total_auc > 0:
            self.weights = {name: auc / total_auc for name, auc in aucs.items()}
        else:
            # Equal weights if all AUCs are 0
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models}

        self.logger.info(f"Optimal ensemble weights: {self.weights}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train ensemble (assumes individual models are already trained).

        Args:
            X_train: Training features (not used)
            y_train: Training labels (not used)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with training metrics
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Mark as trained first (needed for evaluate)
        self.is_trained = True

        # Calculate optimal weights if validation data provided
        if X_val is not None and y_val is not None:
            self.calculate_optimal_weights(X_val, y_val)
            metrics = self.evaluate(X_val, y_val)
            self.logger.info(f"Ensemble validation metrics: {metrics}")
            return metrics

        return {'status': 'ensemble_ready'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features

        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict crash probability using weighted ensemble.

        Args:
            X: Features

        Returns:
            Crash probability (0-1)
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = np.zeros(len(X))
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            pred = model.predict_proba(X)
            predictions += (weight / total_weight) * pred

        return predictions

    def get_model_weights(self) -> Dict[str, float]:
        """Get ensemble model weights."""
        return self.weights.copy()

    def get_model_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all models in ensemble."""
        metrics = {}
        for name, model in self.models.items():
            metrics[name] = model.get_metrics()
        return metrics

