"""
Neural Network (MLP) Crash Prediction Model

Implements Multi-Layer Perceptron for crash prediction.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from src.models.crash_prediction.base_model import BaseCrashModel
from src.utils.config import RANDOM_STATE

logger = logging.getLogger(__name__)


class NeuralNetworkCrashModel(BaseCrashModel):
    """Neural Network (MLP) for crash prediction."""

    def __init__(self, hidden_layers: Tuple[int, ...] = (128, 64, 32),
                 learning_rate: float = 0.001, max_iter: int = 1000,
                 early_stopping: bool = True):
        """
        Initialize Neural Network model.

        Args:
            hidden_layers: Tuple of hidden layer sizes
            learning_rate: Learning rate
            max_iter: Maximum iterations
            early_stopping: Whether to use early stopping
        """
        super().__init__('NeuralNetwork')
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=0,
            n_iter_no_change=20
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              hyperparameter_tuning: bool = False) -> Dict:
        """
        Train Neural Network model.

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
                    'hidden_layer_sizes': [
                        (64, 32),
                        (128, 64),
                        (128, 64, 32),
                        (256, 128, 64),
                    ],
                    'learning_rate_init': [0.0001, 0.001, 0.01],
                    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
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
                self.logger.info(f"Best NN params: {grid_search.best_params_}")
            else:
                # Train with current parameters
                self.model.fit(X_train, y_train)

            self.is_trained = True

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                metrics = self.evaluate(X_val, y_val)
                self.logger.info(f"NN validation metrics: {metrics}")
                return metrics

            return {'status': 'trained'}

        except Exception as e:
            self.logger.error(f"Neural Network training failed: {e}")
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

