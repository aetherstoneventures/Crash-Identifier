"""MLP Bottom Prediction Model."""

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from src.models.bottom_prediction.base_bottom_model import BaseBottomModel


class MLPBottomModel(BaseBottomModel):
    """Multi-Layer Perceptron for bottom prediction."""

    def __init__(self):
        """Initialize MLP bottom model."""
        super().__init__("MLP_Bottom")
        self.scaler = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train MLP model with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels (days to bottom)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training metrics
        """
        try:
            # Handle NaN values
            X_train = np.nan_to_num(X_train, nan=0.0)
            if X_val is not None:
                X_val = np.nan_to_num(X_val, nan=0.0)

            # Hyperparameter grid
            param_grid = {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [500, 1000],
            }

            # GridSearchCV
            base_model = MLPRegressor(
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=0
            )

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.is_trained = True

            # Training metrics
            train_pred = self.model.predict(X_train)
            train_r2 = np.corrcoef(y_train, train_pred)[0, 1] ** 2

            metrics = {
                'train_r2': train_r2,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
            }

            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            self.logger.info(f"MLP training complete. R²: {train_r2:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"MLP training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predicted days to bottom
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

