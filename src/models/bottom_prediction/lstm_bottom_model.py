"""LSTM Bottom Prediction Model (using sklearn-compatible approach)."""

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from src.models.bottom_prediction.base_bottom_model import BaseBottomModel


class LSTMBottomModel(BaseBottomModel):
    """LSTM-inspired model for bottom prediction using time series features."""

    def __init__(self, sequence_length: int = 30):
        """
        Initialize LSTM bottom model.

        Args:
            sequence_length: Length of input sequences
        """
        super().__init__("LSTM_Bottom")
        self.sequence_length = sequence_length
        self.scaler = None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Create sequences for LSTM-like processing.

        Args:
            X: Features
            y: Labels

        Returns:
            Tuple of (X_seq, y_seq)
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length].flatten())
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train LSTM-inspired model.

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

            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)

            if len(X_train_seq) == 0:
                raise ValueError("Not enough data to create sequences")

            # Use Ridge regression with hyperparameter tuning
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            }

            base_model = Ridge()
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            grid_search.fit(X_train_seq, y_train_seq)
            self.model = grid_search.best_estimator_
            self.is_trained = True

            # Training metrics
            train_pred = self.model.predict(X_train_seq)
            train_r2 = np.corrcoef(y_train_seq, train_pred)[0, 1] ** 2

            metrics = {
                'train_r2': train_r2,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
            }

            # Validation metrics if provided
            if X_val is not None and y_val is not None and len(X_val) > self.sequence_length:
                val_metrics = self.evaluate(X_val, y_val)
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            self.logger.info(f"LSTM training complete. R²: {train_r2:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
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

        # Create sequences
        X_seq, _ = self._create_sequences(X, np.zeros(len(X)))

        if len(X_seq) == 0:
            return np.array([])

        predictions = self.model.predict(X_seq)
        return predictions

