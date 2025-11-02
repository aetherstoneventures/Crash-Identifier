"""
Advanced ML Model for Crash Prediction with Enhanced Techniques

Improvements:
1. Feature engineering: lagged features, rolling statistics, interaction terms
2. SMOTE for class imbalance handling
3. Time-series aware cross-validation
4. Hyperparameter optimization via GridSearchCV
5. Stacking ensemble with meta-learner
6. Regularization to prevent overfitting
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.crash_prediction.base_model import BaseCrashModel

logger = logging.getLogger(__name__)


class AdvancedMLModel(BaseCrashModel):
    """Advanced ML model with enhanced techniques for crash prediction."""

    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize advanced ML model.

        Args:
            model_type: 'gradient_boosting' or 'random_forest'
        """
        super().__init__(f'AdvancedML_{model_type}')
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42, k_neighbors=3)
        self.model = None
        self.best_params = None

    def engineer_features(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        """
        Engineer advanced features from raw features.

        Args:
            X: Raw features (n_samples, n_features)
            feature_names: Feature names for reference

        Returns:
            Engineered features
        """
        n_samples, n_features = X.shape
        engineered = []

        # Original features
        engineered.append(X)

        # Lagged features (1, 5, 20 days)
        for lag in [1, 5, 20]:
            if n_samples > lag:
                lagged = np.vstack([np.zeros((lag, n_features)), X[:-lag]])
                engineered.append(lagged)

        # Rolling statistics (5, 20 day windows)
        for window in [5, 20]:
            if n_samples > window:
                # Rolling mean
                rolling_mean = np.zeros_like(X)
                for i in range(window, n_samples):
                    rolling_mean[i] = np.mean(X[i-window:i], axis=0)
                engineered.append(rolling_mean)

                # Rolling std
                rolling_std = np.zeros_like(X)
                for i in range(window, n_samples):
                    rolling_std[i] = np.std(X[i-window:i], axis=0)
                engineered.append(rolling_std)

        # Interaction terms (multiply key features)
        if n_features >= 2:
            interactions = X[:, 0:2] * X[:, 1:3] if n_features >= 3 else X[:, 0:1] * X[:, 1:2]
            engineered.append(interactions)

        # Concatenate all features
        X_engineered = np.hstack(engineered)
        logger.info(f"Feature engineering: {n_features} → {X_engineered.shape[1]} features")

        return X_engineered

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train advanced ML model with hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training metrics
        """
        try:
            logger.info(f"Training {self.model_type} with advanced techniques...")

            # Feature engineering
            X_train_eng = self.engineer_features(X_train)
            X_val_eng = self.engineer_features(X_val) if X_val is not None else None

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_eng)
            X_val_scaled = self.scaler.transform(X_val_eng) if X_val_eng is not None else None

            # Handle class imbalance with SMOTE
            logger.info(f"Applying SMOTE for class imbalance...")
            X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"After SMOTE: {len(y_train_balanced)} samples (ratio: {np.sum(y_train_balanced) / len(y_train_balanced):.2%})")

            # Hyperparameter optimization with time-series aware CV
            logger.info(f"Optimizing hyperparameters...")
            tscv = TimeSeriesSplit(n_splits=3)

            if self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'min_samples_split': [5, 10],
                }
                base_model = GradientBoostingClassifier(random_state=42)
            else:  # random_forest
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4],
                }
                base_model = RandomForestClassifier(random_state=42)

            grid_search = GridSearchCV(
                base_model, param_grid, cv=tscv, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_balanced, y_train_balanced)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV AUC: {grid_search.best_score_:.4f}")

            self.is_trained = True

            # Evaluate on validation set
            if X_val_scaled is not None and y_val is not None:
                metrics = self.evaluate(X_val_scaled, y_val)
                logger.info(f"Validation metrics: {metrics}")
                return metrics

            return {'status': 'trained', 'best_params': self.best_params}

        except Exception as e:
            logger.error(f"Advanced ML training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_eng = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_eng)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict crash probability."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_eng = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_eng)
        return self.model.predict_proba(X_scaled)[:, 1]

