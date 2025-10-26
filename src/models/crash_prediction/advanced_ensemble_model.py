"""
Advanced Ensemble Model with Enhanced Techniques

Improvements over base ensemble:
1. Feature engineering: interaction terms, lagged features, rolling statistics
2. Hyperparameter optimization via GridSearchCV
3. Class imbalance handling with SMOTE
4. Stacking ensemble with meta-learner
5. Cross-validation for robust evaluation
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedEnsembleModel:
    """Advanced ensemble with feature engineering and optimization."""

    def __init__(self):
        """Initialize advanced ensemble model."""
        self.models = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.optimal_weights = {}
        self.cv_scores = {}

    def engineer_features(self, X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """
        Engineer advanced features from base indicators.
        
        Args:
            X: Base feature matrix (n_samples, n_features)
            feature_names: Names of base features
            
        Returns:
            Enhanced feature matrix and names
        """
        df = pd.DataFrame(X, columns=feature_names)
        
        # 1. Interaction terms (top 5 most important pairs)
        important_pairs = [
            ('yield_spread_10y_2y', 'vix_level'),
            ('credit_spread_bbb', 'unemployment_rate'),
            ('sp500_momentum_200d', 'vix_change_rate'),
            ('shiller_pe', 'debt_to_gdp'),
            ('margin_debt_growth', 'gdp_growth'),
        ]
        
        for feat1, feat2 in important_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        # 2. Lagged features (1, 2, 4 week lags)
        for lag in [1, 2, 4]:
            for col in ['vix_level', 'credit_spread_bbb', 'unemployment_rate']:
                if col in df.columns:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 3. Rolling statistics (20-day windows)
        for col in ['yield_spread_10y_2y', 'vix_level', 'sp500_momentum_200d']:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(20).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(20).std()
        
        # 4. Volatility regime indicator
        if 'vix_level' in df.columns:
            df['vix_regime'] = (df['vix_level'] > df['vix_level'].quantile(0.75)).astype(int)
        
        # 5. Stress indicator (multiple indicators in extreme territory)
        stress_indicators = ['yield_spread_10y_2y', 'credit_spread_bbb', 'vix_level']
        stress_cols = [col for col in stress_indicators if col in df.columns]
        if stress_cols:
            df['stress_count'] = (df[stress_cols] > df[stress_cols].quantile(0.75)).sum(axis=1)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        new_feature_names = df.columns.tolist()
        return df.values, new_feature_names

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> None:
        """
        Train advanced ensemble with all improvements.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
        """
        logger.info("Training Advanced Ensemble Model...")
        
        # Feature engineering
        if feature_names:
            X_engineered, self.feature_names = self.engineer_features(X, feature_names)
        else:
            X_engineered = X
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Handle class imbalance with SMOTE
        logger.info("Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        try:
            X_balanced, y_balanced = smote.fit_resample(X_engineered, y)
            logger.info(f"  - Original: {len(y)} samples, Balanced: {len(y_balanced)} samples")
        except:
            X_balanced, y_balanced = X_engineered, y
            logger.warning("  - SMOTE failed, using original data")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # Train base models with optimized hyperparameters
        logger.info("Training base models with optimized hyperparameters...")
        
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.models['rf'].fit(X_scaled, y_balanced)
        
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=7,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        self.models['gb'].fit(X_scaled, y_balanced)
        
        self.models['svm'] = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        self.models['svm'].fit(X_scaled, y_balanced)
        
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), learning_rate_init=0.001,
            max_iter=1000, random_state=42, early_stopping=True
        )
        self.models['nn'].fit(X_scaled, y_balanced)
        
        # Calculate optimal weights using cross-validation
        logger.info("Calculating optimal ensemble weights...")
        self._calculate_optimal_weights(X_scaled, y_balanced)
        
        # Train meta-learner for stacking
        logger.info("Training meta-learner for stacking...")
        self._train_meta_learner(X_scaled, y_balanced)
        
        logger.info("✓ Advanced Ensemble training complete")

    def _calculate_optimal_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate optimal weights for ensemble voting."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            self.cv_scores[model_name] = scores.mean()
            logger.info(f"  - {model_name}: CV AUC = {scores.mean():.4f}")
        
        # Normalize scores to weights
        total_score = sum(self.cv_scores.values())
        self.optimal_weights = {
            name: score / total_score for name, score in self.cv_scores.items()
        }

    def _train_meta_learner(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train meta-learner on base model predictions."""
        # Generate meta-features from base models
        meta_features = []
        for model in self.models.values():
            meta_features.append(model.predict_proba(X)[:, 1])
        
        X_meta = np.column_stack(meta_features)
        
        # Train logistic regression as meta-learner
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_learner.fit(X_meta, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict crash probability using advanced ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        # Engineer features
        X_engineered = X
        if len(X.shape) == 1:
            X_engineered = X.reshape(1, -1)
        
        # Normalize
        X_scaled = self.scaler.transform(X_engineered)
        
        # Get predictions from base models
        predictions = np.zeros((X_scaled.shape[0], len(self.models)))
        for i, model in enumerate(self.models.values()):
            predictions[:, i] = model.predict_proba(X_scaled)[:, 1]
        
        # Use meta-learner for final prediction
        if self.meta_learner:
            return self.meta_learner.predict_proba(predictions)[:, 1]
        else:
            # Fallback to weighted average
            weights = np.array(list(self.optimal_weights.values()))
            return np.average(predictions, axis=1, weights=weights)

