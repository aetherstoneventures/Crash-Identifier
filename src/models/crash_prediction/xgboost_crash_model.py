"""Advanced XGBoost model for crash prediction with Optuna hyperparameter optimization.

This model uses:
- XGBoost with early stopping
- Optuna for hyperparameter optimization
- Walk-forward validation
- SHAP for interpretability
- Proper handling of class imbalance
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.models.crash_prediction.base_model import BaseCrashModel
from src.utils.config import (
    XGBOOST_MAX_DEPTH,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_EARLY_STOPPING_ROUNDS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)


class XGBoostCrashModel(BaseCrashModel):
    """XGBoost-based crash prediction model with Optuna optimization."""
    
    def __init__(
        self,
        optimize_hyperparams: bool = True,
        n_trials: int = OPTUNA_N_TRIALS,
        timeout: int = OPTUNA_TIMEOUT
    ):
        """Initialize XGBoost crash model.
        
        Args:
            optimize_hyperparams: Whether to optimize hyperparameters with Optuna
            n_trials: Number of Optuna trials
            timeout: Timeout for Optuna optimization (seconds)
        """
        super().__init__(name="XGBoost")
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.timeout = timeout
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.shap_values = None
        self.feature_names = None
    
    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Validation AUC score
        """
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
            'random_state': RANDOM_STATE,
            'tree_method': 'hist',
            'eval_metric': 'auc'
        }
        
        # Train model
        model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return auc
    
    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Optimizing hyperparameters with Optuna ({self.n_trials} trials)...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._objective(
                trial,
                X_train.values,
                y_train.values,
                X_val.values,
                y_val.values
            ),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training XGBoost crash prediction model...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Optimize hyperparameters if requested
        if self.optimize_hyperparams and X_val is not None and y_val is not None:
            self.best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            # Use default parameters
            self.best_params = {
                'max_depth': XGBOOST_MAX_DEPTH,
                'learning_rate': XGBOOST_LEARNING_RATE,
                'n_estimators': XGBOOST_N_ESTIMATORS,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'scale_pos_weight': 10.0,  # Handle class imbalance
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                'eval_metric': 'auc'
            }
        
        # Train final model
        eval_set = [(X_val.values, y_val.values)] if X_val is not None and y_val is not None else None
        if eval_set:
            self.best_params['early_stopping_rounds'] = XGBOOST_EARLY_STOPPING_ROUNDS
        self.model = xgb.XGBClassifier(**self.best_params)
        
        self.model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_set,
            verbose=True
        )
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features:\n{self.feature_importance.head(10)}")
        
        # Calculate SHAP values for interpretability
        try:
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_train.values[:1000])  # Sample for speed
            logger.info("SHAP values calculated successfully")
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values: {e}")
            self.shap_values = None
        
        # Calculate metrics
        y_train_pred = self.model.predict(X_train.values)
        y_train_proba = self.model.predict_proba(X_train.values)[:, 1]
        
        metrics = {
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0)
        }
        
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val.values)
            y_val_proba = self.model.predict_proba(X_val.values)[:, 1]
            
            metrics.update({
                'val_auc': roc_auc_score(y_val, y_val_proba),
                'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_val_pred, zero_division=0)
            })
        
        logger.info(f"Training complete. Metrics: {metrics}")
        
        return {
            'metrics': metrics,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance.to_dict('records'),
            'model_type': 'XGBoost'
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary crash predictions (0 or 1).
        
        Args:
            X: Features to predict on
            
        Returns:
            Binary predictions array
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crash probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of crash probabilities (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X.values)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance.
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.feature_importance
    
    def get_shap_values(self) -> Optional[np.ndarray]:
        """Get SHAP values for interpretability.
        
        Returns:
            SHAP values array or None if not calculated
        """
        return self.shap_values

