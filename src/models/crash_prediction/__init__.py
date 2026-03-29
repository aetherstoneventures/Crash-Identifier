"""
Crash Prediction Models

Implements advanced machine learning models for predicting market crashes:
1. LSTM with attention mechanism
2. XGBoost with Optuna optimization
3. Improved statistical model

NOTE: Imports are lazy to avoid mutex lock issues with TensorFlow/XGBoost.
Import models directly when needed:
    from src.models.crash_prediction.base_model import BaseCrashModel
    from src.models.crash_prediction.lstm_crash_model import LSTMCrashModel
    from src.models.crash_prediction.xgboost_crash_model import XGBoostCrashModel
    from src.models.crash_prediction.improved_statistical_model import ImprovedStatisticalModel
    from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
"""

# Removed eager imports to avoid mutex lock issues
# Import models directly when needed

__all__ = [
    'BaseCrashModel',
    'LSTMCrashModel',
    'XGBoostCrashModel',
    'ImprovedStatisticalModel',
    'StatisticalModelV3',
]

