"""
Crash Prediction Models

Implements 5 machine learning models for predicting market crashes:
1. Support Vector Machine (SVM)
2. Random Forest
3. Gradient Boosting
4. Neural Network (MLP)
5. Ensemble (weighted voting)
"""

from src.models.crash_prediction.base_model import BaseCrashModel
from src.models.crash_prediction.svm_model import SVMCrashModel
from src.models.crash_prediction.random_forest_model import RandomForestCrashModel
from src.models.crash_prediction.gradient_boosting_model import GradientBoostingCrashModel
from src.models.crash_prediction.neural_network_model import NeuralNetworkCrashModel
from src.models.crash_prediction.ensemble_model import EnsembleCrashModel
from src.models.crash_prediction.statistical_model import StatisticalCrashModel

__all__ = [
    'BaseCrashModel',
    'SVMCrashModel',
    'RandomForestCrashModel',
    'GradientBoostingCrashModel',
    'NeuralNetworkCrashModel',
    'EnsembleCrashModel',
    'StatisticalCrashModel',
]

