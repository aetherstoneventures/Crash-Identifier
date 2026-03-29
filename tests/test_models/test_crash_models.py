"""
Tests for Crash Prediction Models

Tests all 5 models and ensemble.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from src.models.crash_prediction import (
    SVMCrashModel, RandomForestCrashModel, GradientBoostingCrashModel,
    NeuralNetworkCrashModel, EnsembleCrashModel
)


@pytest.fixture
def synthetic_data():
    """Create synthetic training data."""
    X, y = make_classification(
        n_samples=200,
        n_features=28,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    # Split into train, val, test
    X_train, X_val, X_test = X[:100], X[100:150], X[150:]
    y_train, y_val, y_test = y[:100], y[100:150], y[150:]
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestSVMModel:
    """Test SVM crash prediction model."""

    def test_initialization(self):
        """Test SVM initialization."""
        model = SVMCrashModel()
        assert model.name == 'SVM'
        assert not model.is_trained

    def test_training(self, synthetic_data):
        """Test SVM training."""
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = SVMCrashModel()
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained
        assert 'accuracy' in metrics

    def test_prediction(self, synthetic_data):
        """Test SVM prediction."""
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        model = SVMCrashModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, synthetic_data):
        """Test SVM probability prediction."""
        X_train, y_train, _, _, X_test, _ = synthetic_data
        model = SVMCrashModel()
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert len(proba) == len(X_test)
        assert all(0 <= p <= 1 for p in proba)

    def test_evaluation(self, synthetic_data):
        """Test SVM evaluation."""
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        model = SVMCrashModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 'recall' in metrics
        assert 'false_alarm_rate' in metrics


class TestRandomForestModel:
    """Test Random Forest crash prediction model."""

    def test_initialization(self):
        """Test RF initialization."""
        model = RandomForestCrashModel()
        assert model.name == 'RandomForest'
        assert not model.is_trained

    def test_training(self, synthetic_data):
        """Test RF training."""
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = RandomForestCrashModel()
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained
        assert 'accuracy' in metrics

    def test_feature_importance(self, synthetic_data):
        """Test RF feature importance."""
        X_train, y_train, _, _, _, _ = synthetic_data
        model = RandomForestCrashModel()
        model.train(X_train, y_train)
        importance = model.get_feature_importance()
        assert len(importance) == X_train.shape[1]
        assert all(i >= 0 for i in importance)


class TestGradientBoostingModel:
    """Test Gradient Boosting crash prediction model."""

    def test_initialization(self):
        """Test GB initialization."""
        model = GradientBoostingCrashModel()
        assert model.name == 'GradientBoosting'
        assert not model.is_trained

    def test_training(self, synthetic_data):
        """Test GB training."""
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = GradientBoostingCrashModel()
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained
        assert 'accuracy' in metrics

    def test_feature_importance(self, synthetic_data):
        """Test GB feature importance."""
        X_train, y_train, _, _, _, _ = synthetic_data
        model = GradientBoostingCrashModel()
        model.train(X_train, y_train)
        importance = model.get_feature_importance()
        assert len(importance) == X_train.shape[1]


class TestNeuralNetworkModel:
    """Test Neural Network crash prediction model."""

    def test_initialization(self):
        """Test NN initialization."""
        model = NeuralNetworkCrashModel()
        assert model.name == 'NeuralNetwork'
        assert not model.is_trained

    def test_training(self, synthetic_data):
        """Test NN training."""
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = NeuralNetworkCrashModel()
        metrics = model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained
        assert 'accuracy' in metrics

    def test_prediction(self, synthetic_data):
        """Test NN prediction."""
        X_train, y_train, _, _, X_test, _ = synthetic_data
        model = NeuralNetworkCrashModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestEnsembleModel:
    """Test Ensemble crash prediction model."""

    def test_initialization(self):
        """Test ensemble initialization."""
        model = EnsembleCrashModel()
        assert model.name == 'Ensemble'
        assert not model.is_trained

    def test_add_model(self, synthetic_data):
        """Test adding models to ensemble."""
        X_train, y_train, _, _, _, _ = synthetic_data
        ensemble = EnsembleCrashModel()
        
        # Train individual models
        svm = SVMCrashModel()
        svm.train(X_train, y_train)
        
        # Add to ensemble
        ensemble.add_model('svm', svm, weight=1.0)
        assert 'svm' in ensemble.models

    def test_ensemble_prediction(self, synthetic_data):
        """Test ensemble prediction."""
        X_train, y_train, X_val, y_val, X_test, _ = synthetic_data
        ensemble = EnsembleCrashModel()
        
        # Train individual models
        svm = SVMCrashModel()
        svm.train(X_train, y_train)
        rf = RandomForestCrashModel()
        rf.train(X_train, y_train)
        
        # Add to ensemble
        ensemble.add_model('svm', svm)
        ensemble.add_model('rf', rf)
        
        # Train ensemble
        ensemble.train(X_train, y_train, X_val, y_val)
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_optimal_weights(self, synthetic_data):
        """Test optimal weight calculation."""
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        ensemble = EnsembleCrashModel()
        
        # Train individual models
        svm = SVMCrashModel()
        svm.train(X_train, y_train)
        rf = RandomForestCrashModel()
        rf.train(X_train, y_train)
        
        # Add to ensemble
        ensemble.add_model('svm', svm)
        ensemble.add_model('rf', rf)
        
        # Calculate optimal weights
        ensemble.calculate_optimal_weights(X_val, y_val)
        weights = ensemble.get_model_weights()
        
        assert 'svm' in weights
        assert 'rf' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Weights sum to 1


class TestCrashLabeler:
    """Test crash label generation."""

    def test_label_generation(self):
        """Test crash label generation."""
        from src.models.crash_prediction.crash_labeler import CrashLabeler
        
        # Create synthetic price series with a crash
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                          120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
                          110, 109, 108, 107, 106, 105, 104, 103, 102, 101,
                          100, 99, 98, 97, 96, 95, 94, 93, 92, 91,
                          90, 89, 88, 87, 86, 85, 84, 83, 82, 81,
                          80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
        
        prices_series = pd.Series(prices)
        labeler = CrashLabeler(drawdown_threshold=0.20)
        labels = labeler.generate_labels(prices_series)
        
        assert len(labels) == len(prices)
        assert all(l in [0, 1] for l in labels)

    def test_crash_statistics(self):
        """Test crash statistics."""
        from src.models.crash_prediction.crash_labeler import CrashLabeler
        import pandas as pd
        
        prices = np.array([100] * 100)
        prices_series = pd.Series(prices)
        labeler = CrashLabeler()
        labels = labeler.generate_labels(prices_series)
        stats = labeler.get_crash_statistics(prices_series, labels)
        
        assert 'total_samples' in stats
        assert 'crash_samples' in stats
        assert 'crash_ratio' in stats


# Import pandas for crash labeler test
import pandas as pd

