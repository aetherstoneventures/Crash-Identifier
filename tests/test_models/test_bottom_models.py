"""Tests for bottom prediction models."""

import pytest
import numpy as np
import pandas as pd
from src.models.bottom_prediction import (
    MLPBottomModel, LSTMBottomModel, BottomLabeler
)


@pytest.fixture
def sample_features():
    """Generate sample features."""
    np.random.seed(42)
    return np.random.randn(200, 28)


@pytest.fixture
def sample_prices():
    """Generate sample price series."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    return pd.Series(prices)


@pytest.fixture
def sample_labels(sample_prices):
    """Generate sample bottom labels."""
    labeler = BottomLabeler()
    return labeler.generate_labels(sample_prices)


class TestBottomLabeler:
    """Test BottomLabeler."""

    def test_initialization(self):
        """Test labeler initialization."""
        labeler = BottomLabeler(lookback_window=60, lookforward_window=60)
        assert labeler.lookback_window == 60
        assert labeler.lookforward_window == 60

    def test_label_generation(self, sample_prices):
        """Test label generation."""
        labeler = BottomLabeler()
        labels = labeler.generate_labels(sample_prices)
        assert len(labels) == len(sample_prices)
        assert isinstance(labels, np.ndarray)

    def test_bottom_statistics(self, sample_prices, sample_labels):
        """Test bottom statistics."""
        labeler = BottomLabeler()
        stats = labeler.get_bottom_statistics(sample_prices, sample_labels)
        assert 'total_bottoms' in stats
        assert 'mean_days_to_bottom' in stats
        assert 'median_days_to_bottom' in stats
        assert stats['total_bottoms'] >= 0

    def test_statistics_with_no_bottoms(self, sample_prices):
        """Test statistics with no bottoms."""
        labeler = BottomLabeler()
        labels = np.full(len(sample_prices), np.nan)
        stats = labeler.get_bottom_statistics(sample_prices, labels)
        assert stats['total_bottoms'] == 0


class TestMLPBottomModel:
    """Test MLP bottom model."""

    def test_initialization(self):
        """Test MLP initialization."""
        model = MLPBottomModel()
        assert model.name == "MLP_Bottom"
        assert not model.is_trained

    def test_training(self, sample_features, sample_labels):
        """Test MLP training."""
        model = MLPBottomModel()
        valid_idx = ~np.isnan(sample_labels)
        X_train = sample_features[valid_idx][:50]
        y_train = sample_labels[valid_idx][:50]

        if len(X_train) > 0:
            metrics = model.train(X_train, y_train)
            assert model.is_trained
            assert 'train_r2' in metrics

    def test_prediction(self, sample_features, sample_labels):
        """Test MLP prediction."""
        model = MLPBottomModel()
        valid_idx = ~np.isnan(sample_labels)
        if np.sum(valid_idx) < 60:
            pytest.skip("Not enough valid labels for test")

        X_train = sample_features[valid_idx][:50]
        y_train = sample_labels[valid_idx][:50]

        if len(X_train) > 0:
            model.train(X_train, y_train)
            X_test = sample_features[valid_idx][50:60]
            if len(X_test) > 0:
                predictions = model.predict(X_test)
                assert len(predictions) == len(X_test)
                assert np.all(np.isfinite(predictions))

    def test_evaluation(self, sample_features, sample_labels):
        """Test MLP evaluation."""
        model = MLPBottomModel()
        valid_idx = ~np.isnan(sample_labels)
        X_train = sample_features[valid_idx][:50]
        y_train = sample_labels[valid_idx][:50]
        X_test = sample_features[valid_idx][50:60]
        y_test = sample_labels[valid_idx][50:60]

        if len(X_train) > 0 and len(X_test) > 0:
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            assert 'r2' in metrics
            assert 'mae' in metrics
            assert 'accuracy' in metrics

    def test_nan_handling(self, sample_features, sample_labels):
        """Test NaN handling in training."""
        model = MLPBottomModel()
        X_train = sample_features[:50].copy()
        X_train[0, 0] = np.nan
        y_train = sample_labels[:50]
        valid_idx = ~np.isnan(y_train)
        y_train = y_train[valid_idx]
        X_train = X_train[valid_idx]

        if len(X_train) > 0:
            metrics = model.train(X_train, y_train)
            assert model.is_trained


class TestLSTMBottomModel:
    """Test LSTM bottom model."""

    def test_initialization(self):
        """Test LSTM initialization."""
        model = LSTMBottomModel(sequence_length=30)
        assert model.name == "LSTM_Bottom"
        assert model.sequence_length == 30
        assert not model.is_trained

    def test_sequence_creation(self):
        """Test sequence creation."""
        model = LSTMBottomModel(sequence_length=10)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        X_seq, y_seq = model._create_sequences(X, y)
        assert len(X_seq) == 40  # 50 - 10
        assert X_seq.shape == (40, 50)  # Flattened: 10 * 5 = 50

    def test_training(self, sample_features, sample_labels):
        """Test LSTM training."""
        model = LSTMBottomModel(sequence_length=10)
        valid_idx = ~np.isnan(sample_labels)
        X_train = sample_features[valid_idx][:80]
        y_train = sample_labels[valid_idx][:80]

        if len(X_train) > 20:
            metrics = model.train(X_train, y_train)
            assert model.is_trained
            assert 'train_r2' in metrics

    def test_prediction(self, sample_features, sample_labels):
        """Test LSTM prediction."""
        model = LSTMBottomModel(sequence_length=10)
        valid_idx = ~np.isnan(sample_labels)
        X_train = sample_features[valid_idx][:80]
        y_train = sample_labels[valid_idx][:80]

        if len(X_train) > 20:
            model.train(X_train, y_train)
            X_test = sample_features[valid_idx][80:100]
            predictions = model.predict(X_test)
            assert len(predictions) <= len(X_test)

    def test_evaluation(self, sample_features, sample_labels):
        """Test LSTM evaluation."""
        model = LSTMBottomModel(sequence_length=10)
        valid_idx = ~np.isnan(sample_labels)
        X_train = sample_features[valid_idx][:80]
        y_train = sample_labels[valid_idx][:80]
        X_test = sample_features[valid_idx][80:100]
        y_test = sample_labels[valid_idx][80:100]

        if len(X_train) > 20 and len(X_test) > 10:
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            assert 'r2' in metrics
            assert 'mae' in metrics
            assert 'accuracy' in metrics

    def test_nan_handling(self, sample_features, sample_labels):
        """Test NaN handling in training."""
        model = LSTMBottomModel(sequence_length=10)
        X_train = sample_features[:80].copy()
        X_train[0, 0] = np.nan
        y_train = sample_labels[:80]
        valid_idx = ~np.isnan(y_train)
        y_train = y_train[valid_idx]
        X_train = X_train[valid_idx]

        if len(X_train) > 20:
            metrics = model.train(X_train, y_train)
            assert model.is_trained


class TestBottomModelComparison:
    """Test comparing bottom models."""

    def test_both_models_trainable(self, sample_features, sample_labels):
        """Test that both models can be trained."""
        valid_idx = ~np.isnan(sample_labels)
        X_train = sample_features[valid_idx][:80]
        y_train = sample_labels[valid_idx][:80]

        if len(X_train) > 20:
            mlp = MLPBottomModel()
            lstm = LSTMBottomModel(sequence_length=10)

            mlp_metrics = mlp.train(X_train, y_train)
            lstm_metrics = lstm.train(X_train, y_train)

            assert mlp.is_trained
            assert lstm.is_trained
            assert 'train_r2' in mlp_metrics
            assert 'train_r2' in lstm_metrics

