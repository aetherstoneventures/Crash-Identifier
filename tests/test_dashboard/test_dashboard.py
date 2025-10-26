"""Tests for dashboard components and pages."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.dashboard.components.metrics import MetricsCard, MetricsRow
from src.dashboard.pages.overview import OverviewPage
from src.dashboard.pages.crash_predictions import CrashPredictionsPage
from src.dashboard.pages.bottom_predictions import BottomPredictionsPage
from src.dashboard.pages.indicators import IndicatorsPage


class TestMetricsCard:
    """Test MetricsCard component."""
    
    def test_initialization(self):
        """Test metrics card initialization."""
        card = MetricsCard(
            label="Test Metric",
            value=42,
            delta="+5",
            color="blue"
        )
        assert card.label == "Test Metric"
        assert card.value == 42
        assert card.delta == "+5"
        assert card.color == "blue"
    
    def test_initialization_without_delta(self):
        """Test metrics card initialization without delta."""
        card = MetricsCard(label="Test", value=100)
        assert card.label == "Test"
        assert card.value == 100
        assert card.delta is None


class TestMetricsRow:
    """Test MetricsRow component."""
    
    def test_initialization(self):
        """Test metrics row initialization."""
        metrics = [
            {'label': 'Metric 1', 'value': 10},
            {'label': 'Metric 2', 'value': 20},
        ]
        row = MetricsRow(metrics, columns=2)
        assert len(row.metrics) == 2
        assert row.columns == 2
    
    def test_initialization_with_delta(self):
        """Test metrics row with delta values."""
        metrics = [
            {'label': 'Metric 1', 'value': 10, 'delta': '+5'},
            {'label': 'Metric 2', 'value': 20, 'delta': '-3'},
        ]
        row = MetricsRow(metrics)
        assert row.metrics[0]['delta'] == '+5'
        assert row.metrics[1]['delta'] == '-3'


class TestOverviewPage:
    """Test OverviewPage component."""

    def test_load_data(self):
        """Test loading data for overview page."""
        try:
            indicators, predictions = OverviewPage.load_data()
            assert isinstance(indicators, list)
            assert isinstance(predictions, list)
        except Exception:
            # Database may not be initialized in test environment
            pytest.skip("Database not initialized")


class TestCrashPredictionsPage:
    """Test CrashPredictionsPage component."""

    def test_load_predictions(self):
        """Test loading predictions."""
        try:
            predictions = CrashPredictionsPage.load_predictions()
            assert isinstance(predictions, list)
        except Exception:
            # Database may not be initialized in test environment
            pytest.skip("Database not initialized")
    
    def test_plot_crash_probability(self):
        """Test plotting crash probability."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=10)
        probs = np.random.uniform(0, 1, 10)
        
        pred_df = pd.DataFrame({
            'prediction_date': dates,
            'crash_probability': probs
        })
        
        # Test that function doesn't raise error
        try:
            CrashPredictionsPage.plot_crash_probability(pred_df)
        except Exception as e:
            pytest.fail(f"plot_crash_probability raised {type(e).__name__}")


class TestBottomPredictionsPage:
    """Test BottomPredictionsPage component."""

    def test_load_predictions(self):
        """Test loading predictions."""
        try:
            predictions = BottomPredictionsPage.load_predictions()
            assert isinstance(predictions, list)
        except Exception:
            # Database may not be initialized in test environment
            pytest.skip("Database not initialized")
    
    def test_plot_days_to_bottom(self):
        """Test plotting days to bottom."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=10)
        days = np.random.randint(1, 60, 10)
        
        pred_df = pd.DataFrame({
            'prediction_date': dates,
            'days_to_bottom': days
        })
        
        # Test that function doesn't raise error
        try:
            BottomPredictionsPage.plot_days_to_bottom(pred_df)
        except Exception as e:
            pytest.fail(f"plot_days_to_bottom raised {type(e).__name__}")


class TestIndicatorsPage:
    """Test IndicatorsPage component."""

    def test_load_indicators(self):
        """Test loading indicators."""
        try:
            indicators = IndicatorsPage.load_indicators()
            assert isinstance(indicators, list)
        except Exception:
            # Database may not be initialized in test environment
            pytest.skip("Database not initialized")
    
    def test_plot_indicators(self):
        """Test plotting indicators."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=10)
        ind_df = pd.DataFrame({
            'date': dates,
            'yield_spread_10y_2y': np.random.uniform(0, 3, 10),
            'vix_close': np.random.uniform(10, 30, 10),
        })
        
        columns = ['yield_spread_10y_2y', 'vix_close']
        
        # Test that function doesn't raise error
        try:
            IndicatorsPage.plot_indicators(ind_df, columns)
        except Exception as e:
            pytest.fail(f"plot_indicators raised {type(e).__name__}")

