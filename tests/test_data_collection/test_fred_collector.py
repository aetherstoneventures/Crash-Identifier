"""Tests for FRED data collector."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.data_collection.fred_collector import FREDCollector


class TestFREDCollector:
    """Test cases for FREDCollector."""

    @pytest.fixture
    def collector(self):
        """Create a FREDCollector instance for testing."""
        return FREDCollector(api_key="test_key_12345")

    def test_init_with_valid_key(self):
        """Test initialization with valid API key."""
        collector = FREDCollector(api_key="test_key")
        assert collector.api_key == "test_key"
        assert collector.request_count == 0

    def test_init_with_empty_key(self):
        """Test initialization with empty API key raises error."""
        with pytest.raises(ValueError, match="FRED API key is required"):
            FREDCollector(api_key="")

    def test_indicators_dict_has_16_items(self, collector):
        """Test that INDICATORS dict contains all 16 indicators."""
        assert len(collector.INDICATORS) == 16
        assert 'yield_10y_3m' in collector.INDICATORS
        assert 'unemployment_rate' in collector.INDICATORS

    def test_fetch_indicator_invalid_name(self, collector):
        """Test fetch_indicator with invalid indicator name."""
        with pytest.raises(ValueError, match="Unknown indicator"):
            collector.fetch_indicator("invalid_indicator")

    @patch('fredapi.Fred.get_series')
    def test_fetch_indicator_success(self, mock_get_series, collector):
        """Test successful indicator fetch."""
        # Mock the FRED API response
        mock_series = pd.Series(
            [1.5, 1.6, 1.7],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        mock_get_series.return_value = mock_series

        result = collector.fetch_indicator('yield_10y_3m', '2023-01-01', '2023-01-03')

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        mock_get_series.assert_called_once()

    @patch('fredapi.Fred.get_series')
    def test_fetch_indicator_with_retry(self, mock_get_series, collector):
        """Test fetch_indicator with retry logic."""
        # First call fails, second succeeds
        mock_series = pd.Series(
            [1.5, 1.6],
            index=pd.date_range('2023-01-01', periods=2, freq='D')
        )
        mock_get_series.side_effect = [Exception("API Error"), mock_series]

        result = collector.fetch_indicator('yield_10y_3m')

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert mock_get_series.call_count == 2

    @patch('fredapi.Fred.get_series')
    def test_fetch_all_indicators(self, mock_get_series, collector):
        """Test fetching all indicators."""
        mock_series = pd.Series(
            [1.5, 1.6, 1.7],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        mock_get_series.return_value = mock_series

        result = collector.fetch_all_indicators('2023-01-01', '2023-01-03')

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3  # 3 dates
        assert result.shape[1] == 16  # 16 indicators

    @patch('fredapi.Fred.get_series')
    def test_fetch_all_indicators_with_failures(self, mock_get_series, collector):
        """Test fetch_all_indicators handles some failures gracefully."""
        mock_series = pd.Series(
            [1.5, 1.6],
            index=pd.date_range('2023-01-01', periods=2, freq='D')
        )
        # Alternate between success and failure
        mock_get_series.side_effect = [
            mock_series, Exception("API Error"), mock_series,
            Exception("API Error"), mock_series, Exception("API Error"),
            mock_series, Exception("API Error"), mock_series,
            Exception("API Error"), mock_series, Exception("API Error"),
            mock_series, Exception("API Error"), mock_series,
            Exception("API Error")
        ]

        result = collector.fetch_all_indicators()

        # Should have some columns despite failures
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] > 0

    def test_rate_limit_counter(self, collector):
        """Test rate limit counter increments."""
        initial_count = collector.request_count
        collector._rate_limit()
        assert collector.request_count == initial_count + 1


class TestFREDCollectorIntegration:
    """Integration tests for FREDCollector (requires API key)."""

    @pytest.mark.skip(reason="Requires valid FRED API key")
    def test_real_api_fetch(self):
        """Test with real FRED API (skipped by default)."""
        import os
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set")

        collector = FREDCollector(api_key)
        result = collector.fetch_indicator('unemployment_rate', '2023-01-01', '2023-01-31')

        assert isinstance(result, pd.Series)
        assert len(result) > 0

