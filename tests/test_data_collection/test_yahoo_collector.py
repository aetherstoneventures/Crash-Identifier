"""Tests for Yahoo Finance data collector."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data_collection.yahoo_collector import YahooCollector


class TestYahooCollector:
    """Test cases for YahooCollector."""

    @pytest.fixture
    def collector(self):
        """Create a YahooCollector instance for testing."""
        return YahooCollector()

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': np.linspace(100, 110, 10),
            'High': np.linspace(102, 112, 10),
            'Low': np.linspace(98, 108, 10),
            'Close': np.linspace(101, 111, 10),
            'Volume': np.full(10, 1000000),
            'Adj Close': np.linspace(101, 111, 10)
        }, index=dates)

    def test_symbols_dict(self, collector):
        """Test SYMBOLS dict contains expected symbols."""
        assert 'sp500' in collector.SYMBOLS
        assert 'vix' in collector.SYMBOLS
        assert collector.SYMBOLS['sp500'] == '^GSPC'
        assert collector.SYMBOLS['vix'] == '^VIX'

    def test_fetch_price_data_invalid_symbol(self, collector):
        """Test fetch_price_data with invalid symbol."""
        with pytest.raises(ValueError, match="Unknown symbol"):
            collector.fetch_price_data("invalid_symbol")

    @patch('yfinance.download')
    def test_fetch_price_data_success(self, mock_download, collector, sample_price_data):
        """Test successful price data fetch."""
        mock_download.return_value = sample_price_data

        result = collector.fetch_price_data('sp500', '2023-01-01', '2023-01-10')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert 'Close' in result.columns

    @patch('yfinance.download')
    def test_fetch_price_data_sp500(self, mock_download, collector, sample_price_data):
        """Test fetching S&P 500 data."""
        mock_download.return_value = sample_price_data

        result = collector.fetch_price_data('sp500')

        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[0][0] == '^GSPC'

    @patch('yfinance.download')
    def test_fetch_price_data_vix(self, mock_download, collector, sample_price_data):
        """Test fetching VIX data."""
        mock_download.return_value = sample_price_data

        result = collector.fetch_price_data('vix')

        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[0][0] == '^VIX'

    def test_calculate_returns(self, collector, sample_price_data):
        """Test log returns calculation."""
        prices = sample_price_data['Close']
        returns = collector.calculate_returns(prices)

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1  # One less due to shift
        assert not returns.isnull().any()

    def test_calculate_returns_values(self, collector):
        """Test returns calculation with known values."""
        prices = pd.Series([100, 110, 121])
        returns = collector.calculate_returns(prices)

        # Log return from 100 to 110: ln(110/100) ≈ 0.0953
        assert len(returns) == 2
        assert returns.iloc[0] > 0  # Positive return
        assert returns.iloc[1] > 0  # Positive return

    def test_calculate_volatility(self, collector, sample_price_data):
        """Test volatility calculation."""
        prices = sample_price_data['Close']
        returns = collector.calculate_returns(prices)
        volatility = collector.calculate_volatility(returns, window=5)

        assert isinstance(volatility, pd.Series)
        assert len(volatility) <= len(returns)  # Window size
        # Check non-NaN values are non-negative
        assert (volatility.dropna() >= 0).all()  # All non-negative

    def test_calculate_volatility_window(self, collector):
        """Test volatility with different window sizes."""
        np.random.seed(42)
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        returns = collector.calculate_returns(prices)

        vol_20 = collector.calculate_volatility(returns, window=20)
        vol_10 = collector.calculate_volatility(returns, window=10)

        assert len(vol_20) <= len(vol_10)

    def test_calculate_drawdown(self, collector):
        """Test drawdown calculation."""
        prices = pd.Series([100, 110, 105, 95, 100, 120])
        drawdown = collector.calculate_drawdown(prices)

        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(prices)
        assert drawdown.iloc[0] == 0  # First value is 0
        assert (drawdown <= 0).all()  # All non-positive

    def test_calculate_drawdown_values(self, collector):
        """Test drawdown with known values."""
        prices = pd.Series([100, 110, 90, 95])
        drawdown = collector.calculate_drawdown(prices)

        assert drawdown.iloc[0] == 0  # 100 vs 100
        assert drawdown.iloc[1] == 0  # 110 vs 110 (new high)
        assert drawdown.iloc[2] < 0  # 90 vs 110 (drawdown)
        assert drawdown.iloc[3] < 0  # 95 vs 110 (drawdown)

    @patch('yfinance.download')
    def test_fetch_sp500_and_vix(self, mock_download, collector, sample_price_data):
        """Test fetching both S&P 500 and VIX."""
        mock_download.return_value = sample_price_data

        result = collector.fetch_sp500_and_vix('2023-01-01', '2023-01-10')

        assert isinstance(result, dict)
        assert 'sp500' in result
        assert 'vix' in result
        assert isinstance(result['sp500'], pd.DataFrame)
        assert isinstance(result['vix'], pd.DataFrame)

    @patch('yfinance.download')
    def test_fetch_sp500_and_vix_calls(self, mock_download, collector, sample_price_data):
        """Test that fetch_sp500_and_vix makes correct calls."""
        mock_download.return_value = sample_price_data

        collector.fetch_sp500_and_vix()

        assert mock_download.call_count == 2


class TestYahooCollectorIntegration:
    """Integration tests for YahooCollector (requires internet)."""

    @pytest.mark.skip(reason="Requires internet connection")
    def test_real_api_fetch(self):
        """Test with real Yahoo Finance API (skipped by default)."""
        collector = YahooCollector()
        result = collector.fetch_price_data('sp500', '2023-01-01', '2023-01-31')

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Close' in result.columns

