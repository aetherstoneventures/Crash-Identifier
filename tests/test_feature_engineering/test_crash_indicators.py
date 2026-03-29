"""Tests for crash indicators calculation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.crash_indicators import CrashIndicators


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    data = pd.DataFrame({
        'yield_10y': np.linspace(1.5, 2.5, 252),
        'yield_10y_3m': np.linspace(1.0, 1.5, 252),
        'yield_10y_2y': np.linspace(0.5, 1.0, 252),
        'credit_spread_bbb': np.linspace(2.0, 3.0, 252),
        'vix_close': np.linspace(15, 25, 252),
        'sp500_close': np.linspace(3000, 3500, 252),
        'unemployment_rate': np.linspace(3.5, 4.5, 252),
        'real_gdp': np.linspace(20000, 21000, 252),
        'consumer_sentiment': np.linspace(95, 105, 252),
        'housing_starts': np.linspace(1200, 1400, 252),
        'industrial_production': np.linspace(100, 110, 252),
        'm2_money_supply': np.linspace(15000, 16000, 252),
        'debt_to_gdp': np.linspace(100, 110, 252),
    }, index=dates)
    return data


def test_yield_spread_10y_3m(sample_data):
    """Test 10Y-3M yield spread calculation."""
    spread = CrashIndicators.yield_spread_10y_3m(sample_data)
    assert len(spread) == len(sample_data)
    assert spread.iloc[0] > 0  # 10Y should be higher than 3M


def test_yield_spread_10y_2y(sample_data):
    """Test 10Y-2Y yield spread calculation."""
    spread = CrashIndicators.yield_spread_10y_2y(sample_data)
    assert len(spread) == len(sample_data)
    assert spread.iloc[0] > 0


def test_vix_level(sample_data):
    """Test VIX level extraction."""
    vix = CrashIndicators.vix_level(sample_data)
    assert len(vix) == len(sample_data)
    assert vix.min() >= 15
    assert vix.max() <= 25


def test_vix_change_rate(sample_data):
    """Test VIX change rate calculation."""
    vix_change = CrashIndicators.vix_change_rate(sample_data, window=20)
    assert len(vix_change) == len(sample_data)
    # First 20 values should be NaN
    assert vix_change.iloc[:20].isna().all()


def test_realized_volatility(sample_data):
    """Test realized volatility calculation."""
    vol = CrashIndicators.realized_volatility(sample_data, window=20)
    assert len(vol) == len(sample_data)
    assert vol.min() >= 0  # Volatility should be non-negative


def test_sp500_momentum_200d(sample_data):
    """Test S&P 500 momentum calculation."""
    momentum = CrashIndicators.sp500_momentum_200d(sample_data)
    assert len(momentum) == len(sample_data)
    # Momentum should be positive for uptrend
    assert momentum.iloc[-1] > 0


def test_sp500_drawdown(sample_data):
    """Test drawdown calculation."""
    drawdown = CrashIndicators.sp500_drawdown(sample_data)
    assert len(drawdown) == len(sample_data)
    # Drawdown should be <= 0
    assert drawdown.max() <= 0


def test_unemployment_rate(sample_data):
    """Test unemployment rate extraction."""
    unemp = CrashIndicators.unemployment_rate(sample_data)
    assert len(unemp) == len(sample_data)
    assert unemp.min() >= 3.5
    assert unemp.max() <= 4.5


def test_sahm_rule(sample_data):
    """Test Sahm Rule calculation."""
    sahm = CrashIndicators.sahm_rule(sample_data)
    assert len(sahm) == len(sample_data)
    # Sahm rule should be non-negative
    assert sahm.min() >= 0


def test_gdp_growth(sample_data):
    """Test GDP growth calculation."""
    gdp_growth = CrashIndicators.gdp_growth(sample_data, periods=4)
    assert len(gdp_growth) == len(sample_data)
    # First 4 values should be NaN
    assert gdp_growth.iloc[:4].isna().all()


def test_calculate_all_indicators(sample_data):
    """Test calculation of all 28 indicators."""
    indicators = CrashIndicators.calculate_all_indicators(sample_data)
    
    # Should have 28 indicators
    assert len(indicators.columns) == 28
    
    # Should have same length as input
    assert len(indicators) == len(sample_data)
    
    # Check specific indicators exist
    expected_indicators = [
        'yield_spread_10y_3m',
        'yield_spread_10y_2y',
        'credit_spread_bbb',
        'vix_level',
        'vix_change_rate',
        'realized_volatility',
        'sp500_momentum_200d',
        'sp500_drawdown',
        'unemployment_rate',
        'sahm_rule',
        'gdp_growth',
        'industrial_production_growth',
        'housing_starts_growth',
    ]
    
    for indicator in expected_indicators:
        assert indicator in indicators.columns


def test_calculate_all_indicators_empty_data():
    """Test that empty data raises error."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        CrashIndicators.calculate_all_indicators(empty_df)


def test_missing_columns_handling(sample_data):
    """Test handling of missing columns."""
    # Remove a column
    sample_data_missing = sample_data.drop('vix_close', axis=1)
    
    # Should still calculate other indicators
    indicators = CrashIndicators.calculate_all_indicators(sample_data_missing)
    assert len(indicators) == len(sample_data_missing)
    
    # VIX-related indicators should be NaN
    assert indicators['vix_level'].isna().all()

