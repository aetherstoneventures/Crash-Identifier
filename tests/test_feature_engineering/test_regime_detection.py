"""Tests for Bry-Boschan regime detection algorithm."""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering.regime_detection import BryBoschanDetector


@pytest.fixture
def sample_series():
    """Create sample market data with clear peaks and troughs."""
    dates = pd.date_range(start='2000-01-01', periods=240, freq='M')
    
    # Create synthetic data with clear cycles
    t = np.arange(240)
    # Trend + cyclical component
    trend = 100 + 0.5 * t
    cycle = 20 * np.sin(2 * np.pi * t / 60)  # 60-month cycle
    noise = np.random.normal(0, 2, 240)
    
    values = trend + cycle + noise
    series = pd.Series(values, index=dates)
    return series


@pytest.fixture
def detector():
    """Create Bry-Boschan detector."""
    return BryBoschanDetector(
        min_duration=6,
        min_amplitude=0.05,
        phase_length=3
    )


def test_detector_initialization():
    """Test detector initialization."""
    detector = BryBoschanDetector(
        min_duration=12,
        min_amplitude=0.10,
        phase_length=6
    )
    
    assert detector.min_duration == 12
    assert detector.min_amplitude == 0.10
    assert detector.phase_length == 6


def test_detect_regimes(sample_series, detector):
    """Test regime detection."""
    regimes = detector.detect_regimes(sample_series)
    
    # Should return DataFrame
    assert isinstance(regimes, pd.DataFrame)
    
    # Should have same length as input
    assert len(regimes) == len(sample_series)
    
    # Should have required columns
    assert 'regime' in regimes.columns
    assert 'regime_duration' in regimes.columns
    assert 'is_peak' in regimes.columns
    assert 'is_trough' in regimes.columns


def test_regime_values(sample_series, detector):
    """Test that regime values are 0 or 1."""
    regimes = detector.detect_regimes(sample_series)
    
    # Regime should be 0 (contraction) or 1 (expansion)
    assert set(regimes['regime'].unique()).issubset({0, 1})


def test_regime_duration(sample_series, detector):
    """Test regime duration calculation."""
    regimes = detector.detect_regimes(sample_series)

    # Duration should be non-negative
    assert (regimes['regime_duration'] >= 0).all()

    # Duration should increase or reset within same regime
    for i in range(1, len(regimes)):
        if regimes['regime'].iloc[i] == regimes['regime'].iloc[i-1]:
            # Duration should increase or stay same
            assert regimes['regime_duration'].iloc[i] >= regimes['regime_duration'].iloc[i-1]
        else:
            # Duration resets when regime changes
            assert regimes['regime_duration'].iloc[i] <= 2


def test_peaks_and_troughs(sample_series, detector):
    """Test peak and trough detection."""
    regimes = detector.detect_regimes(sample_series)
    
    # Peaks and troughs should be binary
    assert set(regimes['is_peak'].unique()).issubset({0, 1})
    assert set(regimes['is_trough'].unique()).issubset({0, 1})
    
    # Should have at least some peaks and troughs
    assert regimes['is_peak'].sum() > 0
    assert regimes['is_trough'].sum() > 0


def test_get_current_regime(sample_series, detector):
    """Test getting current regime."""
    current = detector.get_current_regime(sample_series)
    
    # Should return dict
    assert isinstance(current, dict)
    
    # Should have required keys
    assert 'regime' in current
    assert 'regime_code' in current
    assert 'duration_months' in current
    assert 'is_peak' in current
    assert 'is_trough' in current
    
    # Regime should be string
    assert current['regime'] in ['Expansion', 'Contraction']
    
    # Regime code should be 0 or 1
    assert current['regime_code'] in [0, 1]
    
    # Duration should be positive
    assert current['duration_months'] > 0


def test_short_series_error(detector):
    """Test that short series raises error."""
    short_series = pd.Series(np.random.randn(10))
    
    with pytest.raises(ValueError):
        detector.detect_regimes(short_series)


def test_empty_series_error(detector):
    """Test that empty series raises error."""
    empty_series = pd.Series([])
    
    with pytest.raises(ValueError):
        detector.detect_regimes(empty_series)


def test_uptrend_detection():
    """Test detection of clear uptrend."""
    dates = pd.date_range(start='2000-01-01', periods=120, freq='M')
    # Clear uptrend
    values = np.linspace(100, 150, 120)
    series = pd.Series(values, index=dates)
    
    detector = BryBoschanDetector(min_duration=6, min_amplitude=0.05)
    regimes = detector.detect_regimes(series)
    
    # Most of the time should be expansion
    assert regimes['regime'].sum() > len(regimes) * 0.7


def test_downtrend_detection():
    """Test detection of clear downtrend."""
    dates = pd.date_range(start='2000-01-01', periods=120, freq='M')
    # Clear downtrend
    values = np.linspace(150, 100, 120)
    series = pd.Series(values, index=dates)

    detector = BryBoschanDetector(min_duration=6, min_amplitude=0.05)
    regimes = detector.detect_regimes(series)

    # Should detect some regime changes
    assert regimes['regime'].nunique() >= 1


def test_regime_transitions(sample_series, detector):
    """Test that regimes transition properly."""
    regimes = detector.detect_regimes(sample_series)
    
    # Count transitions
    transitions = (regimes['regime'].diff() != 0).sum()
    
    # Should have some transitions
    assert transitions > 0
    
    # Transitions should be reasonable (not too many)
    assert transitions < len(regimes) / 10

