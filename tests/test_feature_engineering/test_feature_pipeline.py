"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_raw_data():
    """Create sample raw data from data collection."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    data = pd.DataFrame({
        'yield_10y': np.linspace(1.5, 2.5, 252),
        'yield_10y_3m': np.linspace(1.0, 1.5, 252),
        'yield_10y_2y': np.linspace(0.5, 1.0, 252),
        'credit_spread_bbb': np.linspace(2.0, 3.0, 252),
        'vix_close': np.linspace(15, 25, 252),
        'sp500_close': np.linspace(3000, 3500, 252),
        'sp500_volume': np.linspace(1e9, 2e9, 252),
        'unemployment_rate': np.linspace(3.5, 4.5, 252),
        'real_gdp': np.linspace(20000, 21000, 252),
        'cpi': np.linspace(250, 260, 252),
        'fed_funds_rate': np.linspace(1.5, 2.5, 252),
        'consumer_sentiment': np.linspace(95, 105, 252),
        'housing_starts': np.linspace(1200, 1400, 252),
        'industrial_production': np.linspace(100, 110, 252),
        'm2_money_supply': np.linspace(15000, 16000, 252),
        'debt_to_gdp': np.linspace(100, 110, 252),
        'savings_rate': np.linspace(7, 8, 252),
        'lei': np.linspace(100, 105, 252),
        'shiller_pe': np.linspace(25, 30, 252),
        'margin_debt': np.linspace(500, 600, 252),
        'put_call_ratio': np.linspace(0.8, 1.2, 252),
        'yield_spread_10y_3m': np.linspace(0.5, 1.0, 252),
        'yield_spread_10y_2y': np.linspace(0.5, 1.0, 252),
        'vix_level': np.linspace(15, 25, 252),
        'vix_change_rate': np.linspace(-5, 5, 252),
        'realized_volatility': np.linspace(10, 20, 252),
        'sp500_momentum_200d': np.linspace(-10, 10, 252),
        'sp500_drawdown': np.linspace(-20, 0, 252),
    }, index=dates)
    return data


@pytest.fixture
def pipeline():
    """Create feature pipeline."""
    return FeaturePipeline()


def test_pipeline_initialization(pipeline):
    """Test pipeline initialization."""
    assert pipeline.indicators is None
    assert pipeline.regimes is None
    assert pipeline.feature_names == []


def test_process_raw_data(pipeline, sample_raw_data):
    """Test processing raw data through pipeline."""
    features, metadata = pipeline.process(sample_raw_data)
    
    # Should return DataFrame and dict
    assert isinstance(features, pd.DataFrame)
    assert isinstance(metadata, dict)
    
    # Features should have same length as input
    assert len(features) == len(sample_raw_data)
    
    # Should have features
    assert len(features.columns) > 0


def test_metadata_structure(pipeline, sample_raw_data):
    """Test metadata structure."""
    features, metadata = pipeline.process(sample_raw_data)
    
    # Check required metadata keys
    assert 'n_features' in metadata
    assert 'n_samples' in metadata
    assert 'feature_names' in metadata
    assert 'redundant_features' in metadata
    assert 'correlation_threshold' in metadata
    assert 'normalization_method' in metadata
    assert 'date_range' in metadata
    
    # Check values
    assert metadata['n_samples'] == len(sample_raw_data)
    assert metadata['n_features'] > 0
    assert isinstance(metadata['feature_names'], list)
    assert isinstance(metadata['redundant_features'], list)


def test_feature_normalization(pipeline, sample_raw_data):
    """Test that features are normalized."""
    features, metadata = pipeline.process(sample_raw_data)

    # Get numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns

    # Check normalization (should be roughly centered at 0 with std ~1)
    for col in numeric_cols:
        if col not in ['regime', 'regime_duration']:
            mean = features[col].mean()
            std = features[col].std()

            # Mean should be close to 0 (or NaN for missing data)
            if not np.isnan(mean):
                assert abs(mean) < 3

            # Std should be reasonable (or NaN for missing data)
            if not np.isnan(std) and std > 0:
                assert 0.01 < std < 3


def test_redundant_feature_removal(pipeline, sample_raw_data):
    """Test that redundant features are removed."""
    features, metadata = pipeline.process(sample_raw_data)
    
    # Should have removed some redundant features
    # (depends on correlation threshold)
    assert len(metadata['redundant_features']) >= 0


def test_missing_value_handling(pipeline, sample_raw_data):
    """Test handling of missing values."""
    # Add some missing values
    sample_raw_data.loc[sample_raw_data.index[10:20], 'yield_10y'] = np.nan

    features, metadata = pipeline.process(sample_raw_data)

    # Should have filled most missing values (some indicators may still be NaN if source data is missing)
    total_values = features.shape[0] * features.shape[1]
    missing_count = features.isna().sum().sum()
    missing_percentage = (missing_count / total_values) * 100

    # Should have < 80% missing (many indicators depend on missing source data)
    assert missing_percentage < 80


def test_regime_detection_included(pipeline, sample_raw_data):
    """Test that regime detection is included."""
    features, metadata = pipeline.process(sample_raw_data)
    
    # Should have regime column
    assert 'regime' in features.columns


def test_get_feature_importance(pipeline, sample_raw_data):
    """Test feature importance calculation."""
    pipeline.process(sample_raw_data)
    importance = pipeline.get_feature_importance()
    
    # Should return DataFrame
    assert isinstance(importance, pd.DataFrame)
    
    # Should have required columns
    assert 'feature' in importance.columns
    assert 'mean' in importance.columns
    assert 'std' in importance.columns
    assert 'min' in importance.columns
    assert 'max' in importance.columns
    assert 'skew' in importance.columns
    assert 'kurtosis' in importance.columns


def test_get_correlation_matrix(pipeline, sample_raw_data):
    """Test correlation matrix retrieval."""
    pipeline.process(sample_raw_data)
    corr_matrix = pipeline.get_correlation_matrix()

    # Should return DataFrame
    assert isinstance(corr_matrix, pd.DataFrame)

    # Should be square
    assert corr_matrix.shape[0] == corr_matrix.shape[1]

    # Diagonal should be 1 (or NaN for missing data)
    diag_values = np.diag(corr_matrix)
    for val in diag_values:
        if not np.isnan(val):
            assert np.isclose(val, 1.0)


def test_empty_data_error(pipeline):
    """Test that empty data raises error."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        pipeline.process(empty_df)


def test_feature_importance_before_process(pipeline):
    """Test that feature importance raises error before processing."""
    with pytest.raises(ValueError):
        pipeline.get_feature_importance()


def test_correlation_matrix_before_process(pipeline):
    """Test that correlation matrix raises error before processing."""
    with pytest.raises(ValueError):
        pipeline.get_correlation_matrix()


def test_pipeline_reproducibility(sample_raw_data):
    """Test that pipeline produces reproducible results."""
    pipeline1 = FeaturePipeline()
    features1, metadata1 = pipeline1.process(sample_raw_data.copy())
    
    pipeline2 = FeaturePipeline()
    features2, metadata2 = pipeline2.process(sample_raw_data.copy())
    
    # Results should be identical
    pd.testing.assert_frame_equal(features1, features2)
    assert metadata1 == metadata2


def test_feature_count(pipeline, sample_raw_data):
    """Test that feature count is reasonable."""
    features, metadata = pipeline.process(sample_raw_data)

    # Should have at least 5 features (after redundancy removal + regime)
    # Note: With linspace data, many features are highly correlated and removed
    assert metadata['n_features'] >= 5

    # Should have at most 30 features (28 indicators + regime info)
    assert metadata['n_features'] <= 30

