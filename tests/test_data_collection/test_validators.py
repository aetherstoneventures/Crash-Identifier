"""Tests for data validators."""

import pytest
import pandas as pd
import numpy as np

from src.utils.validators import DataValidator


class TestDataValidator:
    """Test cases for DataValidator."""

    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe with various data quality issues."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'yield_10y_3m': np.random.uniform(-2, 2, 100),
            'unemployment_rate': np.random.uniform(3, 8, 100),
            'vix_close': np.random.uniform(10, 30, 100),
            'sp500_close': np.random.uniform(4000, 4500, 100),
            'consumer_sentiment': np.random.uniform(80, 120, 100)
        }, index=dates)

    def test_init_with_default_ranges(self, validator):
        """Test initialization with default expected ranges."""
        assert validator.expected_ranges is not None
        assert 'yield_10y_3m' in validator.expected_ranges

    def test_init_with_custom_ranges(self):
        """Test initialization with custom ranges."""
        custom_ranges = {'test_col': (0, 100)}
        validator = DataValidator(expected_ranges=custom_ranges)
        assert validator.expected_ranges == custom_ranges

    def test_validate_missing_values_no_missing(self, validator, sample_dataframe):
        """Test validation with no missing values."""
        result = validator.validate_missing_values(sample_dataframe)

        assert result['total_missing'] == 0
        assert result['missing_percentage'] == 0
        assert len(result['critical_missing']) == 0

    def test_validate_missing_values_with_missing(self, validator, sample_dataframe):
        """Test validation with missing values."""
        # Add missing values
        sample_dataframe.iloc[0:11, sample_dataframe.columns.get_loc('yield_10y_3m')] = np.nan

        result = validator.validate_missing_values(sample_dataframe)

        assert result['total_missing'] > 0
        assert result['missing_percentage'] > 0
        assert 'yield_10y_3m' in result['missing_by_column']

    def test_validate_missing_values_critical(self, validator, sample_dataframe):
        """Test detection of critical missing values."""
        # Add > 5% missing values
        sample_dataframe.iloc[0:11, sample_dataframe.columns.get_loc('unemployment_rate')] = np.nan

        result = validator.validate_missing_values(sample_dataframe)

        assert 'unemployment_rate' in result['critical_missing']

    def test_validate_value_ranges_valid(self, validator, sample_dataframe):
        """Test validation with values in valid ranges."""
        result = validator.validate_value_ranges(sample_dataframe)

        assert result['valid'] is True
        assert len(result['outliers']) == 0

    def test_validate_value_ranges_outliers(self, validator, sample_dataframe):
        """Test detection of outliers."""
        # Add outliers
        sample_dataframe.loc[0, 'unemployment_rate'] = 50  # Above max

        result = validator.validate_value_ranges(sample_dataframe)

        assert result['valid'] is False
        assert len(result['outliers']) > 0

    def test_fill_missing_values_forward_fill(self, validator):
        """Test forward fill for daily indicators."""
        df = pd.DataFrame({
            'yield_10y_3m': [1.0, np.nan, np.nan, 1.5, 1.6],
            'vix_close': [20, np.nan, 21, np.nan, 22]
        })

        filled = validator.fill_missing_values(df)

        assert filled['yield_10y_3m'].iloc[1] == 1.0
        assert filled['yield_10y_3m'].iloc[2] == 1.0
        assert filled['vix_close'].iloc[1] == 20

    def test_fill_missing_values_interpolate(self, validator):
        """Test interpolation for continuous indicators."""
        df = pd.DataFrame({
            'real_gdp': [100, np.nan, np.nan, 110],
            'cpi': [200, np.nan, 210, np.nan]
        })

        filled = validator.fill_missing_values(df)

        # Interpolated values should be between neighbors
        assert 100 < filled['real_gdp'].iloc[1] < 110
        assert 100 < filled['real_gdp'].iloc[2] < 110

    def test_calculate_data_quality_score_perfect(self, validator, sample_dataframe):
        """Test quality score with perfect data."""
        score = validator.calculate_data_quality_score(sample_dataframe)

        assert 0 <= score <= 1
        assert score > 0.9  # Should be high for clean data

    def test_calculate_data_quality_score_poor(self, validator):
        """Test quality score with poor data."""
        df = pd.DataFrame({
            'col1': [np.nan] * 100,
            'col2': [np.nan] * 100
        })

        score = validator.calculate_data_quality_score(df)

        assert score < 0.5  # Should be low for mostly missing data

    def test_validate_dataframe_comprehensive(self, validator, sample_dataframe):
        """Test comprehensive dataframe validation."""
        result = validator.validate_dataframe(sample_dataframe)

        assert 'missing_values' in result
        assert 'value_ranges' in result
        assert 'quality_score' in result
        assert 'shape' in result
        assert 'date_range' in result

        assert result['shape'] == sample_dataframe.shape
        assert 0 <= result['quality_score'] <= 1

    def test_validate_dataframe_with_issues(self, validator):
        """Test comprehensive validation with data issues."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [100, 200, 300, 400, 500]
        })

        result = validator.validate_dataframe(df)

        assert result['missing_values']['total_missing'] > 0
        assert result['quality_score'] < 1.0

    def test_fill_missing_values_preserves_shape(self, validator):
        """Test that fill_missing_values preserves dataframe shape."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3],
            'col2': [np.nan, 5, 6]
        })

        filled = validator.fill_missing_values(df)

        assert filled.shape == df.shape

    def test_fill_missing_values_no_nans_remain(self, validator):
        """Test that fill_missing_values removes most NaNs."""
        df = pd.DataFrame({
            'yield_10y_3m': [1.0, np.nan, np.nan, 1.5],
            'unemployment_rate': [5.0, np.nan, 5.5, np.nan]
        })

        filled = validator.fill_missing_values(df)

        # Should have fewer NaNs
        assert filled.isnull().sum().sum() < df.isnull().sum().sum()

