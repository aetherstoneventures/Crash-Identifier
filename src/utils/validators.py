"""Data validation utilities for market indicators."""

import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

from src.utils.config import EXPECTED_RANGES, MAX_MISSING_PERCENTAGE

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality for market indicators."""

    def __init__(self, expected_ranges: Optional[Dict] = None):
        """
        Initialize validator.

        Args:
            expected_ranges: Dict of column names to (min, max) tuples
        """
        self.expected_ranges = expected_ranges or EXPECTED_RANGES
        self.logger = logging.getLogger(__name__)

    def validate_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing values in dataframe.

        Args:
            df: Input dataframe

        Returns:
            Dict with missing value statistics
        """
        total_missing = df.isnull().sum().sum()
        missing_by_column = (df.isnull().sum() / len(df) * 100).to_dict()
        critical_missing = [
            col for col, pct in missing_by_column.items()
            if pct > MAX_MISSING_PERCENTAGE
        ]

        return {
            'total_missing': total_missing,
            'missing_by_column': missing_by_column,
            'critical_missing': critical_missing,
            'total_cells': df.shape[0] * df.shape[1],
            'missing_percentage': (total_missing / (df.shape[0] * df.shape[1])) * 100
        }

    def validate_value_ranges(self, df: pd.DataFrame) -> Dict:
        """
        Check if values are within expected ranges.

        Args:
            df: Input dataframe

        Returns:
            Dict with validation results
        """
        outliers = {}
        all_valid = True

        for col in df.columns:
            if col in self.expected_ranges:
                min_val, max_val = self.expected_ranges[col]

                if min_val is not None:
                    below_min = df[df[col] < min_val]
                    if not below_min.empty:
                        outliers[f"{col}_below_min"] = len(below_min)
                        all_valid = False

                if max_val is not None:
                    above_max = df[df[col] > max_val]
                    if not above_max.empty:
                        outliers[f"{col}_above_max"] = len(above_max)
                        all_valid = False

        return {'valid': all_valid, 'outliers': outliers}

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using forward-fill and interpolation.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with filled values
        """
        df_filled = df.copy()

        # Forward fill for most indicators (daily data)
        forward_fill_cols = [
            'yield_10y_3m', 'yield_10y_2y', 'credit_spread_bbb',
            'vix_close', 'sp500_close', 'consumer_sentiment'
        ]
        for col in forward_fill_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].ffill()

        # Interpolate continuous indicators (monthly/quarterly data)
        interpolate_cols = [
            'real_gdp', 'cpi', 'industrial_production',
            'unemployment_rate', 'housing_starts', 'm2_money_supply'
        ]
        for col in interpolate_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].interpolate(method='linear')

        # Fill remaining with backward fill
        df_filled = df_filled.bfill()

        return df_filled

    def calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-1).

        Args:
            df: Input dataframe

        Returns:
            Quality score between 0 and 1
        """
        missing_validation = self.validate_missing_values(df)
        range_validation = self.validate_value_ranges(df)

        # Score based on missing values
        missing_pct = missing_validation['missing_percentage']
        missing_score = max(0, 1 - (missing_pct / 100))

        # Score based on outliers
        outlier_count = len(range_validation['outliers'])
        outlier_score = max(0, 1 - (outlier_count / (df.shape[0] * df.shape[1])))

        # Combined score
        quality_score = (missing_score * 0.6) + (outlier_score * 0.4)

        return quality_score

    def validate_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive validation of dataframe.

        Args:
            df: Input dataframe

        Returns:
            Dict with all validation results
        """
        return {
            'missing_values': self.validate_missing_values(df),
            'value_ranges': self.validate_value_ranges(df),
            'quality_score': self.calculate_data_quality_score(df),
            'shape': df.shape,
            'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else None
        }

