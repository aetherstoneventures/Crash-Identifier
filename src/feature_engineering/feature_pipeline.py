"""
Feature Engineering Pipeline

Orchestrates the complete feature engineering process:
1. Calculate 28 crash indicators
2. Detect market regimes (Bry-Boschan)
3. Analyze feature correlations
4. Remove redundant features
5. Normalize and prepare for modeling
"""

import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.feature_engineering.crash_indicators import CrashIndicators
from src.feature_engineering.regime_detection import BryBoschanDetector
from src.utils.config import (
    FEATURE_CORRELATION_THRESHOLD,
    FEATURE_NORMALIZATION_METHOD,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates feature engineering pipeline."""

    def __init__(self):
        """Initialize feature pipeline."""
        self.indicators = None
        self.regimes = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.logger = logging.getLogger(__name__)

    def process(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Process raw data through complete feature pipeline.

        Args:
            raw_data: Raw indicator data from data collection

        Returns:
            Tuple of (features_df, metadata_dict)

        Raises:
            ValueError: If data is invalid
        """
        if raw_data.empty:
            raise ValueError("Raw data is empty")

        self.logger.info("Starting feature engineering pipeline...")

        # Step 1: Calculate 28 indicators
        self.logger.info("Step 1: Calculating 28 crash indicators...")
        self.indicators = CrashIndicators.calculate_all_indicators(raw_data)

        # Step 2: Detect market regimes
        self.logger.info("Step 2: Detecting market regimes (Bry-Boschan)...")
        if 'sp500_close' in raw_data.columns:
            detector = BryBoschanDetector()
            self.regimes = detector.detect_regimes(raw_data['sp500_close'])
        else:
            self.logger.warning("S&P 500 data not available for regime detection")
            self.regimes = pd.DataFrame(index=raw_data.index)

        # Step 3: Analyze correlations
        self.logger.info("Step 3: Analyzing feature correlations...")
        correlation_matrix = self.indicators.corr()
        redundant_features = self._identify_redundant_features(correlation_matrix)

        # Step 4: Remove redundant features
        self.logger.info(f"Step 4: Removing {len(redundant_features)} redundant features...")
        features = self.indicators.drop(columns=redundant_features, errors='ignore')

        # Step 5: Handle missing values
        self.logger.info("Step 5: Handling missing values...")
        features = features.ffill().bfill()

        # Step 6: Normalize features
        self.logger.info("Step 6: Normalizing features...")
        features_normalized = self._normalize_features(features)

        # Step 7: Combine with regimes
        if not self.regimes.empty:
            features_normalized['regime'] = self.regimes['regime']
            if 'regime_duration' in self.regimes.columns:
                features_normalized['regime_duration'] = self.regimes['regime_duration']

        # Step 8: Final NaN handling - fill remaining NaN values
        self.logger.info("Step 8: Final NaN handling...")
        initial_nan_count = features_normalized.isna().sum().sum()
        if initial_nan_count > 0:
            # Forward fill then backward fill
            features_normalized = features_normalized.ffill().bfill()
            # If still NaN (e.g., at start), fill with column mean
            for col in features_normalized.columns:
                if features_normalized[col].isna().any():
                    features_normalized[col] = features_normalized[col].fillna(features_normalized[col].mean())
            self.logger.info(f"Filled {initial_nan_count} NaN values")

        if len(features_normalized) == 0:
            raise ValueError("No valid features after NaN removal. Check data quality.")

        self.feature_names = features_normalized.columns.tolist()

        # Create metadata
        metadata = {
            'n_features': len(features_normalized.columns),
            'n_samples': len(features_normalized),
            'feature_names': self.feature_names,
            'redundant_features': redundant_features,
            'correlation_threshold': FEATURE_CORRELATION_THRESHOLD,
            'normalization_method': FEATURE_NORMALIZATION_METHOD,
            'date_range': f"{features_normalized.index[0]} to {features_normalized.index[-1]}"
        }

        self.logger.info(f"Feature pipeline complete: {metadata['n_features']} features, "
                        f"{metadata['n_samples']} samples")

        return features_normalized, metadata

    def _identify_redundant_features(self, corr_matrix: pd.DataFrame) -> list:
        """
        Identify redundant features based on correlation threshold.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            List of redundant feature names to drop
        """
        redundant = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > FEATURE_CORRELATION_THRESHOLD:
                    # Drop the feature with lower variance
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]

                    # Skip regime-related columns
                    if 'regime' in col_i or 'regime' in col_j:
                        continue

                    redundant.add(col_j)

        return list(redundant)

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using configured method.

        Args:
            features: Feature DataFrame

        Returns:
            Normalized features
        """
        if FEATURE_NORMALIZATION_METHOD == 'standard':
            # StandardScaler: (x - mean) / std
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features_normalized = features.copy()

            # Handle NaN and constant columns
            for col in numeric_cols:
                # Fill any remaining NaN values
                if features[col].isna().any():
                    features_normalized[col] = features[col].fillna(features[col].mean())

            # Only scale columns with variance > 0
            cols_to_scale = [col for col in numeric_cols if features_normalized[col].std() > 0]
            if cols_to_scale:
                features_normalized[cols_to_scale] = self.scaler.fit_transform(features_normalized[cols_to_scale])

            return features_normalized

        elif FEATURE_NORMALIZATION_METHOD == 'minmax':
            # MinMax: (x - min) / (max - min)
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features_normalized = features.copy()
            for col in numeric_cols:
                min_val = features[col].min()
                max_val = features[col].max()
                if max_val > min_val:
                    features_normalized[col] = (features[col] - min_val) / (max_val - min_val)
            return features_normalized

        else:
            # No normalization
            return features

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature statistics for importance analysis.

        Returns:
            DataFrame with feature statistics
        """
        if self.indicators is None:
            raise ValueError("Pipeline not yet processed")

        stats = pd.DataFrame({
            'feature': self.indicators.columns,
            'mean': self.indicators.mean(),
            'std': self.indicators.std(),
            'min': self.indicators.min(),
            'max': self.indicators.max(),
            'skew': self.indicators.skew(),
            'kurtosis': self.indicators.kurtosis()
        })

        return stats.sort_values('std', ascending=False)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of all features.

        Returns:
            Correlation matrix
        """
        if self.indicators is None:
            raise ValueError("Pipeline not yet processed")

        return self.indicators.corr()

