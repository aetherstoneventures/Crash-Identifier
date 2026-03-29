"""Feature Engineering Pipeline.

Orchestrates indicator calculation, redundancy removal, and normalization.

Design principles:
1. fit_transform() learns statistics from TRAINING data only.
2. transform() applies those learned statistics to new data.
3. No look-ahead: scaler is never fit on the full dataset.
4. Redundant feature removal is deterministic (by variance ranking).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.crash_indicators import CrashIndicators
from src.utils.config import FEATURE_CORRELATION_THRESHOLD, RANDOM_STATE

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Feature pipeline with proper train/test separation.

    Usage:
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(raw_train_data)
        X_test = pipeline.transform(raw_test_data)
    """

    def __init__(
        self,
        correlation_threshold: float = FEATURE_CORRELATION_THRESHOLD,
    ):
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.columns_to_drop: List[str] = []
        self.columns_to_scale: List[str] = []

    def compute_indicators(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators from raw data. Stateless, no fitting.

        Args:
            raw_data: Raw indicator data from data collection.

        Returns:
            DataFrame with calculated indicators.
        """
        return CrashIndicators.calculate_all(raw_data)

    def fit_transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fit pipeline on training data and return transformed features.

        Learns: which features to drop (redundancy), scaler parameters.
        Use ONLY on the training set.

        Args:
            raw_data: Raw training data.

        Returns:
            Transformed feature DataFrame.
        """
        logger.info("FeaturePipeline: fit_transform on training data...")

        indicators = self.compute_indicators(raw_data)

        # Forward-fill then backward-fill within available data
        indicators = indicators.ffill().bfill()

        # Determine redundant features from THIS data only
        self.columns_to_drop = self._find_redundant(indicators)
        if self.columns_to_drop:
            logger.info(
                f"Dropping {len(self.columns_to_drop)} redundant features: "
                f"{self.columns_to_drop}"
            )
        indicators = indicators.drop(
            columns=self.columns_to_drop, errors='ignore'
        )

        # Fill any remaining NaN with column median (from THIS data only)
        for col in indicators.columns:
            if indicators[col].isna().any():
                indicators[col] = indicators[col].fillna(
                    indicators[col].median()
                )

        # Identify columns to scale (numeric, non-constant)
        numeric_cols = indicators.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.columns_to_scale = [
            c for c in numeric_cols if indicators[c].std() > 1e-10
        ]

        # Fit scaler on TRAINING data only
        if self.columns_to_scale:
            indicators[self.columns_to_scale] = self.scaler.fit_transform(
                indicators[self.columns_to_scale]
            )

        self.feature_names = indicators.columns.tolist()
        self.is_fitted = True

        logger.info(
            f"Pipeline fitted: {len(self.feature_names)} features, "
            f"{len(indicators)} samples"
        )
        return indicators

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using parameters learned at fit time.

        Use on validation/test sets.

        Args:
            raw_data: Raw data to transform.

        Returns:
            Transformed feature DataFrame with same columns as fit.

        Raises:
            ValueError: If pipeline has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                "Pipeline not fitted. Call fit_transform() first."
            )

        indicators = self.compute_indicators(raw_data)
        indicators = indicators.ffill().bfill()

        # Drop same features as training
        indicators = indicators.drop(
            columns=self.columns_to_drop, errors='ignore'
        )

        # Fill NaN with 0 for test data (no look-ahead from test stats)
        for col in indicators.columns:
            if indicators[col].isna().any():
                indicators[col] = indicators[col].fillna(0.0)

        # Scale using FITTED scaler — no re-fitting
        cols_present = [
            c for c in self.columns_to_scale if c in indicators.columns
        ]
        if cols_present:
            indicators[cols_present] = self.scaler.transform(
                indicators[cols_present]
            )

        # Ensure same column order as training
        missing_cols = set(self.feature_names) - set(indicators.columns)
        for col in missing_cols:
            indicators[col] = 0.0
        indicators = indicators[self.feature_names]

        return indicators

    def _find_redundant(self, df: pd.DataFrame) -> List[str]:
        """Find redundant features by pairwise correlation.

        When two features exceed the correlation threshold, the one with
        LOWER variance is dropped. This is deterministic (not arbitrary).

        Args:
            df: Feature DataFrame.

        Returns:
            List of column names to drop.
        """
        corr = df.corr().abs()
        redundant = set()
        cols = corr.columns.tolist()

        for i in range(len(cols)):
            if cols[i] in redundant:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in redundant:
                    continue
                if corr.iloc[i, j] > self.correlation_threshold:
                    # Drop the feature with lower variance
                    var_i = df[cols[i]].var()
                    var_j = df[cols[j]].var()
                    drop = cols[j] if var_j <= var_i else cols[i]
                    redundant.add(drop)
                    logger.debug(
                        f"Corr({cols[i]}, {cols[j]}) = "
                        f"{corr.iloc[i, j]:.3f} > {self.correlation_threshold}"
                        f" → dropping '{drop}'"
                    )

        return sorted(redundant)

    def get_feature_names(self) -> List[str]:
        """Return the feature names after fitting."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted.")
        return list(self.feature_names)
