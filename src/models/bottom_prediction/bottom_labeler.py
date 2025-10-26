"""Bottom Labeler - Generates bottom labels from historical price data."""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BottomLabeler:
    """Generate bottom labels from price data."""

    def __init__(self, lookback_window: int = 60, lookforward_window: int = 60):
        """
        Initialize bottom labeler.

        Args:
            lookback_window: Days to look back for crash detection
            lookforward_window: Days to look forward for bottom detection
        """
        self.lookback_window = lookback_window
        self.lookforward_window = lookforward_window
        self.logger = logging.getLogger(__name__)

    def generate_labels(self, prices: pd.Series) -> np.ndarray:
        """
        Generate bottom labels (days to bottom from crash).

        Args:
            prices: Price series

        Returns:
            Array of days to bottom for each crash point
        """
        labels = np.full(len(prices), np.nan)

        for i in range(self.lookback_window, len(prices) - self.lookforward_window):
            # Look back for crash (20% drawdown)
            lookback_prices = prices.iloc[i - self.lookback_window:i]
            peak = lookback_prices.max()
            current = prices.iloc[i]
            drawdown = (peak - current) / peak if peak > 0 else 0

            if drawdown >= 0.20:  # Crash detected
                # Find bottom in forward window
                forward_prices = prices.iloc[i:i + self.lookforward_window]
                bottom_idx = forward_prices.idxmin()
                bottom_date_idx = prices.index.get_loc(bottom_idx)
                days_to_bottom = bottom_date_idx - i

                if 0 <= days_to_bottom <= self.lookforward_window:
                    labels[i] = days_to_bottom

        return labels

    def get_bottom_statistics(self, prices: pd.Series, labels: np.ndarray) -> dict:
        """
        Get statistics about bottom predictions.

        Args:
            prices: Price series
            labels: Bottom labels

        Returns:
            Dictionary with statistics
        """
        valid_labels = labels[~np.isnan(labels)]

        if len(valid_labels) == 0:
            return {
                'total_bottoms': 0,
                'mean_days_to_bottom': 0,
                'median_days_to_bottom': 0,
                'std_days_to_bottom': 0,
                'min_days_to_bottom': 0,
                'max_days_to_bottom': 0,
            }

        return {
            'total_bottoms': len(valid_labels),
            'mean_days_to_bottom': float(np.mean(valid_labels)),
            'median_days_to_bottom': float(np.median(valid_labels)),
            'std_days_to_bottom': float(np.std(valid_labels)),
            'min_days_to_bottom': float(np.min(valid_labels)),
            'max_days_to_bottom': float(np.max(valid_labels)),
        }

