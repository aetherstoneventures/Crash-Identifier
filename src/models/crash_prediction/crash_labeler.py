"""
Crash Labeler

Generates crash labels from historical price data for model training.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

from src.utils.config import CRASH_DRAWDOWN_THRESHOLD

logger = logging.getLogger(__name__)


class CrashLabeler:
    """Generate crash labels from price data."""

    def __init__(self, drawdown_threshold: float = CRASH_DRAWDOWN_THRESHOLD,
                 lookforward_window: int = 60):
        """
        Initialize crash labeler.

        Args:
            drawdown_threshold: Drawdown threshold for crash (e.g., 0.20 for 20%)
            lookforward_window: Days to look forward for crash detection
        """
        self.drawdown_threshold = drawdown_threshold
        self.lookforward_window = lookforward_window
        self.logger = logging.getLogger(__name__)

    def generate_labels(self, prices: pd.Series) -> np.ndarray:
        """
        Generate crash labels from price series.

        Args:
            prices: Price series (e.g., S&P 500 close prices)

        Returns:
            Binary labels (1 = crash within lookforward window, 0 = no crash)
        """
        if len(prices) < self.lookforward_window:
            raise ValueError(
                f"Price series must have at least {self.lookforward_window} observations"
            )

        labels = np.zeros(len(prices), dtype=int)

        for i in range(len(prices) - self.lookforward_window):
            current_price = prices.iloc[i]
            future_prices = prices.iloc[i:i + self.lookforward_window]

            # Calculate maximum drawdown in lookforward window
            max_price = future_prices.max()
            min_price = future_prices.min()
            drawdown = (min_price - current_price) / current_price

            # Label as crash if drawdown exceeds threshold
            if drawdown <= -self.drawdown_threshold:
                labels[i] = 1

        # Last few observations get label 0 (can't look forward)
        labels[-self.lookforward_window:] = 0

        return labels

    def get_crash_statistics(self, prices: pd.Series, labels: np.ndarray) -> dict:
        """
        Get statistics about crash labels.

        Args:
            prices: Price series
            labels: Binary crash labels

        Returns:
            Dictionary with crash statistics
        """
        n_crashes = np.sum(labels)
        crash_ratio = n_crashes / len(labels)

        return {
            'total_samples': len(labels),
            'crash_samples': n_crashes,
            'non_crash_samples': len(labels) - n_crashes,
            'crash_ratio': crash_ratio,
            'class_balance': {
                'crash': crash_ratio,
                'non_crash': 1 - crash_ratio,
            }
        }

