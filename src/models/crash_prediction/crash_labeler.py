"""Crash Labeler — single source of truth for crash labels.

Generates binary labels from price data for model training.

THE labeling rule (used by ALL training scripts):
    For each date t, label = 1 if the S&P 500 price falls by at least
    `drawdown_threshold` percent from its ROLLING PEAK within the next
    `lookforward_window` trading days.

This answers: "Looking forward from today, will there be a significant
drawdown from the recent peak in the near future?"

Why rolling peak (not current price):
    Using the rolling peak catches the START of crashes — when the price
    just begins falling from a recent high. Using current price would miss
    dates that are already mid-decline (giving false negatives).

No look-ahead in the labeling definition:
    The label at date t depends on future prices (t+1 to t+W). This is
    intentional — we are creating a TARGET variable. The FEATURES must
    never contain future information. Walk-forward validation ensures
    the model is evaluated on dates it has never seen during training.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import CRASH_DRAWDOWN_THRESHOLD

logger = logging.getLogger(__name__)


class CrashLabeler:
    """Generate crash labels from price data.

    This is the SINGLE labeling scheme used throughout the project.
    All training scripts should use this class instead of implementing
    their own labeling logic.
    """

    def __init__(
        self,
        drawdown_threshold: float = CRASH_DRAWDOWN_THRESHOLD,
        lookforward_window: int = 60,
        rolling_peak_window: int = 252,
    ):
        """
        Args:
            drawdown_threshold: Minimum drawdown to label as crash.
                E.g., 0.20 means a 20% peak-to-trough decline.
            lookforward_window: Days to look forward for crash detection.
            rolling_peak_window: Window for computing the rolling peak.
                252 ≈ 1 year of trading days.
        """
        self.drawdown_threshold = drawdown_threshold
        self.lookforward_window = lookforward_window
        self.rolling_peak_window = rolling_peak_window

    def generate_labels(self, prices: pd.Series) -> np.ndarray:
        """Generate binary crash labels from a price series.

        Algorithm:
            1. Compute the rolling peak over the past `rolling_peak_window` days.
            2. For each date t, look at prices in [t, t + lookforward_window].
            3. Compute the maximum drawdown from the rolling peak at t
               to the minimum price in the forward window.
            4. If drawdown >= threshold, label = 1. Else label = 0.
            5. Last `lookforward_window` dates get label 0 (can't look forward).

        Args:
            prices: Price series (e.g., S&P 500 close prices).
                Must have at least lookforward_window + rolling_peak_window
                observations.

        Returns:
            Binary labels array (1 = crash ahead, 0 = no crash).
        """
        n = len(prices)
        min_required = self.lookforward_window + self.rolling_peak_window
        if n < min_required:
            raise ValueError(
                f"Price series has {n} observations, need at least "
                f"{min_required} (lookforward={self.lookforward_window} "
                f"+ peak_window={self.rolling_peak_window})"
            )

        prices_arr = prices.values.astype(float)
        labels = np.zeros(n, dtype=int)

        # Rolling peak: max over past rolling_peak_window days (inclusive)
        rolling_peak = (
            pd.Series(prices_arr)
            .rolling(window=self.rolling_peak_window, min_periods=1)
            .max()
            .values
        )

        for i in range(n - self.lookforward_window):
            peak = rolling_peak[i]
            if peak <= 0 or np.isnan(peak):
                continue

            # Minimum price in forward window
            forward_min = np.nanmin(
                prices_arr[i : i + self.lookforward_window]
            )

            # Drawdown from rolling peak
            drawdown = (forward_min - peak) / peak

            if drawdown <= -self.drawdown_threshold:
                labels[i] = 1

        n_positive = int(labels.sum())
        logger.info(
            f"Generated labels: {n_positive} crash ({n_positive/n*100:.1f}%), "
            f"{n - n_positive} non-crash, "
            f"threshold={self.drawdown_threshold*100:.0f}%, "
            f"lookforward={self.lookforward_window}d"
        )
        return labels

    def get_statistics(
        self, prices: pd.Series, labels: np.ndarray
    ) -> dict:
        """Get statistics about the generated labels.

        Args:
            prices: Price series.
            labels: Binary labels from generate_labels().

        Returns:
            Dictionary with label statistics.
        """
        n_crash = int(np.sum(labels))
        n_total = len(labels)

        return {
            'total_samples': n_total,
            'crash_samples': n_crash,
            'non_crash_samples': n_total - n_crash,
            'crash_ratio': n_crash / n_total if n_total > 0 else 0.0,
            'drawdown_threshold': self.drawdown_threshold,
            'lookforward_window': self.lookforward_window,
            'rolling_peak_window': self.rolling_peak_window,
        }
