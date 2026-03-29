"""
Bry-Boschan Algorithm for Business Cycle Regime Detection

Identifies peaks and troughs in economic data to determine market regimes:
- Expansion (Bull market)
- Contraction (Bear market)
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from src.utils.config import (
    BOSCHAN_MIN_DURATION,
    BOSCHAN_MIN_AMPLITUDE,
    BOSCHAN_PHASE_LENGTH
)

logger = logging.getLogger(__name__)


class BryBoschanDetector:
    """Implements Bry-Boschan algorithm for regime detection."""

    def __init__(
        self,
        min_duration: int = BOSCHAN_MIN_DURATION,
        min_amplitude: float = BOSCHAN_MIN_AMPLITUDE,
        phase_length: int = BOSCHAN_PHASE_LENGTH
    ):
        """
        Initialize Bry-Boschan detector.

        Args:
            min_duration: Minimum duration of a phase (months)
            min_amplitude: Minimum amplitude for peak/trough (%)
            phase_length: Length of phase for local extrema (months)
        """
        self.min_duration = min_duration
        self.min_amplitude = min_amplitude
        self.phase_length = phase_length
        self.logger = logging.getLogger(__name__)

    def detect_regimes(self, series: pd.Series) -> pd.DataFrame:
        """
        Detect market regimes using Bry-Boschan algorithm.

        Args:
            series: Time series to analyze (e.g., S&P 500 index)

        Returns:
            DataFrame with regime labels (1=Expansion, 0=Contraction)

        Raises:
            ValueError: If series is empty or too short
        """
        if series.empty or len(series) < 24:
            raise ValueError("Series must have at least 24 observations")

        # Step 1: Find local extrema
        peaks, troughs = self._find_extrema(series)

        # Step 2: Apply duration and amplitude filters
        peaks, troughs = self._filter_extrema(series, peaks, troughs)

        # Step 3: Classify regimes
        regimes = self._classify_regimes(series, peaks, troughs)

        self.logger.info(f"Detected {len(peaks)} peaks and {len(troughs)} troughs")
        return regimes

    def _find_extrema(self, series: pd.Series) -> Tuple[List[int], List[int]]:
        """
        Find local peaks and troughs.

        Args:
            series: Time series

        Returns:
            Tuple of (peaks, troughs) as index lists
        """
        peaks = []
        troughs = []

        # Use rolling window to find local extrema
        for i in range(self.phase_length, len(series) - self.phase_length):
            window = series.iloc[i - self.phase_length:i + self.phase_length + 1]

            # Peak: local maximum
            if series.iloc[i] == window.max():
                peaks.append(i)

            # Trough: local minimum
            if series.iloc[i] == window.min():
                troughs.append(i)

        return peaks, troughs

    def _filter_extrema(
        self,
        series: pd.Series,
        peaks: List[int],
        troughs: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply duration and amplitude filters.

        Args:
            series: Time series
            peaks: Peak indices
            troughs: Trough indices

        Returns:
            Filtered (peaks, troughs)
        """
        filtered_peaks = []
        filtered_troughs = []

        # Filter peaks
        for i, peak in enumerate(peaks):
            # Check minimum duration from previous trough
            if i == 0:
                prev_trough_idx = 0
            else:
                if i - 1 < len(troughs):
                    prev_trough_idx = troughs[i - 1]
                else:
                    continue

            if i == 0 or (peak - prev_trough_idx) >= self.min_duration:
                # Check amplitude
                prev_trough = series.iloc[prev_trough_idx]
                amplitude = (series.iloc[peak] - prev_trough) / abs(prev_trough) * 100
                if amplitude >= self.min_amplitude:
                    filtered_peaks.append(peak)

        # Filter troughs
        for i, trough in enumerate(troughs):
            # Check minimum duration from previous peak
            if i == 0:
                prev_peak_idx = 0
            else:
                if i - 1 < len(peaks):
                    prev_peak_idx = peaks[i - 1]
                else:
                    continue

            if i == 0 or (trough - prev_peak_idx) >= self.min_duration:
                # Check amplitude
                prev_peak = series.iloc[prev_peak_idx]
                amplitude = (prev_peak - series.iloc[trough]) / abs(prev_peak) * 100
                if amplitude >= self.min_amplitude:
                    filtered_troughs.append(trough)

        return filtered_peaks, filtered_troughs

    def _classify_regimes(
        self,
        series: pd.Series,
        peaks: List[int],
        troughs: List[int]
    ) -> pd.DataFrame:
        """
        Classify each period as expansion (1) or contraction (0).

        Args:
            series: Time series
            peaks: Peak indices
            troughs: Trough indices

        Returns:
            DataFrame with regime classification
        """
        regimes = pd.DataFrame(index=series.index)
        regimes['regime'] = 1  # Default to expansion

        # Mark contractions (between peak and trough)
        for i in range(len(peaks)):
            if i < len(troughs):
                peak_idx = peaks[i]
                trough_idx = troughs[i]
                if peak_idx < trough_idx:
                    regimes.iloc[peak_idx:trough_idx + 1, 0] = 0

        # Add regime duration
        regimes['regime_duration'] = 0
        current_regime = regimes['regime'].iloc[0]
        duration = 1

        for i in range(1, len(regimes)):
            if regimes['regime'].iloc[i] == current_regime:
                duration += 1
            else:
                current_regime = regimes['regime'].iloc[i]
                duration = 1
            regimes.loc[regimes.index[i], 'regime_duration'] = duration

        # Add peak/trough indicators
        regimes['is_peak'] = 0
        regimes['is_trough'] = 0

        for peak in peaks:
            if peak < len(regimes):
                regimes.loc[regimes.index[peak], 'is_peak'] = 1

        for trough in troughs:
            if trough < len(regimes):
                regimes.loc[regimes.index[trough], 'is_trough'] = 1

        return regimes

    def get_current_regime(self, series: pd.Series) -> Dict:
        """
        Get current market regime.

        Args:
            series: Time series

        Returns:
            Dict with regime info (regime, duration, probability)
        """
        regimes = self.detect_regimes(series)
        current = regimes.iloc[-1]

        return {
            'regime': 'Expansion' if current['regime'] == 1 else 'Contraction',
            'regime_code': int(current['regime']),
            'duration_months': int(current['regime_duration']),
            'is_peak': bool(current['is_peak']),
            'is_trough': bool(current['is_trough'])
        }

