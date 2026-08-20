"""Tunable crash episode extractor.

This module operationalises point 1 from the design brief:

> when we say crash, this should be a tunable parameter! i.e. we should be
> able to extract the same x% reductions in the market from the historical
> data. x could be 2% or 20%, does not matter!!! For the data or "crash"
> part, it is like a matching algorithm in the historical data based on
> several indicators of the price reduction.

The crash threshold x% is therefore a *parameter of the extractor*, not a
fixed constant baked into model training. The Engine 3 analog matcher is
trained once on the full daily history, and at inference the user picks
any x in [2%, 50%] and any h in {21, 63, 126, 252} trading days.

Algorithm (peak-to-trough segmentation):

1. Walk forward through the price series.
2. Track the running 252-trading-day rolling maximum (the "peak").
3. A drawdown EPISODE starts when price first falls below peak by >= x%.
4. It ENDS when the price recovers back to that peak.
5. Filter out episodes whose duration is below `min_duration_td`.
6. Optionally clip overlapping episodes (post-recovery starts new peak).

The result is a list of `CrashEpisode` objects with: peak date/price,
trough date/price/drawdown, recovery date, duration (peak->trough and
peak->recovery), and the start/end indices for downstream KPI snapshots.

NO LOOK-AHEAD: this is purely a labeling pass over the *full* price
history. It is only used (a) to define the "normal" training subset for
Engine 1 and (b) for analog inspection in the dashboard. The Engine 3
forward-outcome computation does NOT depend on these episode labels — it
just looks at realised forward maxDD over h trading days at each date.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class CrashEpisode:
    """A peak-to-trough drawdown episode with metadata."""
    peak_date: pd.Timestamp
    peak_price: float
    trough_date: pd.Timestamp
    trough_price: float
    recovery_date: pd.Timestamp | None  # None if never recovered in sample
    drawdown_pct: float                  # positive number, e.g. 23.5
    duration_to_trough_td: int
    duration_to_recovery_td: int | None
    severity_tier: str                   # 'minor'|'moderate'|'major'|'severe'|'extreme'


def _classify_severity(dd_pct: float) -> str:
    if dd_pct < 5.0:
        return "micro"
    if dd_pct < 10.0:
        return "minor"
    if dd_pct < 15.0:
        return "moderate"
    if dd_pct < 20.0:
        return "major"
    if dd_pct < 30.0:
        return "severe"
    return "extreme"


def extract_crashes(
    prices: pd.Series,
    x_pct: float,
    min_duration_td: int = 5,
    peak_lookback_td: int = 252,
) -> List[CrashEpisode]:
    """Extract crash episodes >= x% drawdown from `prices`.

    Parameters
    ----------
    prices : pd.Series
        Indexed by trading date, values = closing price (any equity index).
    x_pct : float
        Minimum drawdown threshold, in percent (positive). e.g. 10.0 means
        a 10% decline from the rolling peak.
    min_duration_td : int
        Minimum number of trading days from peak to trough for the
        episode to count. Filters out single-day noise spikes.
    peak_lookback_td : int
        Rolling window used to define the running peak. 252 = 1 year.

    Returns
    -------
    List[CrashEpisode]
        Episodes are non-overlapping (a new episode starts only after the
        prior peak is recovered).
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if x_pct <= 0:
        raise ValueError("x_pct must be positive")

    px = prices.dropna().astype(float)
    if len(px) < peak_lookback_td:
        return []

    rolling_peak = px.rolling(peak_lookback_td, min_periods=1).max()
    drawdown = (px / rolling_peak - 1.0) * 100.0  # negative values

    episodes: List[CrashEpisode] = []
    in_episode = False
    peak_idx: int | None = None
    peak_price: float | None = None
    trough_idx: int | None = None
    trough_price: float | None = None

    idx = px.index
    vals = px.values
    dd_vals = drawdown.values
    peak_vals = rolling_peak.values

    for i in range(len(px)):
        if not in_episode:
            if dd_vals[i] <= -x_pct:
                # Episode begins; the peak is the most recent prior high.
                # Locate it by scanning backward for the date where
                # rolling_peak first reached its current level.
                current_peak = peak_vals[i]
                # Walk back to find that peak's location.
                j = i
                while j > 0 and vals[j] < current_peak:
                    j -= 1
                peak_idx = j
                peak_price = float(vals[j])
                trough_idx = i
                trough_price = float(vals[i])
                in_episode = True
        else:
            # Track deepest trough.
            if vals[i] < trough_price:
                trough_idx = i
                trough_price = float(vals[i])
            # Recovery check.
            if vals[i] >= peak_price:
                recovery_idx = i
                dd_pct = (peak_price - trough_price) / peak_price * 100.0
                duration_to_trough = trough_idx - peak_idx
                duration_to_recovery = recovery_idx - peak_idx
                if duration_to_trough >= min_duration_td:
                    episodes.append(
                        CrashEpisode(
                            peak_date=idx[peak_idx],
                            peak_price=peak_price,
                            trough_date=idx[trough_idx],
                            trough_price=trough_price,
                            recovery_date=idx[recovery_idx],
                            drawdown_pct=float(dd_pct),
                            duration_to_trough_td=int(duration_to_trough),
                            duration_to_recovery_td=int(duration_to_recovery),
                            severity_tier=_classify_severity(dd_pct),
                        )
                    )
                in_episode = False
                peak_idx = peak_price = trough_idx = trough_price = None

    # Open episode at end of sample (no recovery yet).
    if in_episode and peak_idx is not None:
        dd_pct = (peak_price - trough_price) / peak_price * 100.0
        duration_to_trough = trough_idx - peak_idx
        if duration_to_trough >= min_duration_td:
            episodes.append(
                CrashEpisode(
                    peak_date=idx[peak_idx],
                    peak_price=peak_price,
                    trough_date=idx[trough_idx],
                    trough_price=trough_price,
                    recovery_date=None,
                    drawdown_pct=float(dd_pct),
                    duration_to_trough_td=int(duration_to_trough),
                    duration_to_recovery_td=None,
                    severity_tier=_classify_severity(dd_pct),
                )
            )

    return episodes


def episodes_to_dataframe(episodes: List[CrashEpisode]) -> pd.DataFrame:
    """Convert a list of CrashEpisode to a tidy DataFrame."""
    if not episodes:
        return pd.DataFrame(
            columns=[
                "peak_date", "peak_price", "trough_date", "trough_price",
                "recovery_date", "drawdown_pct", "duration_to_trough_td",
                "duration_to_recovery_td", "severity_tier",
            ]
        )
    return pd.DataFrame([e.__dict__ for e in episodes])


def label_normal_days(
    dates: pd.DatetimeIndex,
    episodes: List[CrashEpisode],
    buffer_td: int = 20,
) -> np.ndarray:
    """Return boolean mask: True where the day is OUTSIDE any episode's
    [peak - buffer_td, recovery + buffer_td] window.

    Used by Engine 1 to define the "normal-regime" training subset.
    """
    mask = np.ones(len(dates), dtype=bool)
    date_index = pd.DatetimeIndex(dates)
    for ep in episodes:
        start = ep.peak_date - pd.tseries.offsets.BDay(buffer_td)
        end_anchor = ep.recovery_date if ep.recovery_date is not None else ep.trough_date
        end = end_anchor + pd.tseries.offsets.BDay(buffer_td)
        in_episode = (date_index >= start) & (date_index <= end)
        mask &= ~in_episode
    return mask
