"""The forward-outcome label — one definition, used everywhere.

WHY THIS MODULE EXISTS
======================
v6.0.0-alpha trained and evaluated against two different targets:

- Engine 3 learned from **forward maximum drawdown**: "starting today, does
  the index fall x% below its running peak at any point in the next h
  trading days?"
- The validation harness scored against **crash-episode onsets**: "does a
  peak-to-trough episode of at least x% *begin* within the next h days?",
  where episodes came from the segmentation in `crash_extractor.py`.

Those are not the same event. Episode onsets are anchored on the peak, so
the label is true only in the short window before a peak and false through
the entire decline that follows — while an engine that correctly says "we
are falling and will fall further" is scored wrong. Non-overlapping episode
segmentation also drops every drawdown that starts before a prior peak is
recovered, which is most of what happens inside a bear market. The measured
recall was partly an artefact of that mismatch.

This module defines the label once. `forward_maxdd` is the quantity the
system actually predicts, and `crash_label` is its thresholded form. The
analog engine, the aggregator's calibration, and the validation harness all
call these, so the number being optimised is the number being reported.

Episode segmentation still has a job — it defines "normal" days for the
anomaly engine and drives analog inspection in the dashboard — but it is no
longer the yardstick.

NO LOOK-AHEAD
=============
These are *labels*: by definition they read the future, which is legitimate
only for scoring and for fitting on a window whose future has already
happened. The caller is responsible for walk-forward discipline. Dates whose
horizon extends past the end of the supplied price series return NaN rather
than a truncated, optimistically-shallow drawdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_maxdd(prices: pd.Series, horizon_td: int) -> pd.Series:
    """Maximum drawdown over the next `horizon_td` trading days, in percent.

    For each date t, tracks the running peak across ``[t, t+h]`` — starting
    from the price at t — and returns the deepest decline below that peak as
    a positive percentage. A value of 12.5 means the index fell 12.5% from a
    peak within the window.

    Anchoring on the running peak rather than on price at t is what makes
    this a *drawdown* rather than a point-to-point return: a market that
    rises 5% and then falls 15% has suffered a 15% drawdown, and that is the
    experience the system is meant to warn about.

    Returns NaN for the final `horizon_td` dates, whose windows are
    incomplete.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if horizon_td <= 0:
        raise ValueError("horizon_td must be positive")

    px = prices.astype(float)
    values = px.values
    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=px.index)

    for i in range(n - horizon_td):
        window = values[i : i + horizon_td + 1]
        if np.isnan(window).any():
            continue
        running_peak = np.maximum.accumulate(window)
        drawdown = window / running_peak - 1.0
        out[i] = float(-drawdown.min() * 100.0)
    return pd.Series(out, index=px.index)


def crash_label(prices: pd.Series, x_pct: float, horizon_td: int) -> pd.Series:
    """Boolean label: does forward maxDD reach `x_pct` within the horizon?

    This is the event the posterior estimates:
    ``P(maxDD_(t, t+h] >= x% )``.

    Dates with an incomplete forward window are NaN (not False), so scoring
    code can drop them instead of silently counting them as non-events.
    """
    if x_pct <= 0:
        raise ValueError("x_pct must be positive")
    dd = forward_maxdd(prices, horizon_td)
    label = (dd >= float(x_pct)).astype(float)
    return label.where(dd.notna())


def label_coverage(prices: pd.Series, x_pct: float, horizon_td: int) -> dict:
    """Summary of the label on a price series — base rate and support.

    Used by the validation harness so a fold reports the prevalence it was
    scored against, rather than leaving a reader to guess whether "recall
    0.0" reflects a hard problem or an empty label set.
    """
    y = crash_label(prices, x_pct, horizon_td)
    scored = y.dropna()
    return {
        "n_dates": int(len(prices)),
        "n_scorable": int(len(scored)),
        "n_positive": int(scored.sum()),
        "base_rate": float(scored.mean()) if len(scored) else float("nan"),
    }
