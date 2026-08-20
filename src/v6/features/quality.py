"""Data-quality guards for the v6 feature layer.

The v6.0.0-alpha post-mortem found that several DB columns were not real
market data, and that nothing in the pipeline noticed. A constant fill and
a synthetic random walk both look like "data" to a StandardScaler; they
silently poison a covariance matrix, an HMM emission, and a k-NN metric.

This module makes data quality an explicit, testable gate rather than an
assumption. Every column entering the feature builder is screened, and a
column that fails is **excluded with a recorded reason** instead of being
quietly modelled.

Screens
-------
``constant_fill``
    A long run of one repeated value, e.g. ``vix_close`` was the constant
    17.24 for every row before 1990 — a placeholder, not a quote.

``degenerate_variance``
    Near-zero variance relative to level: carries no information but can
    blow up a z-score when divided by a tiny standard deviation.

``implausible_noise``
    A series that claims to be a market sentiment/flow measure but has the
    statistical signature of generated noise: very low lag-1
    autocorrelation combined with an implausibly narrow spread. Real daily
    financial series are strongly autocorrelated in level and fat-tailed.
    ``put_call_ratio`` fails this (autocorrelation 0.09, std 0.033, and no
    reaction to Black Monday, the GFC, or the COVID crash).

``insufficient_history``
    Fewer than ``min_observations`` non-null values, or a first valid date
    so late the column cannot support walk-forward folds.

Usage
-----
    report = screen_dataframe(raw)
    print(report.to_string())
    clean = apply_quality_screen(raw)      # failing columns -> dropped
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Columns that are known-fabricated in the legacy database and must never be
# used as features, regardless of what the automated screens say. Each entry
# records the evidence, so the quarantine is auditable and reversible: if the
# column is ever repopulated from a real source, delete the entry.
QUARANTINE: Dict[str, str] = {
    "put_call_ratio": (
        "Synthetic noise centred on 1.0 (std 0.033, lag-1 autocorrelation "
        "0.09) with no reaction to 1987-10-19, 2008-10-10 or 2020-03-16. "
        "Real CBOE equity put/call starts 1995 and spikes above 1.2 in "
        "panics. No free full-history source is available; the feature is "
        "dropped rather than modelled."
    ),
    "in_crash": "Target label — using it as a feature is leakage.",
    "pre_crash_30d": "Target label — using it as a feature is leakage.",
    "pre_crash_60d": "Target label — using it as a feature is leakage.",
    "data_quality_score": "Bookkeeping column, not market data.",
}

# Bookkeeping / identifier columns that are not features.
NON_FEATURE_COLUMNS = {"id", "created_at", "updated_at", "date"}


@dataclass(frozen=True)
class QualityThresholds:
    """Tunable screen thresholds."""
    min_observations: int = 252
    max_constant_run_frac: float = 0.10   # >10% of history as one repeated value
    min_relative_std: float = 1e-6
    noise_autocorr_max: float = 0.30      # real daily levels are far above this
    noise_relative_std_max: float = 0.05  # ...and vary more than this


@dataclass
class ColumnVerdict:
    column: str
    passed: bool
    reason: str
    n_obs: int
    first_valid: Optional[pd.Timestamp]
    last_valid: Optional[pd.Timestamp]
    longest_constant_run: int
    lag1_autocorr: float
    relative_std: float


def _longest_constant_run(s: pd.Series) -> int:
    """Length of the longest run of an identical repeated value."""
    v = s.dropna().values
    if len(v) == 0:
        return 0
    # Boundaries where the value changes; run lengths are the gaps between.
    change_points = np.flatnonzero(np.diff(v) != 0)
    starts = np.concatenate([[0], change_points + 1])
    ends = np.concatenate([change_points + 1, [len(v)]])
    return int((ends - starts).max())


def screen_column(s: pd.Series, name: str,
                  thresholds: Optional[QualityThresholds] = None) -> ColumnVerdict:
    """Run every screen against one column and return a verdict."""
    th = thresholds or QualityThresholds()
    clean = pd.to_numeric(s, errors="coerce").dropna()
    n = len(clean)
    first = clean.index.min() if n else None
    last = clean.index.max() if n else None

    run = _longest_constant_run(clean) if n else 0
    autocorr = float(clean.autocorr(1)) if n > 2 else np.nan
    level = float(np.abs(clean).mean()) if n else 0.0
    rel_std = float(clean.std() / level) if n and level > 0 else 0.0

    def verdict(passed: bool, reason: str) -> ColumnVerdict:
        return ColumnVerdict(
            column=name, passed=passed, reason=reason, n_obs=n,
            first_valid=first, last_valid=last, longest_constant_run=run,
            lag1_autocorr=autocorr, relative_std=rel_std,
        )

    if name in QUARANTINE:
        return verdict(False, f"quarantined: {QUARANTINE[name]}")
    if n < th.min_observations:
        return verdict(False, f"insufficient_history: {n} obs < {th.min_observations}")
    if run > th.max_constant_run_frac * n:
        return verdict(
            False,
            f"constant_fill: longest repeated-value run is {run} rows "
            f"({100 * run / n:.1f}% of history)",
        )
    if rel_std < th.min_relative_std:
        return verdict(False, f"degenerate_variance: relative std {rel_std:.2e}")
    if (
        np.isfinite(autocorr)
        and autocorr < th.noise_autocorr_max
        and 0 < rel_std < th.noise_relative_std_max
    ):
        return verdict(
            False,
            f"implausible_noise: lag-1 autocorrelation {autocorr:.3f} with "
            f"relative std {rel_std:.4f} — signature of generated noise, "
            f"not a market series",
        )
    return verdict(True, "ok")


def screen_dataframe(df: pd.DataFrame,
                     thresholds: Optional[QualityThresholds] = None) -> pd.DataFrame:
    """Screen every numeric column; return a tidy report DataFrame."""
    verdicts: List[ColumnVerdict] = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        verdicts.append(screen_column(df[col], col, thresholds))
    report = pd.DataFrame([v.__dict__ for v in verdicts])
    if report.empty:
        return report
    return report.sort_values(["passed", "n_obs"], ascending=[True, False])


def apply_quality_screen(df: pd.DataFrame,
                         thresholds: Optional[QualityThresholds] = None,
                         verbose: bool = False) -> pd.DataFrame:
    """Return `df` with quality-failing columns removed."""
    report = screen_dataframe(df, thresholds)
    if report.empty:
        return df
    failed = report.loc[~report["passed"], "column"].tolist()
    if verbose and failed:
        for _, row in report[~report["passed"]].iterrows():
            print(f"  DROP {row['column']:24s} {row['reason']}")
    return df.drop(columns=[c for c in failed if c in df.columns])


def rejected_columns(df: pd.DataFrame,
                     thresholds: Optional[QualityThresholds] = None) -> Dict[str, str]:
    """Map of column -> rejection reason, for logging and dashboards."""
    report = screen_dataframe(df, thresholds)
    if report.empty:
        return {}
    bad = report[~report["passed"]]
    return dict(zip(bad["column"], bad["reason"]))
