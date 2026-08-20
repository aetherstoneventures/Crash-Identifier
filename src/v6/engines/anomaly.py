"""Engine 1 — Density-based Anomaly Detector.

Answers: "Is *something* unusual about today's market state vs. normal regime?"

Method: an ensemble of two complementary one-class models:

1. **Mahalanobis distance** to the multivariate Gaussian of "normal-regime"
   days. Closed-form, gives an interpretable chi-squared p-value, captures
   linear correlation structure.

2. **Isolation Forest** (Liu et al. 2008). Captures non-linear pockets of
   anomaly that Mahalanobis misses. No distance metric assumed.

The two scores are RANK-AVERAGED to produce the final anomaly score in
[0, 1]. When the two models disagree substantially the engine emits a
low confidence flag (downstream aggregator down-weights it).

Training set definition ("normal" days):
    Days that are NOT within ±buffer trading days of any historical
    drawdown episode ≥ buffer_x_pct% (extracted via
    `src.v6.features.crash_extractor.extract_crashes`).

Walk-forward discipline:
    `fit_until(date)` uses only data strictly before `date`. The validation
    harness calls this at each fold boundary; live inference uses the
    most-recent fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.v6.config import AnomalyConfig, CONFIG
from src.v6.features.crash_extractor import extract_crashes, label_normal_days


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class AnomalyOutput:
    """Per-date anomaly engine output."""
    date: pd.Timestamp
    mahalanobis_score: float        # raw chi-squared statistic (>=0)
    mahalanobis_rank: float         # rank in [0,1] vs training set
    isoforest_rank: float           # rank in [0,1] vs training set
    ensemble_rank: float            # mean of the two ranks
    confidence: float               # 1 - |mahal_rank - iso_rank|
    alert: bool                     # ensemble_rank >= threshold


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class AnomalyEngine:
    """Mahalanobis + IsolationForest ensemble, leakage-safe fit."""

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.cfg = config or CONFIG.anomaly
        self.scaler_: Optional[StandardScaler] = None
        self.cov_: Optional[LedoitWolf] = None
        self.iso_: Optional[IsolationForest] = None
        self.feature_cols_: List[str] = []
        self.train_mahal_: Optional[np.ndarray] = None  # for rank-conversion
        self.train_iso_: Optional[np.ndarray] = None
        self.last_fit_date_: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, features: pd.DataFrame, prices: pd.Series) -> "AnomalyEngine":
        """Fit the engine on a labelled-as-normal training subset.

        Parameters
        ----------
        features : pd.DataFrame
            Indexed by date; contains the v6 feature columns. Must NOT
            contain any date that the caller wants to score later (walk-
            forward discipline is the caller's responsibility).
        prices : pd.Series
            Indexed by date; the equity index price used to identify
            "normal" days via crash extraction.
        """
        # 1. Identify normal-regime days using the crash extractor.
        episodes = extract_crashes(
            prices.loc[features.index],
            x_pct=self.cfg.normal_buffer_x_pct,
            min_duration_td=5,
        )
        normal_mask = label_normal_days(
            features.index, episodes, buffer_td=self.cfg.normal_buffer_td
        )

        # 2. Filter to features with HIGH non-null coverage in this training
        #    window. Median-imputing sparse features destroys their variance
        #    and creates a catastrophic scale shift on test — drop them
        #    instead. Threshold = 50% non-null, hard floor of 500 rows.
        X = features.loc[normal_mask].copy()
        n_train = len(X)
        if n_train < 500:
            raise RuntimeError(
                f"Anomaly fit needs ≥500 normal-regime rows, got {n_train}. "
                "Likely too short a training window or too aggressive crash buffer."
            )
        coverage = X.notna().mean()
        valid_cols = [c for c in X.columns if coverage[c] >= 0.5]
        if len(valid_cols) < 5:
            raise RuntimeError(
                f"Anomaly fit needs ≥5 features with ≥50% coverage; got {len(valid_cols)}."
            )
        X = X[valid_cols]
        # Impute remaining NaN with column median — this is now a small fraction.
        col_medians = X.median()
        X = X.fillna(col_medians)

        self.feature_cols_ = valid_cols

        # 3. Standardise and fit covariance with Ledoit-Wolf shrinkage.
        self.scaler_ = StandardScaler().fit(X.values)
        Xz = self.scaler_.transform(X.values)
        self.cov_ = LedoitWolf().fit(Xz)
        self.train_mahal_ = self.cov_.mahalanobis(Xz)

        # 4. Fit Isolation Forest on the same standardised matrix.
        self.iso_ = IsolationForest(
            n_estimators=self.cfg.iso_n_estimators,
            contamination=self.cfg.iso_contamination,
            random_state=42,
            n_jobs=-1,
        ).fit(Xz)
        # decision_function: higher = more normal. Invert for "anomaly".
        self.train_iso_ = -self.iso_.decision_function(Xz)

        self.last_fit_date_ = features.index.max()
        # Stash the median imputation values so score_one is consistent.
        self._train_medians_ = col_medians
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def _rank_of(self, score: float, ref: np.ndarray) -> float:
        if ref is None or len(ref) == 0:
            return float("nan")
        return float(np.searchsorted(np.sort(ref), score) / len(ref))

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score a full DataFrame of dates. Returns one row per input date."""
        if self.cov_ is None or self.iso_ is None or self.scaler_ is None:
            raise RuntimeError("Engine not fit. Call .fit() first.")
        X = features[self.feature_cols_].fillna(self._train_medians_)
        Xz = self.scaler_.transform(X.values)
        mahal = self.cov_.mahalanobis(Xz)
        iso = -self.iso_.decision_function(Xz)
        sorted_mahal = np.sort(self.train_mahal_)
        sorted_iso = np.sort(self.train_iso_)
        mahal_rank = np.searchsorted(sorted_mahal, mahal) / len(sorted_mahal)
        iso_rank = np.searchsorted(sorted_iso, iso) / len(sorted_iso)
        ensemble = 0.5 * (mahal_rank + iso_rank)
        confidence = 1.0 - np.abs(mahal_rank - iso_rank)
        alert = ensemble >= self.cfg.alert_threshold
        out = pd.DataFrame(
            {
                "mahalanobis_score": mahal,
                "mahalanobis_rank": mahal_rank,
                "isoforest_rank": iso_rank,
                "ensemble_rank": ensemble,
                "confidence": confidence,
                "alert": alert,
            },
            index=features.index,
        )
        return out

    def score_one(self, row: pd.Series) -> AnomalyOutput:
        """Convenience for a single date (returns AnomalyOutput)."""
        df = self.score(row.to_frame().T)
        r = df.iloc[0]
        return AnomalyOutput(
            date=df.index[0],
            mahalanobis_score=float(r["mahalanobis_score"]),
            mahalanobis_rank=float(r["mahalanobis_rank"]),
            isoforest_rank=float(r["isoforest_rank"]),
            ensemble_rank=float(r["ensemble_rank"]),
            confidence=float(r["confidence"]),
            alert=bool(r["alert"]),
        )
