"""Engine 3 — Historical Analog Matcher.

Answers: "What happened to setups that looked like today?"

This is the engine that operationalises the user's core insight:

    "when we say crash, this should be a tunable parameter! ... For the
    data or 'crash' part, it is like a matching algorithm in the
    historical data based on several indicators of the price reduction."

Method
------
1. At each historical date t' (with at least h trading days of future
   data), store the standardised feature vector x(t').
2. For each historical t' also pre-compute the realised forward maxDD
   over each horizon h ∈ {21, 63, 126, 252}.
3. At inference time, given today's feature vector x(t) and a user-
   chosen (x_pct, h):
       a. find the k nearest historical neighbours by weighted-Mahalanobis
          distance (using the LedoitWolf-shrunk covariance of training data)
       b. empirical conditional CDF:
              P̂(maxDD ≥ x_pct | x(t)) = (1/k) Σ_i 1[maxDD_i(h) ≥ x_pct]
       c. confidence = ratio of distance-to-nearest neighbour over
          distance-to-k-th, in [0, 1] (1 = tight cluster, 0 = far)
       d. return top-N nearest analog dates with their forward paths.

CRITICAL: x_pct and h are inference-time parameters, NOT training-time.
The engine is fit ONCE on the full training feature matrix; the same fit
serves all (x_pct, h) queries.

Walk-forward discipline: `fit_until(date)` uses only data strictly
before `date`. Forward maxDD over horizon h is computed using prices at
indices [t', t' + h], so t' + h must be ≤ training_end_date — otherwise
that date is excluded from the neighbour pool to avoid look-ahead leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from src.v6.config import AnalogConfig, CONFIG, SUPPORTED_HORIZON_DAYS


@dataclass
class AnalogQuery:
    """A user query to the engine."""
    x_pct: float
    horizon_td: int


@dataclass
class AnalogResult:
    """Result of one analog query at one date."""
    date: pd.Timestamp
    query: AnalogQuery
    prob: float                          # P̂(maxDD ≥ x_pct in next h days)
    confidence: float                    # 1 = tight cluster, 0 = diffuse
    n_neighbours_used: int               # may be < k if pool exhausted
    analog_dates: List[pd.Timestamp]     # top-N analog dates
    analog_distances: List[float]
    analog_forward_maxdd: List[float]    # for the queried horizon


class AnalogEngine:
    """Weighted-Mahalanobis k-NN historical analog matcher."""

    def __init__(self, config: Optional[AnalogConfig] = None):
        self.cfg = config or CONFIG.analog
        self.scaler_: Optional[StandardScaler] = None
        self.cov_: Optional[LedoitWolf] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.feature_cols_: List[str] = []
        self._train_medians_: Optional[pd.Series] = None
        self.train_Xz_: Optional[np.ndarray] = None       # (T, d) standardised
        self.train_dates_: Optional[pd.DatetimeIndex] = None
        # Forward maxDD per horizon, shape (T,) per horizon
        self.forward_maxdd_: Dict[int, np.ndarray] = {}
        self.last_fit_date_: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, features: pd.DataFrame, prices: pd.Series,
            horizons: Tuple[int, ...] = SUPPORTED_HORIZON_DAYS) -> "AnalogEngine":
        """Fit the engine.

        Parameters
        ----------
        features : pd.DataFrame
            Date-indexed feature matrix. Caller is responsible for
            walk-forward discipline (don't include future data).
        prices : pd.Series
            Date-indexed equity index price. Must cover `features.index`.
        horizons : tuple of int
            Forward horizons (trading days) to pre-compute maxDD for.
        """
        # 1. Filter to features with sufficient coverage.
        coverage = features.notna().mean()
        cols = [c for c in features.columns if coverage[c] >= 0.5]
        if len(cols) < 5:
            raise RuntimeError(
                f"AnalogEngine needs ≥5 features with ≥50% coverage; got {len(cols)}."
            )
        self.feature_cols_ = cols
        X = features[cols].copy()
        self._train_medians_ = X.median()
        X = X.fillna(self._train_medians_)

        # 2. Standardise + LedoitWolf covariance.
        self.scaler_ = StandardScaler().fit(X.values)
        Xz = self.scaler_.transform(X.values)
        self.cov_ = LedoitWolf().fit(Xz)
        # Pre-invert for fast distance computation.
        self.inv_cov_ = np.linalg.pinv(self.cov_.covariance_)

        # 3. Compute forward maxDD for each horizon.
        px = prices.loc[features.index].astype(float).values
        T = len(px)
        for h in horizons:
            arr = np.full(T, np.nan)
            for i in range(T - h):
                window = px[i + 1 : i + 1 + h]
                if len(window) == 0:
                    continue
                # max drawdown from window's running max relative to px[i].
                running_max = np.maximum.accumulate(np.concatenate([[px[i]], window]))
                dd = (np.concatenate([[px[i]], window]) / running_max - 1.0) * 100.0
                arr[i] = float(-dd.min())  # positive percent
            self.forward_maxdd_[h] = arr

        # 4. Cache pool — only dates with non-NaN forward maxDD for the
        #    LONGEST horizon are eligible neighbours (ensures all horizons
        #    have a label).
        longest_h = max(horizons)
        eligible = ~np.isnan(self.forward_maxdd_[longest_h])
        self.train_Xz_ = Xz[eligible]
        self.train_dates_ = features.index[eligible]
        for h in horizons:
            self.forward_maxdd_[h] = self.forward_maxdd_[h][eligible]

        self.last_fit_date_ = features.index.max()
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def _distances(self, x: np.ndarray) -> np.ndarray:
        """Weighted-Mahalanobis distances to all training pool members."""
        diff = self.train_Xz_ - x  # (T, d)
        # quadratic form rowwise: d_i = sqrt(diff_i^T Σ⁻¹ diff_i)
        d2 = np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)
        return np.sqrt(np.clip(d2, 0, None))

    def query(self, feature_vec: pd.Series, x_pct: float, horizon_td: int,
              top_n_analogs: int = 10) -> AnalogResult:
        """Run one analog query for a single date.

        Parameters
        ----------
        feature_vec : pd.Series
            Feature vector for the query date (index = feature names,
            name = the query date).
        x_pct : float
            Drawdown threshold in percent (positive).
        horizon_td : int
            Forecast horizon in trading days.
        top_n_analogs : int
            How many analog dates to return.

        Returns
        -------
        AnalogResult
        """
        if self.train_Xz_ is None:
            raise RuntimeError("Engine not fit. Call .fit() first.")
        if horizon_td not in self.forward_maxdd_:
            raise ValueError(
                f"horizon_td={horizon_td} not in pre-computed horizons "
                f"{sorted(self.forward_maxdd_.keys())}."
            )

        # Standardise the query vector.
        x = feature_vec[self.feature_cols_].fillna(self._train_medians_).values
        xz = self.scaler_.transform(x.reshape(1, -1))[0]

        # Compute distances + nearest k.
        d = self._distances(xz)
        k = min(self.cfg.k_neighbors, len(d))
        nn_idx = np.argpartition(d, k - 1)[:k]
        nn_idx = nn_idx[np.argsort(d[nn_idx])]  # sort the k

        nn_d = d[nn_idx]
        forward = self.forward_maxdd_[horizon_td][nn_idx]
        prob = float(np.mean(forward >= x_pct))

        # Confidence: tight cluster -> 1; diffuse -> 0
        eps = self.cfg.min_distance_eps
        d_first = max(nn_d[0], eps)
        d_last = max(nn_d[-1], eps)
        confidence = float(np.clip(1.0 - d_first / d_last, 0.0, 1.0))

        n_top = min(top_n_analogs, k)
        analog_dates = [self.train_dates_[i] for i in nn_idx[:n_top]]
        analog_distances = [float(d[i]) for i in nn_idx[:n_top]]
        analog_fwd = [float(self.forward_maxdd_[horizon_td][i]) for i in nn_idx[:n_top]]

        return AnalogResult(
            date=feature_vec.name,
            query=AnalogQuery(x_pct=x_pct, horizon_td=horizon_td),
            prob=prob,
            confidence=confidence,
            n_neighbours_used=k,
            analog_dates=analog_dates,
            analog_distances=analog_distances,
            analog_forward_maxdd=analog_fwd,
        )

    def query_dataframe(self, features: pd.DataFrame, x_pct: float,
                        horizon_td: int) -> pd.DataFrame:
        """Vectorised scoring: one row per input date, only prob+confidence."""
        if self.train_Xz_ is None:
            raise RuntimeError("Engine not fit. Call .fit() first.")
        X = features[self.feature_cols_].fillna(self._train_medians_).values
        Xz = self.scaler_.transform(X)
        T = len(Xz)
        k = self.cfg.k_neighbors
        probs = np.full(T, np.nan)
        confs = np.full(T, np.nan)
        if self.train_Xz_.shape[0] == 0:
            return pd.DataFrame({"prob": probs, "confidence": confs}, index=features.index)
        fwd = self.forward_maxdd_[horizon_td]
        for i in range(T):
            diff = self.train_Xz_ - Xz[i]
            d2 = np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)
            d = np.sqrt(np.clip(d2, 0, None))
            kk = min(k, len(d))
            nn_idx = np.argpartition(d, kk - 1)[:kk]
            nn_d = np.sort(d[nn_idx])
            probs[i] = float(np.mean(fwd[nn_idx] >= x_pct))
            eps = self.cfg.min_distance_eps
            confs[i] = float(np.clip(1.0 - max(nn_d[0], eps) / max(nn_d[-1], eps), 0.0, 1.0))
        return pd.DataFrame({"prob": probs, "confidence": confs}, index=features.index)
