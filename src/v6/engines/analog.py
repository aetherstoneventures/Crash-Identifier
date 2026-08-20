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
       c. confidence = how tight the retrieved neighbourhood is,
          measured as the k-th-neighbour radius ranked against the
          radii seen in training (1 = dense familiar region,
          0 = further from history than anything in-sample)
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


def _embargo_bounds(pool_dates: pd.DatetimeIndex, query_dates: pd.DatetimeIndex,
                    embargo_td: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pool index range to suppress around each query date.

    Returns (lo, hi) arrays such that pool positions in ``[lo_i, hi_i)`` are
    within the embargo window of query date i and must not be retrieved.

    WHY EMBARGO
    -----------
    Consecutive trading days are nearly identical in feature space, and
    their forward-outcome windows overlap almost entirely: for h = 63, the
    labels of day t and day t+1 share 62 of 63 days. Without an embargo the
    nearest neighbours of an in-sample date are simply its own neighbouring
    dates, which already know that date's outcome. The engine then looks
    excellent in-sample and reverts to noise out-of-sample — precisely the
    train/test gap measured in v6.1 development (posterior clearing 0.60 on
    20% of training days but 1% of out-of-sample days).

    Suppressing a window of +/- h trading days around the query removes the
    overlapping-label neighbours, so in-sample scores mean the same thing as
    out-of-sample ones. This is the purging-and-embargo discipline standard
    in financial cross-validation (Lopez de Prado, *Advances in Financial
    Machine Learning*, ch. 7).

    The window is expressed in calendar days (trading days x 1.6, rounded
    up) so it can be applied with a date search; erring long is safe.
    """
    if embargo_td <= 0:
        empty = np.zeros(len(query_dates), dtype=int)
        return empty, empty
    span = pd.Timedelta(days=int(np.ceil(embargo_td * 1.6)))
    lo = np.searchsorted(pool_dates.values, (query_dates - span).values, side="left")
    hi = np.searchsorted(pool_dates.values, (query_dates + span).values, side="right")
    return lo, hi


# A query date that is also in the training pool matches itself at distance
# ~0. Those self-matches would let a date "predict" its own realised outcome,
# so distances below this are pushed out of contention before neighbours are
# selected.
_SELF_MATCH_EPS = 1e-8


def _whitening_matrix(inv_cov: np.ndarray) -> np.ndarray:
    """Matrix W with W W^T = inv_cov, for Mahalanobis-as-Euclidean.

    Prefers a Cholesky factor; falls back to the symmetric eigendecomposition
    when the shrunk inverse covariance is not numerically positive definite.
    """
    try:
        return np.linalg.cholesky(inv_cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(inv_cov)
        return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))


def _cluster_support(d_k: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
    """How well-supported the analog set is, in [0, 1].

    `d_k` is the distance to the k-th nearest neighbour: the radius of the
    ball holding the retrieved analogs. Support is one minus that radius's
    percentile within the radii seen during training, so 1.0 means today's
    neighbourhood is tighter than almost any training day (a dense, familiar
    region of feature space) and 0.0 means we are further from history than
    we ever were in-sample — the design's "no good analogs found" state.

    Why not the design's literal d_1 / d_k
    --------------------------------------
    Both v6.0.0-alpha and the design document define analog confidence as a
    ratio between the nearest and k-th neighbour distances. That statistic
    collapses whenever any single neighbour is very close: when the pipeline
    scores a date that is itself in the training pool, the date matches
    itself at distance ~0, so d_1/d_k -> 0 and "confidence" reads zero on
    half of all days for a reason that has nothing to do with analog
    quality. Measuring the radius against its own training distribution is
    robust to that and answers the same question.
    """
    if reference is None or len(reference) == 0:
        return np.full(len(d_k), np.nan)
    ranks = np.searchsorted(reference, d_k) / len(reference)
    return np.clip(1.0 - ranks, 0.0, 1.0)


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
        self.whiten_: Optional[np.ndarray] = None
        self.train_Xw_: Optional[np.ndarray] = None
        self._train_sqnorm_: Optional[np.ndarray] = None
        self.train_dk_: Optional[np.ndarray] = None
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
        # Whitening transform W with W W^T = Sigma^-1, so that
        #     (a-b)^T Sigma^-1 (a-b) = || (a-b) W ||^2 .
        # Mahalanobis distance then reduces to a plain Euclidean distance in
        # whitened space, which BLAS computes for the whole matrix at once
        # instead of one einsum per query date.
        self.whiten_ = _whitening_matrix(self.inv_cov_)

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
        self.train_Xw_ = self.train_Xz_ @ self.whiten_
        # Cached ||row||^2 term of the expanded squared-distance identity.
        self._train_sqnorm_ = np.einsum("ij,ij->i", self.train_Xw_, self.train_Xw_)
        self.train_dates_ = features.index[eligible]
        for h in horizons:
            self.forward_maxdd_[h] = self.forward_maxdd_[h][eligible]

        # Reference distribution of the k-th-neighbour radius, measured on the
        # training pool itself with self-matches excluded. `_cluster_support`
        # ranks a query's radius against this, turning an unbounded distance
        # into an interpretable [0, 1] support score.
        self.train_dk_ = self._reference_radii()

        self.last_fit_date_ = features.index.max()
        return self

    def _reference_radii(self) -> np.ndarray:
        """Sorted k-th-neighbour distances across the training pool."""
        pool_n = self.train_Xw_.shape[0]
        if pool_n < 2:
            return np.array([])
        k = min(self.cfg.k_neighbors, pool_n - 1)
        radii = np.empty(pool_n)
        ref_lo, ref_hi = _embargo_bounds(
            self.train_dates_, self.train_dates_,
            max(SUPPORTED_HORIZON_DAYS) if self.cfg.embargo_horizons else 0,
        )
        batch = max(1, int(self.cfg.query_batch_size))
        for lo in range(0, pool_n, batch):
            hi = min(lo + batch, pool_n)
            D = self._distance_matrix(self.train_Xz_[lo:hi])
            # Same embargo as scoring, so the reference radii are drawn from
            # the same retrieval regime the query path will face.
            for r in range(hi - lo):
                D[r, ref_lo[lo + r]:ref_hi[lo + r]] = np.inf
            D[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
            part = np.partition(D, k - 1, axis=1)[:, :k]
            radii[lo:hi] = part.max(axis=1)
        return np.sort(radii)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def _distances(self, x: np.ndarray) -> np.ndarray:
        """Weighted-Mahalanobis distances to all training pool members."""
        return self._distance_matrix(x.reshape(1, -1))[0]

    def _distance_matrix(self, Xz: np.ndarray) -> np.ndarray:
        """Mahalanobis distances from each row of `Xz` to the whole pool.

        Uses the whitened-space identity ||a - b||^2 = ||a||^2 + ||b||^2 -
        2 a.b, so the work is one matrix product per batch rather than a
        quadratic form per query date.
        """
        Xw = Xz @ self.whiten_
        q_sq = np.einsum("ij,ij->i", Xw, Xw)[:, None]
        cross = Xw @ self.train_Xw_.T
        d2 = q_sq + self._train_sqnorm_[None, :] - 2.0 * cross
        return np.sqrt(np.clip(d2, 0.0, None))

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

        # Compute distances + nearest k, ignoring self-matches and any pool
        # date whose forward window overlaps this query's.
        d = self._distances(xz)
        d = np.where(d <= _SELF_MATCH_EPS, np.inf, d)
        if self.cfg.embargo_horizons and feature_vec.name is not None:
            e_lo, e_hi = _embargo_bounds(
                self.train_dates_, pd.DatetimeIndex([feature_vec.name]), horizon_td
            )
            d[e_lo[0]:e_hi[0]] = np.inf
        k = min(self.cfg.k_neighbors, int(np.isfinite(d).sum()))
        nn_idx = np.argpartition(d, k - 1)[:k]
        nn_idx = nn_idx[np.argsort(d[nn_idx])]  # sort the k

        nn_d = d[nn_idx]
        forward = self.forward_maxdd_[horizon_td][nn_idx]
        prob = float(np.mean(forward >= x_pct))

        # Confidence: tight cluster -> 1; diffuse -> 0.
        #
        # This is d_1 / d_k, per design doc §5.3 ("ratio of distance-to-1st-NN
        # over distance-to-50th-NN (tight cluster = high)"). When all k
        # neighbours sit at a similar distance the ratio approaches 1 and the
        # analog set is a coherent cluster; when the nearest neighbour is far
        # closer than the k-th, the ratio collapses toward 0 and we are
        # extrapolating from one lucky match.
        #
        # v6.0.0-alpha computed 1 - d_1/d_k, which is this measure inverted:
        # it reported LOW confidence exactly when the analog cluster was
        # tightest.
        confidence = float(_cluster_support(nn_d[-1:], self.train_dk_)[0])

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
        kk = min(k, self.train_Xw_.shape[0])
        hit = (fwd >= x_pct)
        # Batch the query dates: one (batch x pool) distance matrix at a time
        # keeps peak memory bounded while still doing the heavy lifting in
        # BLAS rather than in Python.
        batch = max(1, int(self.cfg.query_batch_size))
        # Suppress pool dates whose forward windows overlap the query's.
        embargo_td = horizon_td if self.cfg.embargo_horizons else 0
        emb_lo, emb_hi = _embargo_bounds(
            self.train_dates_, pd.DatetimeIndex(features.index), embargo_td
        )
        for lo in range(0, T, batch):
            hi = min(lo + batch, T)
            D = self._distance_matrix(Xz[lo:hi])
            # A scored date that also sits in the training pool would
            # otherwise retrieve itself and read off its own future.
            D[D <= _SELF_MATCH_EPS] = np.inf
            for r in range(hi - lo):
                D[r, emb_lo[lo + r]:emb_hi[lo + r]] = np.inf
            nn_idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
            rows = np.arange(hi - lo)[:, None]
            nn_d = np.sort(D[rows, nn_idx], axis=1)
            probs[lo:hi] = hit[nn_idx].mean(axis=1)
            confs[lo:hi] = _cluster_support(nn_d[:, -1], self.train_dk_)
        return pd.DataFrame({"prob": probs, "confidence": confs}, index=features.index)
