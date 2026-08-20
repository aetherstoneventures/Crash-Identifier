"""Engine 2 — Regime-switching Hidden Markov Model.

Answers: "Which regime are we currently in, and what are the transition
odds toward a stress / crisis regime over horizon h?"

Method: a Gaussian HMM (Hamilton 1989; Kim & Nelson 1999; Ang & Bekaert
2002) with K ∈ {3, 4} latent states fit on a small, robust subset of the
v6 feature vector. We keep the feature set small (vol, drawdown, credit
spread, yield curve, NFCI/EPU) because HMMs become unstable with too
many emission dimensions and limited training data.

Outputs per date (all strictly causal — filtered, never smoothed):
    - posterior P(state = k | observations up to t) for each state k
    - hard-assigned state (argmax of posterior)
    - h-step transition probability to the "crisis" state (highest-vol)
    - regime-stress score in [0, 1] = P(state ∈ {stress, crisis})

The "crisis" state is identified post-hoc as the state with the highest
mean realised-vol z-score in the training data; the "calm" state as the
lowest.

Walk-forward discipline: `fit_until(date)` uses only data strictly
before `date`. The validation harness re-fits each fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.v6.config import CONFIG, RegimeConfig


# Reduced HMM feature set — robust to data gaps.
HMM_FEATURES: List[str] = [
    "rv_21_z", "rv_63_z",
    "dd_from_252h",
    "vix_z",
    "hy_spread_z",
    "yc_10y_3m",
    "ma200_dist",
]


@dataclass
class RegimeOutput:
    """Per-date regime engine output."""
    date: pd.Timestamp
    hard_state: int
    posterior: np.ndarray            # length K
    stress_score: float              # P(state ∈ {stress, crisis})
    h_step_crisis_prob: float        # transition-matrix prediction
    crisis_state: int
    calm_state: int


class RegimeEngine:
    """Gaussian HMM on a reduced macro/vol feature subset."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.cfg = config or CONFIG.regime
        self.model_ = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_cols_: List[str] = []
        self.crisis_state_: int = -1
        self.calm_state_: int = -1
        self.stress_states_: List[int] = []
        self.last_fit_date_: Optional[pd.Timestamp] = None
        self._train_medians_: Optional[pd.Series] = None

    def fit(self, features: pd.DataFrame) -> "RegimeEngine":
        from hmmlearn.hmm import GaussianHMM

        # Filter to available + sufficiently-covered features.
        cols = [c for c in HMM_FEATURES if c in features.columns]
        cov = features[cols].notna().mean()
        cols = [c for c in cols if cov[c] >= 0.5]
        if len(cols) < 3:
            raise RuntimeError(
                f"HMM needs ≥3 features with ≥50% coverage; got {len(cols)}."
            )
        self.feature_cols_ = cols
        X = features[cols].copy()
        self._train_medians_ = X.median()
        X = X.fillna(self._train_medians_)

        self.scaler_ = StandardScaler().fit(X.values)
        Xz = self.scaler_.transform(X.values)

        self.model_ = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type="full",
            n_iter=self.cfg.em_iterations,
            random_state=self.cfg.random_state,
            tol=1e-3,
        ).fit(Xz)

        # Identify crisis / calm states by mean vol-z in each state's emission
        # parameter (mean of standardised rv_21_z).
        if "rv_21_z" in cols:
            vol_idx = cols.index("rv_21_z")
        else:
            vol_idx = 0  # fallback: use first feature

        state_means = self.model_.means_[:, vol_idx]
        order = np.argsort(state_means)  # ascending vol-mean
        self.calm_state_ = int(order[0])
        self.crisis_state_ = int(order[-1])
        # "stress" = top half of states by vol-mean (incl. crisis).
        n = self.cfg.n_states
        self.stress_states_ = [int(s) for s in order[n // 2 :]]

        self.last_fit_date_ = features.index.max()
        return self

    def _score_matrix(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Engine not fit. Call .fit() first.")
        X = features[self.feature_cols_].fillna(self._train_medians_)
        Xz = self.scaler_.transform(X.values)
        return Xz

    def _filtered_posterior(self, Xz: np.ndarray) -> np.ndarray:
        """Forward-only (filtered) state probabilities P(state_t | x_1..x_t).

        Implements the HMM forward recursion in log space:

            log a_1(k) = log pi_k + log b_k(x_1)
            log a_t(k) = logsumexp_j [log a_{t-1}(j) + log A_jk] + log b_k(x_t)

        and normalises each row to a distribution. Unlike
        `hmmlearn.predict_proba` (forward-backward smoothing), this
        conditions only on observations up to and including t, which is the
        only causally admissible quantity for a forecasting model.

        Implemented directly from the fitted parameters rather than through
        hmmlearn internals so it does not depend on private APIs.
        """
        from scipy.special import logsumexp
        from scipy.stats import multivariate_normal

        model = self.model_
        n_states = model.n_components
        T = Xz.shape[0]

        # Emission log-densities: (T, K)
        log_b = np.empty((T, n_states))
        covars = model.covars_
        for k in range(n_states):
            cov = covars[k]
            if cov.ndim == 1:          # 'diag' covariance
                cov = np.diag(cov)
            log_b[:, k] = multivariate_normal.logpdf(
                Xz, mean=model.means_[k], cov=cov, allow_singular=True
            )

        with np.errstate(divide="ignore"):
            log_pi = np.log(model.startprob_)
            log_A = np.log(model.transmat_)

        log_alpha = np.empty((T, n_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            # (K_prev, K_next) broadcast, then reduce over previous state.
            log_alpha[t] = logsumexp(
                log_alpha[t - 1][:, None] + log_A, axis=0
            ) + log_b[t]

        # Normalise each row into a probability distribution.
        log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha - log_norm)

    def score(self, features: pd.DataFrame, h_steps: int = 21) -> pd.DataFrame:
        """Score all dates in `features`.

        Parameters
        ----------
        features : pd.DataFrame
            Indexed by date.
        h_steps : int
            Steps ahead for transition-matrix prediction.

        Returns
        -------
        pd.DataFrame indexed by date with columns:
            posterior_state_0..K-1, hard_state, stress_score,
            h_step_crisis_prob, crisis_state, calm_state.
        """
        Xz = self._score_matrix(features)

        # FILTERED (forward-only) posterior — see `_filtered_posterior`.
        # v6.0.0-alpha called `predict_proba`, which runs forward-BACKWARD
        # smoothing: the state probability at date t was conditioned on
        # observations AFTER t. Since the pipeline scores the full history in
        # one call, every historical date was being told the future. This is
        # the strict-causality fix.
        posterior = self._filtered_posterior(Xz)
        hard_state = np.argmax(posterior, axis=1)
        stress_score = posterior[:, self.stress_states_].sum(axis=1)

        # h-step transition: filtered posterior @ trans_mat^h
        trans = self.model_.transmat_
        trans_h = np.linalg.matrix_power(trans, h_steps)
        h_step_dist = posterior @ trans_h  # shape (T, K)
        h_step_crisis = h_step_dist[:, self.crisis_state_]
        # Probability of being in ANY stress state h steps ahead. This is the
        # horizon-aware quantity the aggregator consumes: it answers "where is
        # this regime heading over the query horizon", not merely "where is it
        # now".
        h_step_stress = h_step_dist[:, self.stress_states_].sum(axis=1)

        cols = {f"posterior_state_{k}": posterior[:, k] for k in range(self.cfg.n_states)}
        cols.update({
            "hard_state": hard_state,
            "stress_score": stress_score,
            "h_step_crisis_prob": h_step_crisis,
            "h_step_stress_prob": h_step_stress,
            "crisis_state": self.crisis_state_,
            "calm_state": self.calm_state_,
        })
        return pd.DataFrame(cols, index=features.index)
