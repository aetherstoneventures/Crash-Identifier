"""Engine 2 — Regime-switching Hidden Markov Model.

Answers: "Which regime are we currently in, and what are the transition
odds toward a stress / crisis regime over horizon h?"

Method: a Gaussian HMM (Hamilton 1989; Kim & Nelson 1999; Ang & Bekaert
2002) with K ∈ {3, 4} latent states fit on a small, robust subset of the
v6 feature vector. We keep the feature set small (vol, drawdown, credit
spread, yield curve, NFCI/EPU) because HMMs become unstable with too
many emission dimensions and limited training data.

Outputs per date:
    - posterior P(state = k) for each state k
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
        # Posterior at each date given full sequence (use predict_proba which
        # uses the forward-backward algorithm — note: this DOES use future
        # observations within the supplied window. For strict walk-forward we
        # rely on the caller to pass only-already-observed `features`.)
        posterior = self.model_.predict_proba(Xz)
        hard_state = np.argmax(posterior, axis=1)
        stress_score = posterior[:, self.stress_states_].sum(axis=1)

        # h-step transition: posterior @ trans_mat^h
        trans = self.model_.transmat_
        trans_h = np.linalg.matrix_power(trans, h_steps)
        h_step_dist = posterior @ trans_h  # shape (T, K)
        h_step_crisis = h_step_dist[:, self.crisis_state_]

        cols = {f"posterior_state_{k}": posterior[:, k] for k in range(self.cfg.n_states)}
        cols.update({
            "hard_state": hard_state,
            "stress_score": stress_score,
            "h_step_crisis_prob": h_step_crisis,
            "crisis_state": self.crisis_state_,
            "calm_state": self.calm_state_,
        })
        return pd.DataFrame(cols, index=features.index)
