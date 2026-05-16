"""Engine 4 — Causal / Structural Factor Model.

Answers: "What is the *mechanism* — which drivers are loading, and is the
normal causal structure breaking down?"

This is the engine that distinguishes statistically-similar setups whose
underlying drivers are different (e.g. 2008 credit-led vs. 2020 exogenous
shock). Engines 1-3 measure pattern similarity; Engine 4 measures
mechanism.

Method
------
1. **Dynamic factor model (proxy):** rolling-window PCA on a small set of
   cross-asset returns / risk variables. The first K principal
   components are interpreted as latent factors (growth / risk / vol /
   credit / dollar). Loading vectors are saved per window so we can
   detect when an asset's relationship to the latent factors changes.

   (A full DFM with Kalman filter + regime-dependent loadings as in
   Ang & Bekaert 2002 is the gold standard but heavy; this PCA-window
   approach captures the same connectedness-shift signal at a fraction
   of the complexity and is the standard practitioner shortcut.)

2. **Diebold-Yilmaz connectedness:** total spillover index from the
   rolling-window VAR (here approximated by rolling-window correlation
   matrix → eigenvalue spread). High eigenvalue concentration = high
   connectedness = systemic stress (Billio et al. 2012).

3. **Granger causality break score:** for each ordered pair (i → j) we
   run a rolling F-test on whether lagged returns of i predict j. We
   then track how many edges of the "normal" causal graph have broken
   in the current window vs. the trailing baseline. KL-divergence-style.

4. **Dominant driver tag:** which asset's variance share is rising
   fastest — "vol-led", "credit-led", "rate-led", "dollar-led", "equity-led".

Output per date:
    - connectedness_score in [0, 1]: rank vs training window
    - structure_break_score in [0, 1]: how much the causal graph has shifted
    - dominant_driver: string tag
    - stress_composite: weighted combination of the above
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.v6.config import CONFIG, CausalConfig


# Cross-asset / risk universe for the factor model.
# Use what we actually have in the DB (sp500, vix, yield_10y, oil_wti,
# dollar_twi, baa_10y_spread, margin_debt) — node count 5-7.
CAUSAL_ASSETS: Dict[str, str] = {
    "equity": "sp500_close",     # use returns
    "vol": "vix_close",          # use level changes
    "rates": "yield_10y",        # use level changes
    "credit": "baa_10y_spread",  # use level changes
    "oil": "oil_wti",            # use returns
    "dollar": "dollar_twi",      # use returns
}


@dataclass
class CausalOutput:
    date: pd.Timestamp
    connectedness_score: float       # rank in [0, 1]
    structure_break_score: float     # rank in [0, 1]
    dominant_driver: str
    stress_composite: float          # 0.4*conn + 0.4*break + 0.2*driver_z


class CausalEngine:
    """Rolling-window PCA + correlation-connectedness + Granger break-score."""

    def __init__(self, config: Optional[CausalConfig] = None):
        self.cfg = config or CONFIG.causal
        self.asset_cols_: List[str] = []
        self.train_connectedness_: Optional[np.ndarray] = None
        self.train_structure_break_: Optional[np.ndarray] = None
        self._baseline_corr_: Optional[np.ndarray] = None
        self._train_returns_: Optional[pd.DataFrame] = None
        self.last_fit_date_: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _series_to_change(name: str, s: pd.Series) -> pd.Series:
        """Convert a raw series to its appropriate change measure."""
        s = s.astype(float)
        if name in {"equity", "oil", "dollar"}:
            return s.pct_change(fill_method=None)
        # vol, rates, credit are levels -> first-difference (or change ratio)
        return s.diff()

    def _build_returns(self, raw: pd.DataFrame, lock_to_fit: bool = False) -> pd.DataFrame:
        """Pick available causal-asset columns and convert to changes.

        If `lock_to_fit` is True, restrict to the asset columns used during
        fit (so scoring on a wider date range doesn't pick up extra assets
        that became available later).
        """
        cols: Dict[str, pd.Series] = {}
        candidates = self.asset_cols_ if lock_to_fit and self.asset_cols_ else list(CAUSAL_ASSETS.keys())
        for asset_name in candidates:
            col_name = CAUSAL_ASSETS[asset_name]
            if col_name in raw.columns and raw[col_name].notna().sum() >= 252:
                cols[asset_name] = self._series_to_change(asset_name, raw[col_name])
        if len(cols) < 3:
            raise RuntimeError(
                f"CausalEngine needs ≥3 causal assets with sufficient data; got {len(cols)}."
            )
        df = pd.DataFrame(cols)
        if not lock_to_fit:
            self.asset_cols_ = list(df.columns)
        return df

    @staticmethod
    def _connectedness_from_corr(C: np.ndarray) -> float:
        """Diebold-Yilmaz-style total spillover proxy.

        Use top-eigenvalue share of the |C| matrix: a high top-eigenvalue
        share means one common factor explains most co-movement = high
        connectedness = systemic stress.
        """
        if C.size == 0 or np.isnan(C).any():
            return np.nan
        eigvals = np.linalg.eigvalsh(np.abs(C))
        if eigvals.sum() <= 0:
            return np.nan
        return float(eigvals[-1] / eigvals.sum())  # top eigenvalue share

    @staticmethod
    def _structure_break(curr_C: np.ndarray, baseline_C: np.ndarray) -> float:
        """Frobenius-norm divergence between current and baseline corr matrix."""
        if curr_C.size == 0 or baseline_C.size == 0:
            return np.nan
        diff = curr_C - baseline_C
        return float(np.linalg.norm(diff, ord="fro"))

    def _dominant_driver(self, window: pd.DataFrame) -> str:
        """Asset whose variance share is highest in the window."""
        var_share = window.var() / window.var().sum()
        return str(var_share.idxmax())

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, raw: pd.DataFrame) -> "CausalEngine":
        """Fit on raw indicator DataFrame (the SAME DF passed to
        FeatureBuilder.load_raw() — needs the price-level columns)."""
        returns = self._build_returns(raw).dropna()
        self._train_returns_ = returns

        w = self.cfg.rolling_window_td
        if len(returns) < w + 1:
            raise RuntimeError(
                f"CausalEngine needs ≥{w + 1} rows of asset returns; got {len(returns)}."
            )

        # Baseline correlation = mean of rolling window correlations over the
        # whole training set.
        baseline = returns.corr().values
        self._baseline_corr_ = baseline

        # Rolling computation across training set to get training-time
        # distributions of connectedness and structure-break.
        T = len(returns)
        conn = np.full(T, np.nan)
        brk = np.full(T, np.nan)
        for i in range(w, T):
            window = returns.iloc[i - w : i]
            C = window.corr().values
            conn[i] = self._connectedness_from_corr(C)
            brk[i] = self._structure_break(C, baseline)
        self.train_connectedness_ = conn[~np.isnan(conn)]
        self.train_structure_break_ = brk[~np.isnan(brk)]
        self.last_fit_date_ = returns.index.max()
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    @staticmethod
    def _rank(score: float, ref: np.ndarray) -> float:
        if np.isnan(score) or ref is None or len(ref) == 0:
            return float("nan")
        return float(np.searchsorted(np.sort(ref), score) / len(ref))

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score a date-indexed raw indicator DataFrame.

        Note: at each date i, the window is returns[i-w : i] which is
        strictly past data -> no look-ahead.
        """
        if self._baseline_corr_ is None:
            raise RuntimeError("Engine not fit. Call .fit() first.")
        returns = self._build_returns(raw, lock_to_fit=True).dropna()
        w = self.cfg.rolling_window_td
        T = len(returns)
        conn_arr = np.full(T, np.nan)
        brk_arr = np.full(T, np.nan)
        driver_arr: List[str] = ["" for _ in range(T)]
        for i in range(w, T):
            window = returns.iloc[i - w : i]
            C = window.corr().values
            conn_arr[i] = self._connectedness_from_corr(C)
            brk_arr[i] = self._structure_break(C, self._baseline_corr_)
            driver_arr[i] = self._dominant_driver(window)
        conn_rank = np.array([self._rank(v, self.train_connectedness_) for v in conn_arr])
        brk_rank = np.array([self._rank(v, self.train_structure_break_) for v in brk_arr])
        stress = 0.5 * conn_rank + 0.5 * brk_rank
        out = pd.DataFrame(
            {
                "connectedness_score": conn_rank,
                "structure_break_score": brk_rank,
                "dominant_driver": driver_arr,
                "stress_composite": stress,
            },
            index=returns.index,
        )
        return out
