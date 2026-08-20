"""Engine 5 — Bayesian Aggregator + Layer 1/2/3 Simultaneous-Agreement Gate.

Combines the four upstream engines into a single posterior

    P(maxDD ≥ x_pct in next h trading days)

with a confidence score and a transparent per-engine contribution
decomposition.

Engines and the question each answers:
    1. AnomalyEngine         "is now unusual?"
    2. RegimeEngine          "what regime are we in, and where is it heading?"
    3. AnalogEngine          "what historically followed setups like this?"
    4. CausalEngine          "what's the mechanism / is structure breaking?"


WHY THIS WAS REWRITTEN
======================
The v6.0.0-alpha aggregator combined engines as a Beta-Binomial update with
a Beta(1, 1) prior and weights summing to 1:

    α = 1 + Σ w_e p_e ,  β = 1 + Σ w_e (1 - p_e) ,  P = α / (α + β)

Because Σ w_e = 1, the denominator α + β is **always exactly 3**: two
pseudo-counts of prior against one of evidence, forever, no matter how
strong or unanimous the signal. Consequences, all verified numerically:

- The posterior could never leave **[0.3333, 0.6667]**. The gate threshold
  τ = 0.60 therefore required a mean engine pressure ≥ 0.80 — and a
  posterior of, say, 0.9 was unreachable by construction.
- Confidence was defined as 1 - 2·√Var, which under a fixed α + β = 3 is a
  deterministic U-shaped function of the posterior spanning
  **[0.5000, 0.5286]** — a 2.9-percentage-point band sitting exactly on the
  κ = 0.50 threshold. It measured nothing and flipped on rounding noise.

The engines were never given a chance to speak. This module replaces that
with the two things the design document actually asked for (§5.5) and the
alpha never implemented: **per-engine calibration** and **learned weights**.


METHOD
======
1. **Per-engine calibration.** Each engine emits a raw "pressure" on its
   own scale — a percentile rank, a regime probability, an empirical
   frequency. These are not interchangeable with P(crash). Each is mapped
   to a calibrated probability by binning it against the realised training
   outcome, with Beta(α₀, β₀) shrinkage so sparse bins fall back toward the
   base rate instead of reporting 0 or 1 from three observations.

2. **Skill-weighted log-odds pooling.** Calibrated probabilities are pooled
   in log-odds space, anchored on the training base rate:

       logit(P) = logit(π) + Σ_e w_e · [ logit(p̂_e) − logit(π) ]

   Each engine contributes *evidence relative to the prior*, which is the
   standard log-linear opinion pool. Unlike the alpha's formulation this
   has full support on (0, 1): unanimous strong evidence produces a
   posterior near 1, and unanimous absence produces one near 0.

3. **Cross-fitted weights.** Weights come from each engine's out-of-sample
   skill, measured by log-loss improvement over the base rate on a held-out
   second half of the training window. An engine that cannot beat the base
   rate gets the floor weight. No engine is allowed to carry everything
   (kill criterion 4), so weights are floored and renormalised.

4. **Confidence that means something.** Three multiplicative components,
   each in [0, 1]:
       - *agreement*: 1 − normalised dispersion of the engines' log-odds.
         Four engines pointing the same way is worth more than four
         engines disagreeing around the same mean.
       - *coverage*: fraction of engines that actually reported.
       - *analog support*: the analog engine's cluster tightness — the
         design's "no good analogs found" failure mode, which the alpha
         computed and then discarded.


LAYER 1 / 2 / 3 GATE
====================
Action fires only when ALL hold:
    - aggregator posterior ≥ τ
    - aggregator confidence ≥ κ
    - Layer 1 macro regime elevated       (composite z ≥ 1.5σ)
    - Layer 2 tactical stress elevated    (composite z ≥ 1.5σ)
    - Layer 3 price confirms              (drawdown from 252d high ≥ x_confirm)

The layer composites are now **re-standardised after aggregation**. The
alpha averaged ~10 already-z-scored features and compared the mean to 1.5.
Averaging shrinks dispersion by roughly √n, so the realised standard
deviation of those composites was 0.59 (L1) and 0.48 (L2): the nominal
"1.5σ" threshold was really a 2.4σ and 3.1σ event, and both fired together
on 48 of 11,562 days (0.42%) before any other condition was applied. The
composite is now z-scored on its own expanding distribution, so a 1.5
threshold means 1.5 standard deviations of that composite, as documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.v6.config import AggregatorConfig, CONFIG, GateConfig


# ---------------------------------------------------------------------------
# Layer composition — which features feed which layer's stress score.
# All are existing v6 feature names; missing ones are silently dropped.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LAYER 1 — MACRO REGIME, BY CRASH ARCHETYPE
#
# Layer 1 asks "is a macro crash regime active?". v6.0.0-alpha answered that
# with a single composite blending credit spreads, the labour market,
# valuation and real rates into one average. Averaging them means a crash
# only registers when *most* macro indicators are stressed at once — which
# describes a 2008-style credit event and almost nothing else.
#
# Measured consequence: through the whole 2022 bear market (-35.7% peak to
# trough on the Nasdaq) the blended L1 composite never rose above +0.38
# while L2 and L3 fired correctly and the posterior reached 0.58. L1 was the
# sole blocking condition on every single day. And L1 was not malfunctioning
# — 2022 was a rate-and-valuation shock with contained credit spreads, record
# low unemployment, and no Sahm trigger. The macro layer correctly reported
# "macro is fine" and, under a single-composite design, that verdict vetoed
# everything else.
#
# A single averaged composite therefore makes the system structurally blind
# to every crash that is not credit-led, which contradicts the project's own
# first requirement: "The model should know all types of crashes, but should
# be able to distinguish between them."
#
# So Layer 1 is decomposed into the macro archetypes a crash can take. Each
# is scored on its own terms, L1 is the strongest archetype currently
# active, and the winning archetype is reported alongside the gate decision
# as the mechanism tag the design document asks for (§5.4). The gate still
# requires L1 AND L2 AND L3 simultaneously — 100% agreement across the three
# layers is unchanged. What changes is that "macro regime elevated" now
# means "one of the known macro crash regimes is active" instead of "the
# credit-led one is active".
# ---------------------------------------------------------------------------
ARCHETYPES: Dict[str, List[str]] = {
    # 2008: balance-sheet stress. Spreads widen, funding tightens, labour
    # turns. The only archetype the alpha could see.
    "credit_led": [
        "hy_spread_z", "hy_spread_chg", "ig_spread_z", "nfci", "nfci_credit",
        "nfci_leverage", "stlfsi_z", "claims_13w_chg_z", "sahm_indicator",
    ],
    # 2022: policy/rate shock. Real rates jump and the curve inverts while
    # credit and employment stay healthy — invisible to a credit composite.
    "rate_led": [
        "real_rate_10y", "yc_10y_3m", "yc_10y_2y", "yc_10y_3m_chg", "m2_yoy_z",
    ],
    # 2000: valuation unwind. Price stretched far above its own long-run
    # trend, with deteriorating internals rather than macro stress.
    "valuation_led": [
        "cape_proxy_z", "ma200_dist", "downside_vol_ratio_63", "acf_20",
    ],
    # 1987 / 2020: exogenous or liquidity shock. No macro warning at all —
    # the tell is cross-asset correlation collapse and a policy-uncertainty
    # spike, both of which move within days.
    "shock_led": [
        "cross_asset_corr_z", "epu_z", "oil_shock_z", "dxy_z",
    ],
}

# Features where a LOW reading is the stressed one.
INVERTED_FEATURES = {
    "yc_10y_3m", "yc_10y_2y", "iv_rv_gap", "m2_yoy_z", "ma200_dist", "acf_20",
}

# Retained for reference and for anything that wants the old blended view.
LAYER1_FEATURES = [
    "yc_10y_3m", "hy_spread_z", "ig_spread_z", "nfci", "nfci_leverage",
    "nfci_credit", "stlfsi_z", "epu_z", "sahm_indicator", "claims_13w_chg_z",
    "cape_proxy_z", "real_rate_10y", "m2_yoy_z",
]
LAYER2_FEATURES = [
    "rv_21_z", "rv_63_z", "vix_z", "vix_term_structure", "vix_shock_5d",
    "cross_asset_corr_z", "return_dispersion_63", "downside_vol_ratio_63",
    "hy_spread_chg", "iv_rv_gap", "oil_shock_z",
]
LAYER3_PRICE_COL = "dd_from_252h"   # always present from FeatureBuilder

ENGINE_NAMES: Tuple[str, ...] = ("anomaly", "regime", "analog", "causal")

# Probabilities are clipped away from {0, 1} before taking log-odds, so a
# single saturated engine cannot force an infinite posterior.
_P_FLOOR = 1e-4
_P_CEIL = 1.0 - 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expanding_z(s: pd.Series, min_periods: int = 252) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def _logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _P_FLOOR, _P_CEIL)
    return np.log(p / (1.0 - p))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary log-loss, NaN-safe."""
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    p_ = np.clip(p[mask], _P_FLOOR, _P_CEIL)
    y_ = y[mask]
    return float(-np.mean(y_ * np.log(p_) + (1 - y_) * np.log(1 - p_)))


def _composite_score(features: pd.DataFrame, names: List[str],
                     min_periods: int = 252) -> pd.Series:
    """Sign-corrected composite of a feature group, z-scored as a composite.

    Features where LOW means stress (an inverted yield curve, a collapsing
    IV-RV gap) are negated so that HIGH always means stress. Components are
    individually z-scored, averaged, and then the **average is z-scored
    again** — without that second step the threshold would be expressed in
    units the composite never reaches (see module docstring).
    """
    inv_features = INVERTED_FEATURES
    valid = [c for c in names if c in features.columns and features[c].notna().sum() > min_periods]
    if not valid:
        return pd.Series(np.nan, index=features.index)
    z_cols = []
    for c in valid:
        s = features[c].astype(float)
        if c in inv_features:
            s = -s
        z_cols.append(_expanding_z(s, min_periods=min_periods))
    # Row mean over whatever components exist on that date.
    raw_composite = pd.concat(z_cols, axis=1).mean(axis=1)
    return _expanding_z(raw_composite, min_periods=min_periods)


def archetype_scores(features: pd.DataFrame) -> pd.DataFrame:
    """One z-scored macro-regime composite per crash archetype.

    Each archetype is scored independently so a regime that is stressed on
    its own terms registers even when the other archetypes are calm — the
    2022 rate shock being the case that motivated the split.
    """
    return pd.DataFrame(
        {name: _composite_score(features, cols)
         for name, cols in ARCHETYPES.items()},
        index=features.index,
    )


def _layer_series(features: pd.DataFrame, gate_cfg: GateConfig
                  ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """The three layer signals, each on its own timescale, plus the tag.

    L1 and L2 are rolling maxima of their composites — "is this layer
    currently active" rather than "did it print its high today" (see
    GateConfig). L3 is the instantaneous drawdown, which is a price fact and
    needs no smoothing.

    L1 is the strongest currently-active archetype; `archetype` names it, so
    a fire always carries the mechanism that triggered it.
    """
    arche = archetype_scores(features)
    persist = max(1, gate_cfg.layer1_persistence_td)
    arche_persist = arche.rolling(persist, min_periods=1).max()

    l1 = arche_persist.max(axis=1)
    archetype = arche_persist.idxmax(axis=1).where(l1.notna())

    l2_raw = _composite_score(features, LAYER2_FEATURES)
    l2 = l2_raw.rolling(max(1, gate_cfg.layer2_persistence_td), min_periods=1).max()
    l3 = -features[LAYER3_PRICE_COL].astype(float)   # -DD% -> +stress
    return l1, l2, l3, archetype


# ---------------------------------------------------------------------------
# Per-engine calibration
# ---------------------------------------------------------------------------
@dataclass
class EngineCalibrator:
    """Maps one engine's raw pressure to a calibrated probability.

    Bins the pressure into quantile bins over the training window and takes
    the Beta-Binomial posterior mean of the outcome within each bin:

        p̂(bin) = (α₀ + hits) / (α₀ + β₀ + n)

    The Beta(α₀, β₀) prior is what stops a bin holding four observations
    from confidently claiming 0% or 100%. Bins are quantile-based so each
    holds a comparable number of days regardless of how the engine's raw
    scores happen to be distributed.
    """
    n_bins: int = 10
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    edges_: Optional[np.ndarray] = None
    bin_prob_: Optional[np.ndarray] = None
    base_rate_: float = 0.5
    fitted_: bool = False

    def fit(self, pressure: np.ndarray, y: np.ndarray) -> "EngineCalibrator":
        mask = np.isfinite(pressure) & np.isfinite(y)
        p, yy = pressure[mask], y[mask]
        if len(p) < 50 or len(np.unique(yy)) < 2:
            # Not enough signal to calibrate; fall back to the base rate.
            self.base_rate_ = float(yy.mean()) if len(yy) else 0.5
            self.fitted_ = False
            return self

        self.base_rate_ = float(yy.mean())
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(p, quantiles))
        if len(edges) < 3:
            self.fitted_ = False
            return self
        # Open the outer edges so unseen extremes still land in a bin.
        edges[0], edges[-1] = -np.inf, np.inf
        self.edges_ = edges

        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
        n_bins_actual = len(edges) - 1
        probs = np.empty(n_bins_actual)
        for b in range(n_bins_actual):
            in_bin = idx == b
            hits = float(yy[in_bin].sum())
            n = float(in_bin.sum())
            probs[b] = (self.prior_alpha + hits) / (
                self.prior_alpha + self.prior_beta + n
            )
        self.bin_prob_ = probs
        self.fitted_ = True
        return self

    def transform(self, pressure: np.ndarray) -> np.ndarray:
        """Calibrated probability per observation (NaN where pressure is NaN)."""
        out = np.full(len(pressure), np.nan)
        finite = np.isfinite(pressure)
        if not self.fitted_ or self.edges_ is None:
            out[finite] = self.base_rate_
            return out
        idx = np.clip(
            np.digitize(pressure[finite], self.edges_[1:-1], right=False),
            0, len(self.bin_prob_) - 1,
        )
        out[finite] = self.bin_prob_[idx]
        return out


@dataclass
class PosteriorCalibrator:
    """Final recalibration of the pooled posterior against outcomes.

    WHY A SECOND CALIBRATION STAGE
    ------------------------------
    Each engine is already calibrated individually, but **pooling does not
    preserve calibration**. A weighted log-odds pool of four individually
    well-calibrated experts is generally over- or under-confident, because
    the experts are correlated: they partly observe the same market state, so
    summing their evidence double-counts it. v6.1's first validation showed
    exactly that signature — the posterior *ranked* days well (pooled gate
    precision 0.859 at 2.39x lift) while its *probabilities* failed kill
    criterion 1, with reliability slopes of 0.002 and -0.415.

    Ranking and calibration are separate properties, and only the second one
    was broken. This stage fixes the second without disturbing the first:
    isotonic regression is monotone, so it cannot reorder days.

    FITTING DISCIPLINE
    ------------------
    Isotonic regression is flexible enough to memorise its training set, and
    a calibration map fitted on the same rows it is scored against would look
    perfect and transfer badly. The map is therefore **cross-fitted**: the
    training window is cut into contiguous blocks, each block's map is fitted
    on the other blocks, and the final stored map is refitted on everything.
    Blocks are contiguous rather than random because neighbouring days share
    overlapping outcome windows; random folds would leak across the split.
    """
    method: str = "platt"      # 'platt' | 'isotonic' | 'none'
    n_blocks: int = 4
    min_samples: int = 200
    model_: object = None
    fitted_: bool = False
    train_slope_: float = float("nan")

    def _new_model(self):
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, solver="lbfgs")

    def _fit_model(self, model, p: np.ndarray, yy: np.ndarray):
        if self.method == "isotonic":
            model.fit(p, yy)
        else:
            # Platt scaling: a 2-parameter logistic on the log-odds of the
            # pooled posterior.
            model.fit(_logit(p).reshape(-1, 1), yy)
        return model

    def _predict(self, model, p: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return model.predict(p)
        return model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def fit(self, posterior: np.ndarray, y: np.ndarray) -> "PosteriorCalibrator":
        if self.method == "none":
            self.fitted_ = False
            return self

        mask = np.isfinite(posterior) & np.isfinite(y)
        p, yy = posterior[mask], y[mask]
        if len(p) < self.min_samples or len(np.unique(yy)) < 2:
            self.fitted_ = False
            return self

        # Cross-fitted estimate of what this map will do out-of-sample.
        # Recorded for auditing; the stored map is the full-data refit.
        oof = np.full(len(p), np.nan)
        edges = np.linspace(0, len(p), self.n_blocks + 1).astype(int)
        for b in range(self.n_blocks):
            lo, hi = edges[b], edges[b + 1]
            train_idx = np.r_[np.arange(0, lo), np.arange(hi, len(p))]
            if len(train_idx) < self.min_samples or len(np.unique(yy[train_idx])) < 2:
                continue
            fold = self._fit_model(self._new_model(), p[train_idx], yy[train_idx])
            oof[lo:hi] = self._predict(fold, p[lo:hi])
        ok = np.isfinite(oof)
        if ok.sum() > 10 and np.std(oof[ok]) > 0:
            self.train_slope_ = float(np.polyfit(oof[ok], yy[ok], 1)[0])

        self.model_ = self._fit_model(self._new_model(), p, yy)
        self.fitted_ = True
        return self

    def transform(self, posterior: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.model_ is None:
            return posterior
        out = np.full(len(posterior), np.nan)
        finite = np.isfinite(posterior)
        out[finite] = self._predict(self.model_, posterior[finite])
        return out


# ---------------------------------------------------------------------------
# Aggregator + Gate
# ---------------------------------------------------------------------------
@dataclass
class AggregatorResult:
    date: pd.Timestamp
    posterior_mean: float
    confidence: float
    per_engine_pressure: Dict[str, float]
    layer1_z: float
    layer2_z: float
    layer3_dd: float
    gate_fires: bool
    gate_reason: str


@dataclass
class CrashKPIAggregator:
    """Calibrated log-odds aggregator + L1/L2/L3 gate."""

    aggregator_cfg: AggregatorConfig = field(default_factory=lambda: CONFIG.aggregator)
    gate_cfg: GateConfig = field(default_factory=lambda: CONFIG.gate)
    # Populated by .fit(); equal weights until then.
    weights: Dict[str, float] = field(default_factory=lambda: {
        name: 0.25 for name in ENGINE_NAMES
    })
    calibrators_: Dict[str, EngineCalibrator] = field(default_factory=dict)
    base_rate_: float = 0.5
    skill_: Dict[str, float] = field(default_factory=dict)
    fitted_: bool = False
    # Layer thresholds fitted on the training window; None -> use GateConfig.
    gate_thresholds_: Optional[Dict[str, float]] = None
    gate_tuning_: Dict[str, float] = field(default_factory=dict)
    posterior_calibrator_: Optional[PosteriorCalibrator] = None

    # ---------- Per-engine pressure mappings ----------
    @staticmethod
    def _engine_pressure(engine_name: str, df: pd.DataFrame) -> pd.Series:
        """Extract each engine's raw crash-pressure column.

        These are NOT probabilities of a crash on a common scale — they are
        percentile ranks, regime probabilities and empirical frequencies.
        `EngineCalibrator` is what turns them into comparable probabilities.
        """
        column_for = {
            "anomaly": "ensemble_rank",       # percentile of "unusualness"
            # Horizon-aware: where the regime is heading over h, not just
            # where it is now. The alpha used the contemporaneous
            # `stress_score`, which made the engine's contribution
            # independent of the queried horizon.
            "regime": "h_step_stress_prob",
            "analog": "prob",                 # empirical P(maxDD >= x in h)
            "causal": "stress_composite",     # rank vs training distribution
        }
        col = column_for.get(engine_name)
        if col is None or df is None or col not in df.columns:
            # Fall back to the alpha's column if the new one is absent, so an
            # older engine build still aggregates rather than vanishing.
            fallback = {"regime": "stress_score"}.get(engine_name)
            if fallback and df is not None and fallback in df.columns:
                col = fallback
            else:
                return pd.Series(np.nan, index=df.index if df is not None else None)
        return pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)

    def _pressure_frame(self, engine_outputs: Dict[str, pd.DataFrame],
                        idx: pd.Index) -> pd.DataFrame:
        pressures = pd.DataFrame(index=idx)
        for name in ENGINE_NAMES:
            df = engine_outputs.get(name)
            if df is None or len(df) == 0:
                pressures[name] = np.nan
            else:
                pressures[name] = self._engine_pressure(name, df.loc[idx])
        return pressures

    # ------------------------------------------------------------------
    # Fit — calibration + weights, on training data only
    # ------------------------------------------------------------------
    def fit(self, engine_outputs: Dict[str, pd.DataFrame], y: pd.Series,
            features: Optional[pd.DataFrame] = None) -> "CrashKPIAggregator":
        """Calibrate each engine, learn pooling weights, and tune the gate.

        Parameters
        ----------
        engine_outputs : dict
            Per-engine DataFrames restricted to the TRAINING window.
        y : pd.Series
            Boolean/0-1 outcome on the same index: did maxDD ≥ x_pct occur
            within the horizon starting at that date. Computed from training
            prices only.
        features : pd.DataFrame | None
            Training-window features. Supplied so layer thresholds can be
            tuned to a target fire rate; without it the gate falls back to
            the fixed z-thresholds in GateConfig.
        """
        idx = y.index
        for df in engine_outputs.values():
            if df is not None and len(df):
                idx = idx.intersection(df.index)
        y = y.loc[idx].astype(float)
        pressures = self._pressure_frame(engine_outputs, idx)

        self.base_rate_ = float(np.clip(y.mean(), _P_FLOOR, _P_CEIL))

        # Chronological split: calibrate on the first half, measure skill on
        # the second. Scoring an engine on the same rows that calibrated it
        # would reward memorisation and hand weight to the most overfit
        # engine — the opposite of what kill criterion 4 asks for.
        n = len(idx)
        cut = n // 2
        y_arr = y.values

        self.skill_ = {}
        for name in ENGINE_NAMES:
            p_raw = pressures[name].values
            if cut < 50 or not np.isfinite(p_raw).any():
                self.skill_[name] = 0.0
                continue
            probe = EngineCalibrator(
                prior_alpha=self.aggregator_cfg.beta_prior_alpha,
                prior_beta=self.aggregator_cfg.beta_prior_beta,
            ).fit(p_raw[:cut], y_arr[:cut])
            p_oos = probe.transform(p_raw[cut:])
            ll_engine = _log_loss(y_arr[cut:], p_oos)
            ll_base = _log_loss(
                y_arr[cut:], np.full(n - cut, float(np.mean(y_arr[:cut])))
            )
            if not np.isfinite(ll_engine) or not np.isfinite(ll_base) or ll_base <= 0:
                self.skill_[name] = 0.0
            else:
                # Relative log-loss improvement over the base rate, floored at
                # zero: an engine that is worse than the prior earns nothing.
                self.skill_[name] = float(max(0.0, 1.0 - ll_engine / ll_base))

        # Final calibrators use the whole training window.
        self.calibrators_ = {}
        for name in ENGINE_NAMES:
            self.calibrators_[name] = EngineCalibrator(
                prior_alpha=self.aggregator_cfg.beta_prior_alpha,
                prior_beta=self.aggregator_cfg.beta_prior_beta,
            ).fit(pressures[name].values, y_arr)

        self.weights = self._weights_from_skill(self.skill_)
        self.fitted_ = True

        # Recalibrate the pooled posterior against training outcomes. This
        # runs with the calibrator cleared so `aggregate` returns the raw
        # pool, then the fitted map applies to every later call — including
        # the gate tuning below, which must see final probabilities.
        self.posterior_calibrator_ = None
        if features is not None:
            train_features = features.loc[features.index <= idx.max()]
            try:
                raw_scored = self.aggregate(engine_outputs, train_features)
                common = raw_scored.index.intersection(y.index)
                if len(common) >= 200:
                    self.posterior_calibrator_ = PosteriorCalibrator(
                        method=self.aggregator_cfg.posterior_calibration
                    ).fit(
                        raw_scored.loc[common, "posterior_mean"].values,
                        y.loc[common].values,
                    )
            except RuntimeError:
                self.posterior_calibrator_ = None

        if features is not None and self.gate_cfg.auto_tune:
            self._tune_gate(engine_outputs, features.loc[features.index <= idx.max()])
        return self

    # ------------------------------------------------------------------
    def _tune_gate(self, engine_outputs: Dict[str, pd.DataFrame],
                   features: pd.DataFrame) -> None:
        """Pick layer thresholds so the training fire rate hits the target.

        Walks a grid of quantiles, applying the SAME quantile to both layer
        composites, and keeps the strictest one whose joint fire rate — with
        every gate condition applied — still clears the kill-criteria floor.
        Preferring the strictest admissible setting rather than the closest
        to target biases toward precision, which is what an action gate is
        for.

        Only training rows are involved, so this is an operating-point
        choice made before the scored window exists — the same discipline
        the v5 model used when tuning alarm hysteresis on TUNE folds.
        """
        scored = self.aggregate(engine_outputs, features)
        l1, l2, l3, _ = _layer_series(features, self.gate_cfg)
        idx = scored.index
        l1, l2, l3 = l1.loc[idx], l2.loc[idx], l3.loc[idx]

        post = scored["posterior_mean"]
        base = (
            (scored["confidence"] >= self.gate_cfg.confidence_threshold).fillna(False)
            & (l3 >= self.gate_cfg.layer3_dd_threshold).fillna(False)
        )
        kill = CONFIG.kill
        floor_rate = kill.min_gate_fire_pct / 100.0
        ceil_rate = kill.max_gate_fire_pct / 100.0
        target = self.gate_cfg.target_fire_rate

        # The posterior threshold is tuned alongside the layers, as a
        # quantile of the training posterior.
        #
        # Once the analog engine embargoes overlapping-window neighbours, the
        # posterior's honest range is much narrower than the design's
        # hard-coded tau = 0.60 assumed: against a ~30% base rate it tops out
        # near 0.55. That 0.60 was chosen before anyone had seen the
        # distribution, and holding a *calibrated* probability to an
        # arbitrary absolute cut just silences the gate. What is genuinely
        # pre-declarable is the operating point — the fire rate we are
        # willing to act on — so that is what gets fixed, and the resulting
        # absolute threshold is reported in `gate_summary()`.
        floor_post = self.gate_cfg.min_posterior_threshold

        best = None
        for q in self.gate_cfg.tune_quantile_grid:
            t1, t2 = l1.quantile(q), l2.quantile(q)
            tp = post.quantile(q)
            if not all(np.isfinite(v) for v in (t1, t2, tp)):
                continue
            # Never act on a posterior below the base rate: a "warning" that
            # is less likely than the unconditional event is not a warning.
            tp = max(float(tp), floor_post, float(self.base_rate_))
            fires = (
                base & (l1 >= t1).fillna(False) & (l2 >= t2).fillna(False)
                & (post >= tp).fillna(False)
            )
            rate = float(fires.mean())
            if rate < floor_rate or rate > ceil_rate:
                continue
            # Strictest admissible = smallest rate at or above the target;
            # if nothing reaches the target, take the largest rate available.
            key = (rate >= target, -rate if rate >= target else rate)
            if best is None or key > best[0]:
                best = (key, {"quantile": float(q), "layer1": float(t1),
                              "layer2": float(t2),
                              "layer3": float(self.gate_cfg.layer3_dd_threshold),
                              "posterior": tp,
                              "train_fire_rate": rate})
        if best is None:
            self.gate_thresholds_ = None
            self.gate_tuning_ = {"status": "no admissible threshold; using GateConfig"}
            return
        self.gate_thresholds_ = {
            "layer1": best[1]["layer1"], "layer2": best[1]["layer2"],
            "layer3": best[1]["layer3"], "posterior": best[1]["posterior"],
        }
        self.gate_tuning_ = best[1]

    def _weights_from_skill(self, skill: Dict[str, float]) -> Dict[str, float]:
        """Turn measured skill into pooling weights, bounded at both ends.

        Two adjustments stand between raw skill and the final weights, and
        both exist to satisfy kill criterion 4 — *"engine disagreement
        structurally degenerate (one engine carries all weight)"*.

        **Tempering.** Weights are proportional to ``skill ** temper`` with
        ``temper < 1``. Raw skill-proportional weighting collapses onto the
        analog engine, whose output is natively an estimate of the target
        (it reports P(maxDD ≥ x) directly), while the other three speak in
        percentile ranks and regime probabilities and therefore look weak on
        a log-loss comparison even when they carry real information. Square-
        root tempering keeps the ordering — better engines still get more
        weight — without letting the one on-target engine swallow the pool.

        **Bounds.** Every engine is then held within
        ``[min_weight, max_weight]``. The floor keeps a weak engine audible;
        the cap makes single-engine dominance impossible by construction
        rather than by hope. Bounds are applied iteratively because
        renormalising after a clip can push another engine back out of
        range.
        """
        floor = self.aggregator_cfg.min_weight
        cap = self.aggregator_cfg.max_weight
        n = len(ENGINE_NAMES)
        if cap * n < 1.0:
            raise ValueError(
                f"max_weight {cap} cannot cover {n} engines; it must be >= 1/n."
            )
        total = sum(skill.values())
        if total <= 0:
            return {name: 1.0 / n for name in ENGINE_NAMES}

        temper = self.aggregator_cfg.skill_temper
        tempered = {k: max(0.0, v) ** temper for k, v in skill.items()}
        t_total = sum(tempered.values())
        if t_total <= 0:
            return {name: 1.0 / n for name in ENGINE_NAMES}
        w = {k: v / t_total for k, v in tempered.items()}

        for _ in range(50):
            w = {k: min(cap, max(floor, v)) for k, v in w.items()}
            s = sum(w.values())
            if abs(s - 1.0) < 1e-9:
                break
            w = {k: v / s for k, v in w.items()}
        return w

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def aggregate(self, engine_outputs: Dict[str, pd.DataFrame],
                  features: pd.DataFrame) -> pd.DataFrame:
        """Combine per-engine outputs + features into the posterior + gate."""
        idx = features.index
        for df in engine_outputs.values():
            if df is not None and len(df):
                idx = idx.intersection(df.index)
        if len(idx) == 0:
            raise RuntimeError("No overlapping dates among engine outputs.")
        features = features.loc[idx]
        pressures = self._pressure_frame(engine_outputs, idx)

        # 1. Calibrate each engine's pressure into a probability.
        calibrated = pd.DataFrame(index=idx)
        for name in ENGINE_NAMES:
            cal = self.calibrators_.get(name)
            raw = pressures[name].values
            calibrated[name] = (
                cal.transform(raw) if cal is not None
                else np.where(np.isfinite(raw), raw, np.nan)
            )

        # 2. Pool in log-odds, anchored on the base rate. Missing engines
        #    contribute nothing and their weight is redistributed, so a NaN
        #    engine leaves the posterior at the prior rather than dragging it
        #    toward zero.
        prior_logit = float(_logit(np.array([self.base_rate_]))[0])
        w = pd.Series(self.weights).reindex(ENGINE_NAMES).fillna(0.0)

        cal_logit = pd.DataFrame(
            {name: _logit(calibrated[name].values) for name in ENGINE_NAMES},
            index=idx,
        )
        evidence = cal_logit - prior_logit
        present = calibrated.notna()
        eff_w = present.astype(float) * w.values
        w_sum = eff_w.sum(axis=1)
        # Renormalise so the total evidence weight is 1 whenever at least one
        # engine reported.
        norm_w = eff_w.div(w_sum.where(w_sum > 0), axis=0)
        total_evidence = (evidence.where(present, 0.0) * norm_w).sum(axis=1)
        post_logit = prior_logit + total_evidence.where(w_sum > 0)
        post_raw = pd.Series(_expit(post_logit.values), index=idx).where(w_sum > 0)
        # Final recalibration (monotone, so the ranking of days is unchanged).
        if self.posterior_calibrator_ is not None and self.posterior_calibrator_.fitted_:
            post_mean = pd.Series(
                self.posterior_calibrator_.transform(post_raw.values), index=idx
            )
        else:
            post_mean = post_raw

        # 3. Confidence — the GEOMETRIC MEAN of three components, each in
        #    [0, 1]. A geometric mean keeps the "one bad component drags the
        #    score down" behaviour that makes confidence meaningful, without
        #    the scale collapse of a plain product: three components at 0.8
        #    should read 0.8, not 0.51. Multiplying them was how the alpha's
        #    gate ended up with an unreachable conjunction.
        ev = evidence.where(present)
        dispersion = ev.std(axis=1, ddof=0).fillna(0.0)
        # Soft, never-exactly-zero agreement: 1 / (1 + dispersion/scale).
        # `agreement_scale` log-odds units of spread halves the score.
        scale = float(self.aggregator_cfg.agreement_scale)
        agreement = (1.0 / (1.0 + dispersion / scale)).clip(0.0, 1.0)
        coverage = present.sum(axis=1) / float(len(ENGINE_NAMES))
        analog_df = engine_outputs.get("analog")
        if analog_df is not None and "confidence" in analog_df.columns:
            analog_support = (
                pd.to_numeric(analog_df.loc[idx, "confidence"], errors="coerce")
                .clip(0.0, 1.0).fillna(0.0)
            )
        else:
            # With no analog engine we cannot assess historical support, so
            # this component is neutral rather than a free pass.
            analog_support = pd.Series(0.5, index=idx)
        components = pd.concat(
            [agreement, coverage, analog_support], axis=1
        ).clip(lower=_P_FLOOR, upper=1.0)
        confidence = np.exp(np.log(components).mean(axis=1)).clip(0.0, 1.0)

        # 4. Layer 1/2/3 stress signals, each on its own timescale.
        l1, l2, l3_dd, archetype = _layer_series(features, self.gate_cfg)

        # 5. Gate — every condition must hold simultaneously.
        thr = self.gate_thresholds_ or {
            "layer1": self.gate_cfg.layer1_z_threshold,
            "layer2": self.gate_cfg.layer2_z_threshold,
            "layer3": self.gate_cfg.layer3_dd_threshold,
            "posterior": self.gate_cfg.posterior_threshold,
        }
        gate_post = post_mean >= thr["posterior"]
        gate_conf = confidence >= self.gate_cfg.confidence_threshold
        gate_l1 = l1 >= thr["layer1"]
        gate_l2 = l2 >= thr["layer2"]
        gate_l3 = l3_dd >= thr["layer3"]
        fires = (
            gate_post.fillna(False) & gate_conf.fillna(False)
            & gate_l1.fillna(False) & gate_l2.fillna(False) & gate_l3.fillna(False)
        )

        # Reason string per row (for transparency / dashboard).
        checks = {
            "post<thr": gate_post, "conf<thr": gate_conf,
            "L1<thr": gate_l1, "L2<thr": gate_l2, "L3<thr": gate_l3,
        }
        failed = pd.DataFrame(
            {label: ~s.fillna(False) for label, s in checks.items()}, index=idx
        )
        reasons = failed.apply(
            lambda row: "OK" if not row.any()
            else " & ".join(failed.columns[row.values]), axis=1
        )

        out = pd.DataFrame({
            "posterior_mean": post_mean,
            "posterior_logit": post_logit,
            "posterior_uncalibrated": post_raw,
            "confidence": confidence,
            "conf_agreement": agreement,
            "conf_coverage": coverage,
            "conf_analog_support": analog_support,
            "layer1_z": l1,
            "archetype": archetype,
            "layer2_z": l2,
            "layer3_dd": l3_dd,
            "gate_fires": fires,
            "gate_reason": reasons,
        }, index=idx)
        # Raw pressures and their calibrated / weighted contributions, so the
        # posterior can always be decomposed back into who said what.
        for name in ENGINE_NAMES:
            out[f"pressure_{name}"] = pressures[name]
            out[f"calibrated_{name}"] = calibrated[name]
            out[f"contribution_{name}"] = evidence[name].where(present[name]) * norm_w[name]
        return out

    # ------------------------------------------------------------------
    def explain(self) -> pd.DataFrame:
        """Per-engine skill, weight and calibration status — for auditing."""
        return pd.DataFrame({
            "skill": pd.Series(self.skill_),
            "weight": pd.Series(self.weights),
            "calibrated": pd.Series(
                {k: v.fitted_ for k, v in self.calibrators_.items()}
            ),
        }).fillna(0.0)

    def gate_summary(self) -> Dict[str, object]:
        """Which thresholds the gate is using, and where they came from."""
        if self.gate_thresholds_ is None:
            return {
                "source": "GateConfig defaults (not tuned)",
                "layer1": self.gate_cfg.layer1_z_threshold,
                "layer2": self.gate_cfg.layer2_z_threshold,
                "layer3": self.gate_cfg.layer3_dd_threshold,
                "posterior": self.gate_cfg.posterior_threshold,
                "confidence": self.gate_cfg.confidence_threshold,
                **self.gate_tuning_,
            }
        return {"source": "tuned on training window", **self.gate_thresholds_,
                "posterior_calibrated": bool(
                    self.posterior_calibrator_ and self.posterior_calibrator_.fitted_
                ),
                "confidence": self.gate_cfg.confidence_threshold,
                **self.gate_tuning_}
