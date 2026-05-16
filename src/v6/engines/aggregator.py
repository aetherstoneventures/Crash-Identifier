"""Engine 5 — Bayesian Aggregator + Layer 1/2/3 Simultaneous-Agreement Gate.

Combines the four upstream engines into a single posterior

    P(maxDD ≥ x_pct in next h trading days)

with a confidence (inverse posterior interval width) and a transparent
per-engine contribution decomposition.

Engines and the question each answers:
    1. AnomalyEngine         "is now unusual?"
    2. RegimeEngine          "what regime are we in?"
    3. AnalogEngine          "what historically followed setups like this?"
    4. CausalEngine          "what's the mechanism / is structure breaking?"

Aggregation method
------------------
Each engine emits a "crash-pressure" probability p_e ∈ [0, 1] (see
`_engine_pressure` below for the per-engine mapping). We treat each as a
noisy Bernoulli observation of the unobserved latent crash indicator and
combine via weighted Beta-Binomial:

    α = α_0 + Σ_e w_e · p_e
    β = β_0 + Σ_e w_e · (1 - p_e)
    posterior_mean = α / (α + β)
    posterior_var  = αβ / ((α+β)^2 (α+β+1))

The weights w_e are equal at v0 (1/4 each). Future work: learn per-engine
weights from past walk-forward fold calibration error.

Confidence = 1 - 2·sqrt(posterior_var) clipped to [0, 1].

Layer 1/2/3 Gate (the user's "100% agreement" constraint)
---------------------------------------------------------
Action fires only when ALL hold:
    - aggregator posterior > τ           (default 0.60)
    - aggregator confidence > κ          (default 0.50)
    - Layer 1 macro regime elevated      (rolling z of macro-stress > 1.5σ)
    - Layer 2 tactical stress elevated   (rolling z of vol/breadth > 1.5σ)
    - Layer 3 price/vol confirms         (current drawdown >= dd_threshold)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.v6.config import AggregatorConfig, CONFIG, GateConfig


# ---------------------------------------------------------------------------
# Layer composition — which features feed which layer's stress score.
# All are existing v6 feature names; missing ones are silently dropped.
# ---------------------------------------------------------------------------
LAYER1_FEATURES = [
    "yc_10y_3m", "hy_spread_z", "ig_spread_z", "nfci",
    "epu_z", "sahm_indicator", "cape_proxy_z", "real_rate_10y",
    "m2_yoy_z", "margin_debt_z",
]
LAYER2_FEATURES = [
    "rv_21_z", "rv_63_z", "vix_z", "vix_term_structure", "skew_z",
    "put_call_z", "cross_asset_corr_z", "return_dispersion_63",
    "hy_spread_chg", "iv_rv_gap", "oil_shock_z",
]
LAYER3_PRICE_COL = "dd_from_252h"   # always present from FeatureBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expanding_z(s: pd.Series, min_periods: int = 252) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def _composite_score(features: pd.DataFrame, names: List[str]) -> pd.Series:
    """Sign-flipped, expanding-z'd composite of feature group.

    For features where high = stress (vix, dd, etc.) we leave as-is.
    For features where LOW = stress (yc_10y_3m -inverted curve), invert.
    Then take row mean of z-scores -> single composite series.
    """
    inv_features = {"yc_10y_3m", "yc_10y_2y"}  # inverted curve = stress
    valid = [c for c in names if c in features.columns and features[c].notna().sum() > 252]
    if not valid:
        return pd.Series(np.nan, index=features.index)
    z_cols = []
    for c in valid:
        s = features[c].astype(float)
        if c in inv_features:
            s = -s
        z_cols.append(_expanding_z(s, min_periods=252))
    return pd.concat(z_cols, axis=1).mean(axis=1)


# ---------------------------------------------------------------------------
# Aggregator + Gate
# ---------------------------------------------------------------------------
@dataclass
class AggregatorResult:
    date: pd.Timestamp
    posterior_mean: float
    posterior_std: float
    confidence: float
    per_engine_pressure: Dict[str, float]
    layer1_z: float
    layer2_z: float
    layer3_dd: float
    gate_fires: bool
    gate_reason: str


@dataclass
class CrashKPIAggregator:
    """Bayesian aggregator + L1/L2/L3 gate."""

    aggregator_cfg: AggregatorConfig = field(default_factory=lambda: CONFIG.aggregator)
    gate_cfg: GateConfig = field(default_factory=lambda: CONFIG.gate)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "anomaly": 0.25, "regime": 0.25, "analog": 0.25, "causal": 0.25,
    })

    # ---------- Per-engine pressure mappings ----------
    @staticmethod
    def _engine_pressure(engine_name: str, row: pd.Series) -> float:
        """Map each engine's row to a 'crash-pressure' probability in [0, 1].

        - anomaly: ensemble_rank IS already a probability of being unusual.
        - regime:  stress_score IS already P(in stress/crisis state).
        - analog:  prob IS the empirical P(maxDD ≥ x in next h).
        - causal:  stress_composite IS rank in [0, 1] vs training.
        """
        if engine_name == "anomaly":
            v = row.get("ensemble_rank", np.nan)
        elif engine_name == "regime":
            v = row.get("stress_score", np.nan)
        elif engine_name == "analog":
            v = row.get("prob", np.nan)
        elif engine_name == "causal":
            v = row.get("stress_composite", np.nan)
        else:
            v = np.nan
        return float(np.clip(v, 0.0, 1.0)) if not np.isnan(v) else np.nan

    def aggregate(self, engine_outputs: Dict[str, pd.DataFrame],
                  features: pd.DataFrame) -> pd.DataFrame:
        """Combine per-engine outputs + features into the aggregator + gate.

        Parameters
        ----------
        engine_outputs : dict
            Keys 'anomaly', 'regime', 'analog', 'causal'; each value is a
            date-indexed DataFrame from the respective engine.
        features : pd.DataFrame
            Date-indexed feature DataFrame (from FeatureBuilder.build()) —
            used to compute the L1/L2/L3 gate scores.
        """
        # Align all engines on the intersection of their indices.
        idx = features.index
        for df in engine_outputs.values():
            idx = idx.intersection(df.index)
        if len(idx) == 0:
            raise RuntimeError("No overlapping dates among engine outputs.")
        features = features.loc[idx]
        eng = {k: v.loc[idx] for k, v in engine_outputs.items()}

        # Per-engine pressure series.
        pressures = pd.DataFrame(index=idx)
        for name in ("anomaly", "regime", "analog", "causal"):
            df = eng.get(name)
            if df is None or len(df) == 0:
                pressures[name] = np.nan
                continue
            pressures[name] = [self._engine_pressure(name, df.iloc[i]) for i in range(len(idx))]

        a0 = self.aggregator_cfg.beta_prior_alpha
        b0 = self.aggregator_cfg.beta_prior_beta

        # Weighted Beta-Binomial update (row-wise vectorised).
        w = pd.Series(self.weights)
        w = w[pressures.columns]
        # Treat NaN pressure as missing observation (weight 0).
        valid = pressures.notna().astype(float)
        eff_w = valid * w.values
        weighted_p = (pressures.fillna(0.0) * eff_w).sum(axis=1)
        weighted_one_minus = ((1.0 - pressures.fillna(0.0)) * eff_w).sum(axis=1)
        alpha = a0 + weighted_p
        beta = b0 + weighted_one_minus
        post_mean = alpha / (alpha + beta)
        post_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        post_std = np.sqrt(post_var)
        confidence = (1.0 - 2.0 * post_std).clip(0.0, 1.0)

        # Layer 1/2/3 stress composites.
        l1 = _composite_score(features, LAYER1_FEATURES)
        l2 = _composite_score(features, LAYER2_FEATURES)
        l3_dd = -features[LAYER3_PRICE_COL].astype(float)  # convert -DD% to +stress

        # Gate (all must hold).
        gate_post = post_mean >= self.gate_cfg.posterior_threshold
        gate_conf = confidence >= self.gate_cfg.confidence_threshold
        gate_l1 = l1 >= self.gate_cfg.layer1_z_threshold
        gate_l2 = l2 >= self.gate_cfg.layer2_z_threshold
        gate_l3 = l3_dd >= self.gate_cfg.layer3_dd_threshold
        fires = gate_post & gate_conf & gate_l1 & gate_l2 & gate_l3

        # Reason string per row (for transparency / dashboard).
        reasons = pd.Series("", index=idx, dtype=object)
        for i in range(len(idx)):
            flags = []
            if not gate_post.iloc[i]:
                flags.append("post<thr")
            if not gate_conf.iloc[i]:
                flags.append("conf<thr")
            if not gate_l1.iloc[i]:
                flags.append("L1<thr")
            if not gate_l2.iloc[i]:
                flags.append("L2<thr")
            if not gate_l3.iloc[i]:
                flags.append("L3<thr")
            reasons.iloc[i] = "OK" if not flags else " & ".join(flags)

        out = pd.DataFrame({
            "posterior_mean": post_mean,
            "posterior_std": post_std,
            "confidence": confidence,
            "pressure_anomaly": pressures.get("anomaly"),
            "pressure_regime": pressures.get("regime"),
            "pressure_analog": pressures.get("analog"),
            "pressure_causal": pressures.get("causal"),
            "layer1_z": l1,
            "layer2_z": l2,
            "layer3_dd": l3_dd,
            "gate_fires": fires,
            "gate_reason": reasons,
        }, index=idx)
        return out
