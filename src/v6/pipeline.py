"""v6 Pipeline orchestrator.

`CrashKPIPipeline.fit_until(date)` fits all five engines on data strictly
before `date`. `score(date_range, x_pct, horizon_td)` runs all engines
forward, aggregates via Bayesian aggregator, applies L1/L2/L3 gate, and
returns a per-date DataFrame.

This is the single object the walk-forward harness, the BLIND evaluator,
and the dashboard all talk to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.v6.config import CONFIG, DEFAULT_X_PCT, DEFAULT_HORIZON_DAYS
from src.v6.features import FeatureBuilder
from src.v6.engines import (
    AnomalyEngine, RegimeEngine, AnalogEngine, CausalEngine, CrashKPIAggregator,
)


@dataclass
class CrashKPIPipeline:
    anomaly: AnomalyEngine = field(default_factory=AnomalyEngine)
    regime: RegimeEngine = field(default_factory=RegimeEngine)
    analog: AnalogEngine = field(default_factory=AnalogEngine)
    causal: CausalEngine = field(default_factory=CausalEngine)
    aggregator: CrashKPIAggregator = field(default_factory=CrashKPIAggregator)
    features_: Optional[pd.DataFrame] = None
    raw_: Optional[pd.DataFrame] = None
    fit_through_: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    def fit_until(self, fit_through: str) -> "CrashKPIPipeline":
        """Train all engines on data strictly through `fit_through`.

        Parameters
        ----------
        fit_through : str
            ISO date — engines see data with index <= this date only.
        """
        fb = FeatureBuilder()
        raw_full = fb.load_raw()
        features_full = fb.build()
        prices_full = features_full["_price"]
        feats_full = features_full.drop(columns="_price")

        cutoff = pd.Timestamp(fit_through)
        train_raw = raw_full.loc[raw_full.index <= cutoff]
        train_feats = feats_full.loc[feats_full.index <= cutoff]
        train_prices = prices_full.loc[prices_full.index <= cutoff]

        # Fit each engine on training-only data.
        self.anomaly.fit(train_feats, train_prices)
        # Regime engine needs the HMM-specific subset of features.
        hmm_feats = self.regime.HMM_FEATURES if hasattr(self.regime, "HMM_FEATURES") else None
        self.regime.fit(train_feats)
        self.analog.fit(train_feats, train_prices)
        self.causal.fit(train_raw)

        # Stash full-history (training + future) features for scoring.
        self.features_ = feats_full
        self.raw_ = raw_full
        self.fit_through_ = cutoff
        return self

    # ------------------------------------------------------------------
    def score(self, start: Optional[str] = None, end: Optional[str] = None,
              x_pct: float = DEFAULT_X_PCT,
              horizon_td: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
        """Score a date range. Returns the aggregator+gate DataFrame.

        Parameters
        ----------
        start, end : str | None
            Inclusive date range to score. Defaults to (fit_through+1, last).
        x_pct, horizon_td : float, int
            INFERENCE-time tunable crash threshold and horizon.
        """
        if self.features_ is None:
            raise RuntimeError("Pipeline not fit. Call .fit_until() first.")
        # IMPORTANT: run engines on FULL history so expanding stats (z-scores,
        # rolling windows) have enough warm-up. Slice the OUTPUT, not the input.
        feats_full = self.features_

        anomaly_df = self.anomaly.score(feats_full)
        regime_df = self.regime.score(feats_full, h_steps=horizon_td)
        analog_df = self.analog.query_dataframe(feats_full, x_pct=x_pct, horizon_td=horizon_td)
        causal_df = self.causal.score(self.raw_)

        engine_outputs = {
            "anomaly": anomaly_df,
            "regime": regime_df,
            "analog": analog_df,
            "causal": causal_df,
        }
        out = self.aggregator.aggregate(engine_outputs, feats_full)
        if start is not None:
            out = out.loc[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out.loc[out.index <= pd.Timestamp(end)]
        out["x_pct"] = x_pct
        out["horizon_td"] = horizon_td
        return out
