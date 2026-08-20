"""v6 Pipeline orchestrator.

`CrashKPIPipeline.fit_until(date)` fits all five engines on data strictly
before `date`. `score(date_range, x_pct, horizon_td)` runs all engines
forward, aggregates via the calibrated log-odds aggregator, applies the
L1/L2/L3 gate, and returns a per-date DataFrame.

This is the single object the walk-forward harness, the BLIND evaluator,
and the dashboard all talk to.

HOW THE TUNABLE x% SURVIVES CALIBRATION
=======================================
The design's central promise is that the crash threshold x% and horizon h
are chosen at **inference** time, not baked into training. The aggregator,
however, needs to know what event it is estimating in order to calibrate
each engine against it.

Both hold at once because calibration is cheap and uses only training data:

1. `fit_until(cutoff)` fits the four upstream engines once. They are
   x-agnostic — the analog engine stores forward drawdowns for every
   supported horizon and thresholds them at query time.
2. `score(..., x_pct, horizon_td)` derives the outcome label for *that*
   query from **training-window prices only**, calibrates the aggregator on
   the training rows, then applies it to the scored range.

So one fit serves every (x, h) pair, and no calibration ever sees a price
from the scoring window. The label for the last `horizon_td` days of the
training window is dropped, since its outcome was not yet knowable at the
cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from src.v6.config import CONFIG, DEFAULT_X_PCT, DEFAULT_HORIZON_DAYS
from src.v6.features import FeatureBuilder
from src.v6.features.labels import crash_label
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
    prices_: Optional[pd.Series] = None
    fit_through_: Optional[pd.Timestamp] = None
    price_col_: Optional[str] = None
    engine_errors_: Dict[str, str] = field(default_factory=dict)

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
        features_full = fb.build(raw_full)
        prices_full = features_full["_price"]
        feats_full = features_full.drop(columns="_price")
        self.price_col_ = fb.resolved_price_col_

        cutoff = pd.Timestamp(fit_through)
        train_raw = raw_full.loc[raw_full.index <= cutoff]
        train_feats = feats_full.loc[feats_full.index <= cutoff]
        train_prices = prices_full.loc[prices_full.index <= cutoff]

        # Fit each engine on training-only data. An engine that cannot fit
        # (too little history in an early fold, say) is recorded and skipped
        # rather than taking the whole pipeline down — the aggregator treats
        # it as a missing observation and redistributes its weight.
        self.engine_errors_ = {}
        for name, fit_call in (
            ("anomaly", lambda: self.anomaly.fit(train_feats, train_prices)),
            ("regime", lambda: self.regime.fit(train_feats)),
            ("analog", lambda: self.analog.fit(train_feats, train_prices)),
            ("causal", lambda: self.causal.fit(train_raw)),
        ):
            try:
                fit_call()
            except Exception as exc:  # noqa: BLE001 — recorded, not silenced
                self.engine_errors_[name] = f"{type(exc).__name__}: {exc}"

        if len(self.engine_errors_) == len(("anomaly", "regime", "analog", "causal")):
            raise RuntimeError(
                "No engine could be fit through "
                f"{cutoff.date()}: {self.engine_errors_}"
            )

        # Stash full-history (training + future) features for scoring.
        self.features_ = feats_full
        self.raw_ = raw_full
        self.prices_ = prices_full
        self.fit_through_ = cutoff
        return self

    # ------------------------------------------------------------------
    def _engine_outputs(self, horizon_td: int, x_pct: float) -> Dict[str, pd.DataFrame]:
        """Run every fitted engine over the full feature history.

        Engines run on FULL history so expanding statistics (z-scores,
        rolling windows) have their warm-up; the OUTPUT is sliced later.
        This is safe because each engine was *fit* on training data only and
        scores each date from its own past.
        """
        feats_full = self.features_
        outputs: Dict[str, pd.DataFrame] = {}
        runners = {
            "anomaly": lambda: self.anomaly.score(feats_full),
            "regime": lambda: self.regime.score(feats_full, h_steps=horizon_td),
            "analog": lambda: self.analog.query_dataframe(
                feats_full, x_pct=x_pct, horizon_td=horizon_td
            ),
            "causal": lambda: self.causal.score(self.raw_),
        }
        for name, run in runners.items():
            if name in self.engine_errors_:
                continue
            try:
                outputs[name] = run()
            except Exception as exc:  # noqa: BLE001
                self.engine_errors_[name] = f"score: {type(exc).__name__}: {exc}"
        if not outputs:
            raise RuntimeError(f"No engine produced output: {self.engine_errors_}")
        return outputs

    # ------------------------------------------------------------------
    def score(self, start: Optional[str] = None, end: Optional[str] = None,
              x_pct: float = DEFAULT_X_PCT,
              horizon_td: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
        """Score a date range. Returns the aggregator+gate DataFrame.

        Parameters
        ----------
        start, end : str | None
            Inclusive date range to score. Defaults to the full history.
        x_pct, horizon_td : float, int
            INFERENCE-time tunable crash threshold and horizon.
        """
        if self.features_ is None:
            raise RuntimeError("Pipeline not fit. Call .fit_until() first.")

        engine_outputs = self._engine_outputs(horizon_td=horizon_td, x_pct=x_pct)

        # Calibrate the aggregator on the training window for THIS (x, h).
        self._fit_aggregator(engine_outputs, x_pct=x_pct, horizon_td=horizon_td)

        out = self.aggregator.aggregate(engine_outputs, self.features_)
        if start is not None:
            out = out.loc[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out.loc[out.index <= pd.Timestamp(end)]
        out["x_pct"] = x_pct
        out["horizon_td"] = horizon_td
        return out

    def _fit_aggregator(self, engine_outputs: Dict[str, pd.DataFrame],
                        x_pct: float, horizon_td: int) -> None:
        """Calibrate engines against the training-window realisation of (x, h)."""
        cutoff = self.fit_through_
        train_prices = self.prices_.loc[self.prices_.index <= cutoff]
        y_train = crash_label(train_prices, x_pct=x_pct, horizon_td=horizon_td).dropna()
        if len(y_train) == 0:
            # No complete forward window inside training — leave the
            # aggregator uncalibrated (equal weights, base rate 0.5) rather
            # than fitting on nothing.
            return
        train_outputs = {
            name: df.loc[df.index <= cutoff]
            for name, df in engine_outputs.items()
        }
        train_features = self.features_.loc[self.features_.index <= cutoff]
        self.aggregator.fit(train_outputs, y_train, features=train_features)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        """What was fit, on what, and with which engine weights."""
        return {
            "price_column": self.price_col_,
            "fit_through": str(self.fit_through_.date()) if self.fit_through_ else None,
            "n_feature_dates": int(len(self.features_)) if self.features_ is not None else 0,
            "engine_errors": dict(self.engine_errors_),
            "base_rate": self.aggregator.base_rate_,
            "weights": dict(self.aggregator.weights),
            "skill": dict(self.aggregator.skill_),
            "gate": self.aggregator.gate_summary(),
        }
