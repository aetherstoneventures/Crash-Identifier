"""Regression tests for the defects found in the v6.0.0-alpha post-mortem.

Each test here corresponds to a specific bug documented in
`docs/V6_POSTMORTEM.md`. They are written to fail against the alpha and
pass against v6.1, so the same mistakes cannot return unnoticed.

These are pure unit tests on synthetic data — no database, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v6.config import CONFIG
from src.v6.engines.aggregator import (
    ARCHETYPES, CrashKPIAggregator, EngineCalibrator, _composite_score,
    _expit, _layer_series, _logit,
)
from src.v6.features.labels import crash_label, forward_maxdd
from src.v6.features.quality import QUARANTINE, screen_column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dates(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _price_series(n: int = 800, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.011, n)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=_dates(n))


# ---------------------------------------------------------------------------
# DEFECT 1 — the aggregator posterior was confined to [1/3, 2/3]
# ---------------------------------------------------------------------------
class TestPosteriorRange:
    """The alpha's Beta-Binomial update could not leave [0.3333, 0.6667].

    With a Beta(1,1) prior and weights summing to 1, alpha + beta was always
    exactly 3, so unanimous certainty still returned 0.667 and the gate's
    tau = 0.60 needed a mean engine pressure of 0.80.
    """

    def _engine_frames(self, pressure: float, n: int = 400):
        idx = _dates(n)
        return {
            "anomaly": pd.DataFrame({"ensemble_rank": pressure}, index=idx),
            "regime": pd.DataFrame({"h_step_stress_prob": pressure}, index=idx),
            "analog": pd.DataFrame(
                {"prob": pressure, "confidence": 0.8}, index=idx
            ),
            "causal": pd.DataFrame({"stress_composite": pressure}, index=idx),
        }, idx

    def _features(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        cols = {c: rng.normal(size=len(idx))
                for group in ARCHETYPES.values() for c in group}
        cols["dd_from_252h"] = -rng.uniform(0, 20, len(idx))
        return pd.DataFrame(cols, index=idx)

    def test_unanimous_evidence_escapes_the_alpha_ceiling(self):
        agg = CrashKPIAggregator()
        engines, idx = self._engine_frames(0.99)
        features = self._features(idx)
        # Calibrate on a training set where high pressure really does mean
        # the event happened, so the evidence is genuine.
        y = pd.Series(1.0, index=idx)
        y.iloc[: len(idx) // 4] = 0.0
        agg.fit(engines, y, features=features)
        out = agg.aggregate(engines, features)
        assert out["posterior_mean"].max() > 0.70, (
            "posterior should exceed the alpha's hard ceiling of 0.6667 "
            f"under unanimous evidence; got {out['posterior_mean'].max():.4f}"
        )

    def test_unanimous_absence_goes_low(self):
        agg = CrashKPIAggregator()
        engines, idx = self._engine_frames(0.01)
        features = self._features(idx)
        y = pd.Series(0.0, index=idx)
        y.iloc[: len(idx) // 4] = 1.0
        agg.fit(engines, y, features=features)
        out = agg.aggregate(engines, features)
        assert out["posterior_mean"].min() < 0.30, (
            "posterior should fall below the alpha's floor of 0.3333 when "
            f"every engine is quiet; got {out['posterior_mean'].min():.4f}"
        )

    def test_logit_expit_roundtrip(self):
        p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        assert np.allclose(_expit(_logit(p)), p, atol=1e-6)


# ---------------------------------------------------------------------------
# DEFECT 2 — confidence was a constant, and the gate compared it to 0.50
# ---------------------------------------------------------------------------
class TestConfidenceIsInformative:
    """The alpha's confidence spanned [0.5000, 0.5286] across all inputs."""

    def _run(self, pressures: dict) -> pd.DataFrame:
        idx = _dates(300)
        engines = {
            "anomaly": pd.DataFrame({"ensemble_rank": pressures["anomaly"]}, index=idx),
            "regime": pd.DataFrame({"h_step_stress_prob": pressures["regime"]}, index=idx),
            "analog": pd.DataFrame(
                {"prob": pressures["analog"], "confidence": pressures["support"]},
                index=idx,
            ),
            "causal": pd.DataFrame({"stress_composite": pressures["causal"]}, index=idx),
        }
        rng = np.random.default_rng(2)
        cols = {c: rng.normal(size=len(idx))
                for group in ARCHETYPES.values() for c in group}
        cols["dd_from_252h"] = -rng.uniform(0, 20, len(idx))
        features = pd.DataFrame(cols, index=idx)
        agg = CrashKPIAggregator()
        return agg.aggregate(engines, features)

    def test_agreement_beats_disagreement(self):
        agreed = self._run({"anomaly": 0.9, "regime": 0.9, "analog": 0.9,
                            "causal": 0.9, "support": 0.9})
        split = self._run({"anomaly": 0.95, "regime": 0.05, "analog": 0.95,
                           "causal": 0.05, "support": 0.9})
        assert agreed["confidence"].mean() > split["confidence"].mean(), (
            "four engines agreeing must be more confident than four engines "
            "split down the middle"
        )

    def test_analog_support_moves_confidence(self):
        strong = self._run({"anomaly": 0.8, "regime": 0.8, "analog": 0.8,
                            "causal": 0.8, "support": 0.95})
        weak = self._run({"anomaly": 0.8, "regime": 0.8, "analog": 0.8,
                          "causal": 0.8, "support": 0.05})
        assert strong["confidence"].mean() > weak["confidence"].mean(), (
            "'no good analogs found' must reduce confidence — the alpha "
            "computed this signal and then discarded it"
        )


# ---------------------------------------------------------------------------
# DEFECT 3 — layer composites were never re-standardised
# ---------------------------------------------------------------------------
class TestCompositeStandardisation:
    """Averaging n z-scores shrinks dispersion by ~sqrt(n).

    The alpha compared that shrunken average to 1.5, so a nominal "1.5 sigma"
    threshold was really a 2.4-3.1 sigma event.
    """

    def test_composite_has_unit_scale(self):
        idx = _dates(3000)
        rng = np.random.default_rng(3)
        names = [f"f{i}" for i in range(10)]
        features = pd.DataFrame(
            {n: rng.normal(size=len(idx)) for n in names}, index=idx
        )
        composite = _composite_score(features, names).dropna()
        assert 0.75 < composite.std() < 1.35, (
            f"composite std should be ~1 after re-standardisation, got "
            f"{composite.std():.3f} (the alpha produced ~0.5)"
        )

    def test_threshold_is_reached_at_a_plausible_rate(self):
        idx = _dates(3000)
        rng = np.random.default_rng(4)
        names = [f"f{i}" for i in range(10)]
        features = pd.DataFrame(
            {n: rng.normal(size=len(idx)) for n in names}, index=idx
        )
        composite = _composite_score(features, names).dropna()
        rate = float((composite >= 1.5).mean())
        assert 0.02 < rate < 0.15, (
            f"a 1.5-sigma threshold should be crossed on a few percent of "
            f"days, got {100 * rate:.2f}%"
        )


# ---------------------------------------------------------------------------
# DEFECT 4 — a single averaged Layer 1 was blind to non-credit crashes
# ---------------------------------------------------------------------------
class TestArchetypeAwareLayer1:
    """2022 was rate-led: credit calm, labour strong, real rates spiking.

    A blended macro composite scores that as "macro is fine" and vetoes the
    gate. Layer 1 must recognise a stressed archetype on its own terms.
    """

    def test_rate_stress_alone_lifts_layer1(self):
        idx = _dates(1200)
        rng = np.random.default_rng(5)
        cols = {c: rng.normal(0, 1, len(idx))
                for group in ARCHETYPES.values() for c in group}
        features = pd.DataFrame(cols, index=idx)
        features["dd_from_252h"] = -rng.uniform(0, 5, len(idx))

        # Credit-led inputs stay calm; only the rate archetype is stressed,
        # in the last quarter of the sample.
        tail = features.index[-300:]
        for col in ARCHETYPES["rate_led"]:
            # yc_* and m2_yoy_z are inverted (low = stress); push both ways
            # so the archetype composite rises either way.
            features.loc[tail, col] = -6.0 if col in {
                "yc_10y_3m", "yc_10y_2y", "m2_yoy_z"
            } else 6.0

        l1, _, _, archetype = _layer_series(features, CONFIG.gate)
        assert l1.loc[tail].max() > 1.5, (
            "a purely rate-led macro regime must register on Layer 1; "
            f"peak was {l1.loc[tail].max():.3f}"
        )
        assert (archetype.loc[tail] == "rate_led").any(), (
            "the winning archetype should be reported as rate_led, got "
            f"{archetype.loc[tail].value_counts().to_dict()}"
        )

    def test_every_archetype_is_reachable(self):
        """No archetype may be dead code — each must be able to win."""
        idx = _dates(1200)
        rng = np.random.default_rng(6)
        for target, cols in ARCHETYPES.items():
            frame = pd.DataFrame(
                {c: rng.normal(0, 1, len(idx))
                 for group in ARCHETYPES.values() for c in group},
                index=idx,
            )
            frame["dd_from_252h"] = -rng.uniform(0, 5, len(idx))
            tail = frame.index[-300:]
            for col in cols:
                frame.loc[tail, col] = 8.0
                frame.loc[tail, col] *= -1 if col in {
                    "yc_10y_3m", "yc_10y_2y", "m2_yoy_z", "ma200_dist", "acf_20"
                } else 1
            _, _, _, archetype = _layer_series(frame, CONFIG.gate)
            assert (archetype.loc[tail] == target).any(), (
                f"archetype {target!r} never wins even when all of its own "
                f"features are extreme"
            )


# ---------------------------------------------------------------------------
# DEFECT 5 — training target and evaluation label were different events
# ---------------------------------------------------------------------------
class TestLabels:
    def test_forward_maxdd_measures_peak_to_trough(self):
        # Rises 10% then falls 20% from that peak inside the window.
        px = pd.Series(
            [100, 105, 110, 88, 95, 100, 100, 100],
            index=_dates(8),
        )
        dd = forward_maxdd(px, horizon_td=3)
        # From index 0 the window [0..3] peaks at 110 and troughs at 88.
        assert dd.iloc[0] == pytest.approx(20.0, abs=0.01), (
            "drawdown must be measured from the running peak inside the "
            "window, not from the price on the signal date"
        )

    def test_incomplete_windows_are_nan_not_false(self):
        px = _price_series(100)
        dd = forward_maxdd(px, horizon_td=20)
        assert dd.iloc[-20:].isna().all(), (
            "dates whose forward window runs past the data must be NaN, so "
            "scoring can drop them instead of counting them as non-events"
        )
        y = crash_label(px, x_pct=5.0, horizon_td=20)
        assert y.iloc[-20:].isna().all()

    def test_label_is_monotone_in_threshold(self):
        px = _price_series(600, seed=7)
        strict = crash_label(px, x_pct=15.0, horizon_td=63).fillna(0)
        loose = crash_label(px, x_pct=5.0, horizon_td=63).fillna(0)
        assert (loose >= strict).all(), (
            "every 15% drawdown is also a 5% drawdown"
        )

    def test_tunable_x_actually_changes_the_label(self):
        px = _price_series(600, seed=8)
        rates = [
            crash_label(px, x_pct=x, horizon_td=63).mean()
            for x in (2.0, 5.0, 10.0)
        ]
        assert rates[0] > rates[1] > rates[2], (
            f"base rate must fall as x rises; got {rates}"
        )


# ---------------------------------------------------------------------------
# DEFECT 6 — fabricated columns were modelled as if they were market data
# ---------------------------------------------------------------------------
class TestQualityScreen:
    def test_constant_fill_is_rejected(self):
        idx = _dates(2000)
        s = pd.Series(np.r_[np.full(1200, 17.24), np.random.default_rng(9).normal(20, 5, 800)],
                      index=idx)
        verdict = screen_column(s, "vix_close")
        assert not verdict.passed
        assert "constant_fill" in verdict.reason

    def test_synthetic_noise_is_rejected(self):
        rng = np.random.default_rng(10)
        idx = _dates(2000)
        s = pd.Series(rng.normal(1.0, 0.033, len(idx)), index=idx)
        verdict = screen_column(s, "some_ratio")
        assert not verdict.passed
        assert "implausible_noise" in verdict.reason

    def test_real_looking_series_passes(self):
        idx = _dates(2000)
        px = _price_series(2000, seed=11)
        px.index = idx
        assert screen_column(px, "nasdaq_close").passed

    def test_target_labels_are_quarantined(self):
        for col in ("in_crash", "pre_crash_30d", "pre_crash_60d"):
            assert col in QUARANTINE, f"{col} is a label and must never be a feature"


# ---------------------------------------------------------------------------
# DEFECT 7 — one engine took ~86% of the pooling weight (kill criterion 4)
# ---------------------------------------------------------------------------
class TestWeightBounds:
    def test_no_engine_can_dominate(self):
        agg = CrashKPIAggregator()
        weights = agg._weights_from_skill(
            {"anomaly": 0.0, "regime": 0.0, "analog": 0.99, "causal": 0.0}
        )
        assert max(weights.values()) <= CONFIG.aggregator.max_weight + 1e-9, (
            "kill criterion 4 forbids one engine carrying all the weight"
        )
        assert min(weights.values()) >= CONFIG.aggregator.min_weight - 1e-9
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_zero_skill_falls_back_to_equal_weights(self):
        agg = CrashKPIAggregator()
        weights = agg._weights_from_skill({n: 0.0 for n in
                                           ("anomaly", "regime", "analog", "causal")})
        assert all(w == pytest.approx(0.25) for w in weights.values())

    def test_more_skill_still_earns_more_weight(self):
        agg = CrashKPIAggregator()
        weights = agg._weights_from_skill(
            {"anomaly": 0.4, "regime": 0.1, "analog": 0.2, "causal": 0.05}
        )
        assert weights["anomaly"] > weights["analog"] > weights["regime"] > weights["causal"]


# ---------------------------------------------------------------------------
# Calibrator behaviour
# ---------------------------------------------------------------------------
class TestEngineCalibrator:
    def test_maps_pressure_onto_observed_frequency(self):
        rng = np.random.default_rng(12)
        n = 4000
        pressure = rng.uniform(0, 1, n)
        # True event probability equals the pressure itself.
        y = (rng.uniform(0, 1, n) < pressure).astype(float)
        cal = EngineCalibrator().fit(pressure, y)
        out = cal.transform(np.array([0.05, 0.5, 0.95]))
        assert out[0] < out[1] < out[2]
        assert out[2] > 0.7 and out[0] < 0.3

    def test_sparse_bins_shrink_toward_the_prior(self):
        # Three observations all positive should not yield a confident 1.0.
        cal = EngineCalibrator(prior_alpha=1.0, prior_beta=1.0)
        pressure = np.r_[np.zeros(200), np.ones(3)]
        y = np.r_[np.zeros(200), np.ones(3)]
        cal.fit(pressure, y)
        assert cal.transform(np.array([1.0]))[0] < 1.0

    def test_uninformative_input_returns_base_rate(self):
        rng = np.random.default_rng(13)
        pressure = rng.uniform(0, 1, 1000)
        y = np.full(1000, 1.0)          # single class -> nothing to learn
        cal = EngineCalibrator().fit(pressure, y)
        assert cal.transform(np.array([0.9]))[0] == pytest.approx(1.0)
