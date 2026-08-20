"""Causality tests — the system must never read the future.

The v6.0.0-alpha post-mortem found two look-ahead paths that no test would
have caught:

1. The regime engine scored with `hmmlearn.predict_proba`, which runs
   forward-BACKWARD smoothing. The state probability at date t was
   conditioned on observations after t, and the pipeline scored the whole
   history in one call, so every historical date was told its own future.

2. The analog engine retrieved neighbours without an embargo. Adjacent
   trading days are nearly identical in feature space and their forward
   windows overlap almost completely, so a date's nearest analogs were its
   own neighbouring dates — which already knew its outcome.

The decisive property for both is **prefix stability**: scoring a truncated
history must give the same answer for the dates that history contains. A
model that uses the future fails this by construction, which is what makes
it a good test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v6.config import CONFIG
from src.v6.engines.analog import AnalogEngine, _embargo_bounds
from src.v6.engines.regime import RegimeEngine


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("1995-01-02", periods=n)


def _regime_features(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic two-regime data: a calm stretch then a volatile one."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0, 0.5, (n // 2, 5))
    stress = rng.normal(2.0, 1.5, (n - n // 2, 5))
    data = np.vstack([calm, stress])
    return pd.DataFrame(
        data, index=_dates(n),
        columns=["rv_21_z", "rv_63_z", "dd_from_252h", "vix_z", "hy_spread_z"],
    )


class TestRegimeEngineIsCausal:
    """Filtered posteriors must not move when future data is appended."""

    def test_prefix_stability(self):
        features = _regime_features()
        engine = RegimeEngine().fit(features)

        cut = 1000
        full = engine.score(features, h_steps=21)
        prefix = engine.score(features.iloc[:cut], h_steps=21)

        common = prefix.index
        np.testing.assert_allclose(
            full.loc[common, "stress_score"].values,
            prefix["stress_score"].values,
            atol=1e-8,
            err_msg=(
                "regime stress score changed when later data was appended — "
                "the engine is using observations from after each date "
                "(forward-backward smoothing rather than filtering)"
            ),
        )

    def test_filtered_differs_from_smoothed(self):
        """Guards against a silent revert to `predict_proba`."""
        features = _regime_features()
        engine = RegimeEngine().fit(features)
        Xz = engine._score_matrix(features)

        filtered = engine._filtered_posterior(Xz)
        smoothed = engine.model_.predict_proba(Xz)

        assert not np.allclose(filtered, smoothed, atol=1e-6), (
            "filtered and smoothed posteriors are identical, which means "
            "the causal path is not actually being used"
        )
        # Both are still valid distributions.
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-8)
        assert (filtered >= 0).all()

    def test_horizon_changes_the_forward_probability(self):
        features = _regime_features()
        engine = RegimeEngine().fit(features)
        short = engine.score(features, h_steps=5)["h_step_stress_prob"]
        long = engine.score(features, h_steps=252)["h_step_stress_prob"]
        assert not np.allclose(short.values, long.values), (
            "the h-step forward probability must depend on h — the alpha fed "
            "the aggregator a contemporaneous score that ignored the horizon"
        )


class TestAnalogEmbargo:
    def test_embargo_window_covers_the_horizon(self):
        pool = _dates(500)
        query = pd.DatetimeIndex([pool[250]])
        lo, hi = _embargo_bounds(pool, query, embargo_td=63)
        excluded = pool[lo[0]:hi[0]]
        assert pool[250] in excluded, "the query's own date must be excluded"
        # Must span at least +/- 63 trading days around the query.
        assert (pool[250] - excluded.min()).days >= 63
        assert (excluded.max() - pool[250]).days >= 63

    def test_zero_embargo_excludes_nothing(self):
        pool = _dates(100)
        lo, hi = _embargo_bounds(pool, pool, embargo_td=0)
        assert (hi - lo == 0).all()

    def test_neighbours_exclude_the_overlapping_window(self):
        """A scored date must not retrieve its own temporal neighbourhood."""
        n = 1200
        idx = _dates(n)
        rng = np.random.default_rng(3)
        # A slow trend guarantees that a date's nearest neighbours in feature
        # space are its temporal neighbours, which is exactly the leak.
        trend = np.linspace(0, 10, n)
        features = pd.DataFrame(
            {f"f{i}": trend + rng.normal(0, 0.01, n) for i in range(6)},
            index=idx,
        )
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

        engine = AnalogEngine().fit(features, prices, horizons=(21, 63))
        query_date = idx[600]
        result = engine.query(features.loc[query_date], x_pct=5.0, horizon_td=63)

        embargo_days = pd.Timedelta(days=int(np.ceil(63 * 1.6)))
        for d in result.analog_dates:
            assert abs(d - query_date) > embargo_days, (
                f"analog {d.date()} is inside the embargo window of "
                f"{query_date.date()}; its forward outcome overlaps the "
                f"query's and leaks the answer"
            )

    def test_confidence_is_not_destroyed_by_self_matching(self):
        """The alpha's d_1/d_k read ~0 whenever a near-identical date existed."""
        n = 900
        idx = _dates(n)
        rng = np.random.default_rng(4)
        features = pd.DataFrame(
            {f"f{i}": rng.normal(size=n) for i in range(6)}, index=idx
        )
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        engine = AnalogEngine().fit(features, prices, horizons=(21, 63))
        out = engine.query_dataframe(features, x_pct=5.0, horizon_td=63)
        conf = out["confidence"].dropna()
        assert conf.median() > 0.05, (
            f"analog support collapsed to {conf.median():.4f} at the median; "
            "self-matches or overlapping neighbours are still dominating"
        )
        assert conf.max() <= 1.0 and conf.min() >= 0.0


class TestTunableThreshold:
    """The design's core promise: one fit serves every (x, h) query."""

    def test_one_fit_answers_many_thresholds(self):
        n = 1200
        idx = _dates(n)
        rng = np.random.default_rng(5)
        features = pd.DataFrame(
            {f"f{i}": rng.normal(size=n) for i in range(6)}, index=idx
        )
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=idx)
        engine = AnalogEngine().fit(features, prices, horizons=(21, 63, 126, 252))

        probs = {
            x: engine.query_dataframe(features, x_pct=x, horizon_td=63)["prob"].mean()
            for x in (2.0, 5.0, 10.0, 20.0)
        }
        values = list(probs.values())
        assert all(a >= b for a, b in zip(values, values[1:])), (
            f"probability must fall as the threshold rises; got {probs}"
        )

    def test_unsupported_horizon_is_rejected(self):
        n = 700
        idx = _dates(n)
        rng = np.random.default_rng(6)
        features = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)}, index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        engine = AnalogEngine().fit(features, prices, horizons=(21, 63))
        with pytest.raises(ValueError, match="horizon_td"):
            engine.query(features.iloc[300], x_pct=5.0, horizon_td=999)
