"""Causality tests for online recalibration.

`AdaptiveCalibrator` is handed the **full-history** label series, including
labels for dates in the future relative to the row being scored. That is only
legitimate because it consumes them with a strict `t - horizon_td` lag: the
outcome of date `t - h` is fully determined by prices at date `t`, so a reader
sitting at `t` already knows it.

That is a strong claim, and it is exactly the kind of claim that silently
turns into leakage after a refactor. These tests hold it in place:

- **Lag-window arithmetic** — the fit window must never reach past `i - h`.
- **Prefix stability** — scoring a truncated history must reproduce the
  answers for the dates that history contains. A calibrator peeking at future
  labels fails this immediately, because appending data would change earlier
  rows.
- **Label perturbation** — corrupting labels strictly *after* a date must not
  change that date's output. This is the sharpest form of the test: it
  isolates the information channel rather than the arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v6.engines.aggregator import AdaptiveCalibrator, CrashKPIAggregator


HORIZON = 63


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("1990-01-01", periods=n)


def _series(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    posterior = pd.Series(rng.uniform(0.05, 0.95, n), index=idx)
    # Prevalence drifts across the sample: 0.2 early, 0.6 late.
    drift = np.linspace(0.2, 0.6, n)
    y = pd.Series((rng.uniform(0, 1, n) < drift).astype(float), index=idx)
    return posterior, y


class TestLagWindow:
    def test_window_never_reaches_past_the_lag(self):
        """Either the window is empty, or it ends at least `h` in the past."""
        cal = AdaptiveCalibrator(window_td=500)
        for i in (0, 10, 62, 63, 100, 500, 1500):
            lo, hi = cal._fit_window(i, HORIZON)
            assert 0 <= lo <= hi, (
                f"position {i}: window ({lo}, {hi}) is malformed; a negative "
                f"bound would slice as values[0:-h] and read almost "
                f"everything"
            )
            if hi > lo:
                assert hi <= i - HORIZON, (
                    f"fit window at position {i} ends at {hi}, which is not "
                    f"at least {HORIZON} positions in the past"
                )

    def test_early_positions_have_no_usable_window(self):
        cal = AdaptiveCalibrator()
        lo, hi = cal._fit_window(10, HORIZON)
        assert hi <= 0, "no labels can have resolved this early"

    def test_window_is_trailing_not_expanding(self):
        cal = AdaptiveCalibrator(window_td=500)
        lo, hi = cal._fit_window(2000, HORIZON)
        assert hi - lo == 500, (
            "a trailing window is required to track drift; an expanding one "
            "dilutes a regime shift with stale prevalence"
        )


class TestCausality:
    def test_prefix_stability(self):
        """Appending future data must not change earlier answers."""
        posterior, y = _series()
        cut = 1400

        full, _ = AdaptiveCalibrator().apply(posterior, y, HORIZON, 0.3)
        prefix, _ = AdaptiveCalibrator().apply(
            posterior.iloc[:cut], y.iloc[:cut], HORIZON, 0.3
        )

        np.testing.assert_allclose(
            full.iloc[:cut].values, prefix.values, atol=1e-10,
            err_msg=("recalibrated posteriors changed when later data was "
                     "appended — the calibrator is reading the future"),
        )

    def test_base_rate_prefix_stability(self):
        posterior, y = _series()
        cut = 1400
        _, full_br = AdaptiveCalibrator().apply(posterior, y, HORIZON, 0.3)
        _, prefix_br = AdaptiveCalibrator().apply(
            posterior.iloc[:cut], y.iloc[:cut], HORIZON, 0.3
        )
        np.testing.assert_allclose(full_br.iloc[:cut].values, prefix_br.values,
                                   atol=1e-10)

    def test_corrupting_future_labels_changes_nothing_before(self):
        """The sharpest form: perturb labels after date k, check dates <= k."""
        posterior, y = _series()
        k = 1200

        clean, _ = AdaptiveCalibrator().apply(posterior, y, HORIZON, 0.3)

        poisoned = y.copy()
        # Flip every label strictly after k. If any of it reaches a date <= k,
        # that date's output moves.
        poisoned.iloc[k + 1:] = 1.0 - poisoned.iloc[k + 1:]
        dirty, _ = AdaptiveCalibrator().apply(posterior, poisoned, HORIZON, 0.3)

        np.testing.assert_allclose(
            clean.iloc[:k + 1].values, dirty.iloc[:k + 1].values, atol=1e-10,
            err_msg=("corrupting future labels changed a past date's "
                     "posterior — future information is leaking in"),
        )

    def test_corrupting_resolved_labels_does_change_later_output(self):
        """Control: the channel must actually be open for resolved labels.

        Without this, a calibrator that ignored labels entirely would pass
        every test above.
        """
        posterior, y = _series()
        k = 300

        clean, _ = AdaptiveCalibrator().apply(posterior, y, HORIZON, 0.3)
        poisoned = y.copy()
        poisoned.iloc[:k] = 1.0 - poisoned.iloc[:k]
        dirty, _ = AdaptiveCalibrator().apply(posterior, poisoned, HORIZON, 0.3)

        assert not np.allclose(clean.values, dirty.values, atol=1e-10), (
            "corrupting already-resolved labels changed nothing — the "
            "calibrator is not actually using them"
        )


class TestDriftTracking:
    def test_tracks_a_drifting_base_rate(self):
        """The whole point: the live prior must follow prevalence."""
        posterior, y = _series(n=3000)
        _, base_rate = AdaptiveCalibrator(window_td=750).apply(
            posterior, y, HORIZON, 0.3
        )
        early = base_rate.iloc[800:1000].mean()
        late = base_rate.iloc[-500:].mean()
        assert late > early + 0.10, (
            f"base rate should climb with the drift (0.2 -> 0.6); got "
            f"{early:.3f} early vs {late:.3f} late"
        )

    def test_improves_calibration_under_drift(self):
        """Adaptive should beat a single static map when prevalence moves."""
        from src.v6.engines.aggregator import PosteriorCalibrator

        rng = np.random.default_rng(7)
        n = 3000
        idx = _dates(n)
        truth = np.concatenate([np.full(n // 2, 0.2), np.full(n - n // 2, 0.6)])
        posterior = pd.Series(rng.uniform(0.05, 0.95, n), index=idx)
        y = pd.Series((rng.uniform(0, 1, n) < truth).astype(float), index=idx)

        adaptive, _ = AdaptiveCalibrator(window_td=750).apply(
            posterior, y, HORIZON, 0.3
        )
        static = PosteriorCalibrator(method="platt").fit(
            posterior.values[: n // 2], y.values[: n // 2]
        )
        static_out = pd.Series(static.transform(posterior.values), index=idx)

        tail = slice(n // 2 + 800, n)
        brier_adaptive = np.mean((adaptive.values[tail] - y.values[tail]) ** 2)
        brier_static = np.mean((static_out.values[tail] - y.values[tail]) ** 2)
        assert brier_adaptive < brier_static, (
            f"after a prevalence shift the adaptive map should win: "
            f"{brier_adaptive:.4f} vs static {brier_static:.4f}"
        )


class TestAggregatorIntegration:
    def test_labels_are_only_read_through_the_lag(self):
        agg = CrashKPIAggregator()
        idx = _dates(500)
        y = pd.Series(1.0, index=idx)
        agg.set_realized_labels(y, horizon_td=21)
        assert agg.horizon_td_ == 21
        assert agg.realized_y_ is not None

    def test_insufficient_history_falls_back_to_the_prior(self):
        posterior, y = _series(n=300)
        out, br = AdaptiveCalibrator(min_samples=250).apply(
            posterior, y, HORIZON, 0.42
        )
        # Too few resolved labels to fit anything: pass through untouched.
        np.testing.assert_allclose(out.values, posterior.values)
        np.testing.assert_allclose(br.values, 0.42)
