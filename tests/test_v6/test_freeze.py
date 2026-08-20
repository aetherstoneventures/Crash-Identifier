"""Tests for configuration freezing and posterior recalibration.

The freeze mechanism exists because v6.1 development broke the "no retuning"
promise without anything noticing. These tests check that the mechanism would
actually catch it: drift must be detected, and an unchanged config must not
raise a false alarm. A freeze that cries wolf gets ignored, which is as bad as
one that misses a real change.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import src.v6.freeze as fz
from src.v6.engines.aggregator import PosteriorCalibrator


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A freeze record written from the live configuration."""
    path = tmp_path / "frozen.json"
    fz.write_freeze("test", lock_date="2026-08-19", note="unit test",
                    path=str(path))
    return str(path)


class TestFreezeRoundTrip:
    def test_unchanged_config_matches(self, frozen):
        status = fz.verify_freeze(frozen)
        assert status["matches"], (
            f"an untouched configuration must verify; drift reported: "
            f"{status['differences']}"
        )
        assert status["differences"] == {}

    def test_json_roundtrip_does_not_create_drift(self, frozen):
        """Tuples become lists through JSON; that must not read as a change."""
        record = json.load(open(frozen))
        assert fz.config_hash(record["config"]) == record["config_hash"]

    def test_hash_is_stable_across_calls(self):
        assert fz.config_hash() == fz.config_hash()

    def test_record_carries_lock_date_and_note(self, frozen):
        record = json.load(open(frozen))
        assert record["lock_date"] == "2026-08-19"
        assert record["note"] == "unit test"
        assert len(record["config_hash"]) == 64


class TestDriftDetection:
    def _with_gate(self, monkeypatch, **changes):
        gate = dataclasses.replace(fz.CONFIG.gate, **changes)
        monkeypatch.setattr(fz, "CONFIG", dataclasses.replace(fz.CONFIG, gate=gate))

    def test_threshold_change_is_caught(self, frozen, monkeypatch):
        self._with_gate(monkeypatch, posterior_threshold=0.45)
        status = fz.verify_freeze(frozen)
        assert not status["matches"]
        assert "gate.posterior_threshold" in status["differences"]
        delta = status["differences"]["gate.posterior_threshold"]
        assert delta["live"] == 0.45

    def test_persistence_change_is_caught(self, frozen, monkeypatch):
        self._with_gate(monkeypatch, layer1_persistence_td=999)
        status = fz.verify_freeze(frozen)
        assert not status["matches"]
        assert "gate.layer1_persistence_td" in status["differences"]

    def test_aggregator_change_is_caught(self, frozen, monkeypatch):
        agg = dataclasses.replace(fz.CONFIG.aggregator, posterior_calibration="none")
        monkeypatch.setattr(fz, "CONFIG",
                            dataclasses.replace(fz.CONFIG, aggregator=agg))
        status = fz.verify_freeze(frozen)
        assert not status["matches"]
        assert "aggregator.posterior_calibration" in status["differences"]

    def test_kill_criteria_change_is_caught(self, frozen, monkeypatch):
        """Moving the goalposts must be as visible as moving the model."""
        kill = dataclasses.replace(fz.CONFIG.kill, min_cagr_delta_vs_bh_pp=-99.0)
        monkeypatch.setattr(fz, "CONFIG", dataclasses.replace(fz.CONFIG, kill=kill))
        status = fz.verify_freeze(frozen)
        assert not status["matches"]
        assert "kill.min_cagr_delta_vs_bh_pp" in status["differences"]


class TestPosteriorCalibrator:
    """Platt is the shipped default; isotonic and none stay selectable."""

    def _data(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        # Posterior is systematically over-confident vs the true rate.
        truth = rng.uniform(0, 1, n)
        posterior = np.clip(truth * 1.6, 0.01, 0.99)
        y = (rng.uniform(0, 1, n) < truth).astype(float)
        return posterior, y

    def test_platt_preserves_ordering(self):
        """Monotone by construction — the gate's ranking must be untouched."""
        p, y = self._data()
        cal = PosteriorCalibrator(method="platt").fit(p, y)
        out = cal.transform(p)
        order_in = np.argsort(p)
        order_out = np.argsort(out)
        assert np.array_equal(order_in, order_out), (
            "Platt scaling must not reorder days; gate precision depends on it"
        )

    def test_platt_improves_calibration(self):
        p, y = self._data()
        cal = PosteriorCalibrator(method="platt").fit(p, y)
        assert cal.fitted_
        before = np.mean((p - y) ** 2)
        after = np.mean((cal.transform(p) - y) ** 2)
        assert after < before, (
            f"Brier should improve on a systematically over-confident input: "
            f"{before:.4f} -> {after:.4f}"
        )

    def test_none_is_a_passthrough(self):
        p, y = self._data()
        cal = PosteriorCalibrator(method="none").fit(p, y)
        assert not cal.fitted_
        np.testing.assert_allclose(cal.transform(p), p)

    def test_isotonic_is_available(self):
        p, y = self._data()
        cal = PosteriorCalibrator(method="isotonic").fit(p, y)
        assert cal.fitted_
        out = cal.transform(p)
        assert np.all((out >= 0) & (out <= 1))

    def test_too_little_data_declines_to_fit(self):
        """Better an uncalibrated posterior than one fitted on 20 rows."""
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 20)
        y = (rng.uniform(0, 1, 20) < 0.5).astype(float)
        cal = PosteriorCalibrator(method="platt").fit(p, y)
        assert not cal.fitted_
        np.testing.assert_allclose(cal.transform(p), p)

    def test_single_class_declines_to_fit(self):
        p = np.linspace(0.01, 0.99, 500)
        y = np.ones(500)
        cal = PosteriorCalibrator(method="platt").fit(p, y)
        assert not cal.fitted_
