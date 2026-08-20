"""Centralised configuration for the v6 Crash KPI Engine.

All hyperparameters, paths, and validation boundaries live here so the
five engines, aggregator, and validation harness share one source of truth.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "market_crash.db"
ARTIFACTS_DIR = DATA_DIR / "v6_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Walk-forward + BLIND boundaries (locked, matching prior v5 protocol for
# direct comparability)
# ---------------------------------------------------------------------------
WALK_FORWARD_FOLDS: Tuple[str, ...] = (
    "1999-01-01",
    "2005-01-01",
    "2012-01-01",
    "2020-01-01",
)
BLIND_START = "2021-01-01"

# ---------------------------------------------------------------------------
# Tunable user query defaults (these are inference-time parameters,
# NOT training-time. Engine 3 is trained once and serves all (x, h) pairs.)
# ---------------------------------------------------------------------------
DEFAULT_X_PCT: float = 10.0          # drawdown threshold (percent)
DEFAULT_HORIZON_DAYS: int = 63       # forecast horizon (trading days)
SUPPORTED_X_PCT: Tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
SUPPORTED_HORIZON_DAYS: Tuple[int, ...] = (21, 63, 126, 252)

# Trading-day length used for episode duration filtering in the crash
# extractor. Conservative: a "crash" must persist at least this long.
DEFAULT_MIN_CRASH_DURATION_TD: int = 5

# ---------------------------------------------------------------------------
# Engine 1 — Density anomaly detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnomalyConfig:
    """Mahalanobis + IsolationForest ensemble."""
    iso_n_estimators: int = 200
    iso_contamination: float = 0.05
    refit_every_days: int = 252  # annual re-fit
    normal_buffer_td: int = 20    # exclude ±N days around past crashes
    normal_buffer_x_pct: float = 10.0  # define "normal" as not within buffer of >=x% drawdown
    alert_threshold: float = 0.99


# ---------------------------------------------------------------------------
# Engine 2 — Regime-switching HMM
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeConfig:
    """Gaussian HMM on reduced macro/vol feature set."""
    n_states: int = 4
    refit_every_days: int = 63    # quarterly re-fit
    em_iterations: int = 100
    random_state: int = 42


# ---------------------------------------------------------------------------
# Engine 3 — Analog matcher
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalogConfig:
    """Weighted-Mahalanobis k-NN (LMNN-optional) with empirical forward CDF."""
    k_neighbors: int = 50
    refit_every_days: int = 252
    use_lmnn: bool = False   # If True and metric-learn installed, use LMNN
    min_distance_eps: float = 1e-6
    # Query dates per distance-matrix batch. Trades peak memory against the
    # number of BLAS calls; 512 x ~14k pool is ~55 MB per batch.
    query_batch_size: int = 512
    # Suppress pool dates within +/- horizon of the query, whose forward
    # windows overlap the query's and would leak its outcome. See
    # analog._embargo_bounds. Disable only to reproduce the alpha's numbers.
    embargo_horizons: bool = True


# ---------------------------------------------------------------------------
# Engine 4 — Causal / structural factor model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CausalConfig:
    """Dynamic factor model + Granger causality network."""
    n_factors: int = 5
    granger_lag: int = 5
    rolling_window_td: int = 252
    refit_every_days: int = 63


# ---------------------------------------------------------------------------
# Engine 5 — Bayesian aggregator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AggregatorConfig:
    """Calibrated log-odds pooling with per-engine Beta-Binomial calibration."""
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    # Weight bounds. The cap enforces kill criterion 4 ("one engine carries
    # all weight") structurally; it must be >= 1/n_engines to be satisfiable.
    min_weight: float = 0.10
    max_weight: float = 0.45
    # Exponent applied to measured skill before normalising into weights.
    # Below 1.0 this compresses skill differences, so the one engine whose
    # output is natively on-target cannot swamp the pool. See
    # CrashKPIAggregator._weights_from_skill.
    skill_temper: float = 0.5
    # Log-odds units of inter-engine dispersion that halve the agreement
    # component of confidence.
    agreement_scale: float = 1.5
    # Final recalibration of the pooled posterior against training outcomes:
    # 'platt' (2-parameter logistic), 'isotonic' (free monotone), or 'none'.
    # See CrashKPIAggregator / PosteriorCalibrator for the measured
    # comparison behind this default.
    posterior_calibration: str = "platt"


# ---------------------------------------------------------------------------
# Layer 1/2/3 gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateConfig:
    """Simultaneous-agreement gate ('100% agreement' constraint).

    Every condition must hold on the same day — that requirement is the
    point of the gate and is not relaxed. What changed after the alpha is
    how each layer is *measured*.

    **Persistence.** Macro regime stress and tactical stress move on
    different clocks: credit and labour conditions deteriorate over months,
    volatility and dispersion spike over days. Comparing both to a threshold
    on the identical trading day asks two different processes to peak in the
    same tick. Each layer is therefore evaluated as a rolling maximum over
    its own natural window — "is this layer currently active", not "did this
    layer print its high today". Measured lag-correlations back this up:
    L1 leads L2 (corr 0.53 at 0 lag, 0.41 at -63d).

    **Calibrated thresholds.** Layer thresholds are fitted on the TRAINING
    window as quantiles of each layer's own distribution, chosen so the
    joint fire rate lands inside the kill-criteria band. A hard-coded 1.5σ
    is arbitrary once composites are correlated; a target fire rate is the
    quantity that is actually pre-declarable. The z-thresholds below remain
    the fallback when auto-tuning is disabled or has too little data.
    """
    posterior_threshold: float = 0.60
    confidence_threshold: float = 0.50
    layer1_z_threshold: float = 1.5
    layer2_z_threshold: float = 1.5
    layer3_dd_threshold: float = 2.0  # percent, current drawdown trigger

    # Rolling windows over which a layer counts as "currently active".
    layer1_persistence_td: int = 63   # macro regime: slow-moving state
    layer2_persistence_td: int = 10   # tactical stress: fast spikes

    # Threshold auto-tuning on the training window.
    auto_tune: bool = True
    # A tuned posterior threshold is never allowed below this, nor below the
    # training base rate: acting on a probability lower than the
    # unconditional event rate is not a warning.
    min_posterior_threshold: float = 0.35
    target_fire_rate: float = 0.010          # 1% of training days
    tune_quantile_grid: Tuple[float, ...] = (
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    )


# ---------------------------------------------------------------------------
# Kill criteria (pre-declared, single-shot evaluation on BLIND >= 2021-01-01)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KillCriteria:
    max_calibration_error_pp: float = 10.0   # reliability slope in [0.5, 1.5]
    min_cagr_delta_vs_bh_pp: float = -2.0
    max_maxdd_ratio_vs_bh: float = 1.10
    max_gate_fire_pct: float = 10.0
    min_gate_fire_pct: float = 0.10


# ---------------------------------------------------------------------------
# Top-level config bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class V6Config:
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    analog: AnalogConfig = field(default_factory=AnalogConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    kill: KillCriteria = field(default_factory=KillCriteria)


CONFIG = V6Config()
