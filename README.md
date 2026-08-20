# Crash-Identifier — v6 Crash KPI Engine

A crash detector for the US equity market (Nasdaq Composite, 1971–present).

Five engines score every trading day, a calibrated Bayesian aggregator pools
them into `P(maxDD ≥ x% within h trading days)`, and a simultaneous-agreement
gate turns that into an action signal — with the **crash threshold x% and the
horizon h chosen at query time**, not baked into training.

```
POOLED WALK-FORWARD (1999-2026, x = 10%, h = 63d)
──────────────────────────────────────────────────
  Gate precision .......... 0.859    (2.39× base rate)
  Median lead ............. 40 trading days
  Reliability slope ....... 0.419    ← FAILS kill criterion 1
  CAGR .................... 8.82%    vs 9.08% buy & hold

BLIND (2021-01-01 → 2026-08-19)
  Gate precision .......... 0.750    (2.00× base rate)
  MaxDD ................... -25.7%   vs -36.4% buy & hold
  Sharpe .................. 0.71     vs 0.67
  Kill criteria ........... FAILS (reliability slope 0.330)
```

> **Every window fails its kill criteria.** The gate *ranks* days well —
> precision 0.859 at 2.39× the base rate with a ~40-day lead — but the
> posterior is not a trustworthy probability on any window, the two earliest
> folds never fire, and the `credit_led` archetype has never fired at all.
> The 2021+ window is also **not a clean holdout**: it was inspected during
> development. Read
> [`docs/V6_HONEST_SCORECARD.md`](docs/V6_HONEST_SCORECARD.md) before using
> any number above. The honest case for this system is drawdown reduction at
> comparable return — not excess return, and not calibrated probabilities.

---

## Quick start

```bash
./run.sh
```

The launcher creates or reuses `venv/`, repairs and refreshes the data from
FRED, relabels crash episodes, runs walk-forward + BLIND validation, and opens
the dashboard at <http://localhost:8501>.

| flag | effect |
|---|---|
| `./run.sh --fresh` | recreate venv, reinstall requirements |
| `./run.sh --reuse` | reuse existing venv, skip pip install |
| `./run.sh --dashboard-only` | skip the pipeline, just open the dashboard |
| `./run.sh --skip-backfill` | skip the FRED refresh (offline) |
| `./run.sh --x 20 --h 126` | different crash threshold / horizon |

Requires Python 3.9–3.12 and a free [FRED API
key](https://fred.stlouisfed.org/docs/api/api_key.html) in `.env` as
`FRED_API_KEY` (see `.env.example`).

Ask the engine a question directly:

```python
from src.v6.pipeline import CrashKPIPipeline

pipe = CrashKPIPipeline().fit_until("2020-12-31")

# One fit answers any (x, h). Nothing is retrained between these calls.
mild  = pipe.score(start="2021-01-01", x_pct=5.0,  horizon_td=21)
severe = pipe.score(start="2021-01-01", x_pct=20.0, horizon_td=252)

print(severe[["posterior_mean", "confidence", "archetype", "gate_fires"]].tail())
```

---

## How it works

```
                    FEATURE VECTOR x(t) — 40 features, point-in-time
        math/stats │ macro │ options-implied │ breadth │ geopolitical
                                    │
      ┌───────────┬────────────┬────┴───────┬────────────┐
      ▼           ▼            ▼            ▼            │
  ┌────────┐ ┌────────┐  ┌──────────┐ ┌──────────┐       │
  │ENGINE 1│ │ENGINE 2│  │ ENGINE 3 │ │ ENGINE 4 │       │ features also
  │Density │ │ Regime │  │  Analog  │ │  Causal  │       │ feed the gate
  │anomaly │ │  HMM   │  │  k-NN    │ │ network  │       │
  │        │ │        │  │(tunable  │ │          │       │
  │"wrong?"│ │"regime?"│  │  x, h)  │ │  "why?"  │       │
  └───┬────┘ └───┬────┘  └────┬─────┘ └────┬─────┘       │
      └──────────┴────────────┴────────────┘             │
                             ▼                           │
              ┌──────────────────────────────┐           │
              │ ENGINE 5 — AGGREGATOR        │           │
              │ per-engine calibration       │           │
              │ skill-weighted log-odds pool │           │
              │ P(maxDD ≥ x% in [t, t+h])    │           │
              └──────────────┬───────────────┘           │
                             ▼                           ▼
              ┌──────────────────────────────────────────────┐
              │ LAYER 1/2/3 GATE — all must hold at once     │
              │  L1 macro archetype active (credit / rate /  │
              │     valuation / shock)  ← reports which      │
              │  L2 tactical stress elevated                 │
              │  L3 price confirms                           │
              │  posterior ≥ τ  AND  confidence ≥ κ          │
              └──────────────────────────────────────────────┘
```

**The tunable threshold.** Engine 3 stores realised forward drawdowns for
every supported horizon and thresholds them at query time; the aggregator
calibrates against the training-window realisation of whichever `(x, h)` you
ask for. One fit serves `x ∈ {2, 5, 10, 15, 20, 30}%` and
`h ∈ {21, 63, 126, 252}` trading days.

**Crash archetypes.** A single averaged macro layer only recognises
credit-led crashes, and vetoed everything else — it blocked the gate on every
day of the 2022 bear market because credit and employment were healthy. Layer 1
is therefore scored per archetype (`credit_led`, `rate_led`, `valuation_led`,
`shock_led`) and reports which one fired. Full reasoning:
[`docs/V6_POSTMORTEM.md`](docs/V6_POSTMORTEM.md) §11.

**Leakage discipline.** Macro series are stamped on their real publication
dates via ALFRED vintages; the HMM uses forward filtering, never
forward-backward smoothing; the analog engine embargoes neighbours whose
forward windows overlap the query's. Each is enforced by a test in
`tests/test_v6/test_causality.py`.

---

## Repository layout

```
.
├── run.sh                          # single-command launcher
├── requirements.txt
├── data/
│   ├── market_crash.db             # SQLite — 14 394 rows, 1971-02-05 → 2026-08-19
│   ├── v6_artifacts/               # validation scorecards (JSON)
│   └── backups/                    # pre-repair DB snapshots
├── scripts/
│   ├── data/
│   │   ├── backfill_fred.py        # point-in-time FRED repair + refresh
│   │   ├── collect_data.py         # legacy collector
│   │   └── populate_crash_events.py# crash episode labelling
│   ├── v6/
│   │   ├── validate.py             # walk-forward + BLIND + kill criteria
│   │   └── holdout_eval.py         # single-shot eval against a frozen config
│   └── database/migrate_to_postgresql.py
├── src/v6/
│   ├── config.py                   # every hyperparameter and boundary
│   ├── pipeline.py                 # fit_until() / score() — the entry point
│   ├── freeze.py                   # config hashing / drift detection
│   ├── features/
│   │   ├── builder.py              # 40-feature point-in-time vector
│   │   ├── quality.py              # fabricated-data screen
│   │   ├── labels.py               # the forward-drawdown label (one definition)
│   │   └── crash_extractor.py      # tunable x% episode segmentation
│   └── engines/
│       ├── anomaly.py              # E1 Mahalanobis + IsolationForest
│       ├── regime.py               # E2 Gaussian HMM (forward-filtered)
│       ├── analog.py               # E3 k-NN analog matcher (embargoed)
│       ├── causal.py               # E4 PCA + Diebold-Yilmaz connectedness
│       └── aggregator.py           # E5 calibrated log-odds pool + gate
├── src/dashboard/pages/v6_kpi_engine.py
├── docs/
│   ├── V6_HONEST_SCORECARD.md      # results, failures, and caveats
│   ├── V6_POSTMORTEM.md            # why the alpha failed; is the idea possible?
│   ├── CRASH_KPI_ENGINE_DESIGN.md  # the approved design
│   ├── DATA_SOURCES.md             # every column, source, and PIT rule
│   ├── HISTORICAL_CRASHES_REFERENCE.md
│   ├── INVESTOR_LAWS.md
│   └── CHANGELOG.md
└── tests/
    ├── test_v6/                    # 32 regression + causality tests
    └── test_data_collection/
```

---

## Data

All market and macro data comes from **FRED**. `scripts/data/backfill_fred.py`
is idempotent and prints a before/after coverage report.

Three point-in-time rules are applied, one per series type:

| Rule | Applies to | Behaviour |
|---|---|---|
| `close` | daily market data | known at that day's close |
| `lag` | weekly releases (NFCI, jobless claims) | shifted by the publication lag |
| `vintage` | monthly macro (UNRATE, CPI, M2, INDPRO) | stamped on the **first release date** from ALFRED |

Values go stale after a bounded window rather than being forward-filled
forever, so a discontinued series becomes absent instead of a long constant.
See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

---

## Freezing and honest holdouts

The "no retuning" rule is enforced, not promised. `src/v6/freeze.py` hashes
every decision-affecting setting; `scripts/v6/holdout_eval.py` refuses to run
if the live config drifts from the freeze, or if the window is not strictly
after the lock date.

```bash
# Freeze the current configuration
python -c "from src.v6.freeze import write_freeze; print(write_freeze('6.1.0'))"

# Later, once new data exists
python scripts/v6/holdout_eval.py --freeze data/v6_artifacts/frozen_config_v6.1.0.json
```

## Tests

```bash
venv/bin/pytest -q          # 109 passed, 2 skipped
```

`tests/test_v6/` locks down every defect found in the post-mortem: the
posterior range, the confidence measure, composite standardisation, archetype
reachability, label definitions, the quality screen, weight bounds, HMM
prefix-stability, and the analog embargo.

---

## Previous versions

v5 (XGBoost + StatV3 blend) is frozen at tag `v5-BENCHMARK` and branch
`v5-benchmark-protected`. The pre-v6 tree is at tag `pre-v6-archive`. The
v6.0.0-alpha state, whose failure prompted the current work, is at tag
`v6.0.0-alpha`.

## License

See [LICENSE](LICENSE).
