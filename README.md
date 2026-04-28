# Crash-Identifier — v5 Production

A walk-forward-validated crash detector for the US equity market (Nasdaq Composite).
The production model **v5** combines an XGBoost classifier with the StatV3
multi-factor risk score (50/50 blend), gated by alarm hysteresis and a
disciplined re-entry rule.

```
BLIND (out-of-sample, 2021-now, frozen config)
─────────────────────────────────────────────────
  Event detection ......... 100%   (2 / 2 crashes caught)
  Day-precision ............ 95.8%
  CAGR ..................... 23.0%   vs B&H ~15%
  Sharpe ................... 1.45    MaxDD -13.2%
  Median lead from peak .... -9d     (near-coincident, by design)
```

> v5 has been frozen at git tag [`v5-BENCHMARK`](https://github.com/aetherstoneventures/Crash-Identifier)
> and mirrored at branch `v5-benchmark-protected`. Three independent
> attempts to improve on it (v5.1 / v6 / v5_multi) failed kill criteria —
> see [docs/FUTURE_WORK_RESULTS.md](docs/FUTURE_WORK_RESULTS.md).

---

## Quick start

```bash
./run.sh
```

The launcher:
- Detects whether `venv/` already exists and asks **FRESH** (recreate +
  reinstall requirements) or **REUSE** (skip pip install).
- Runs the lean pipeline (collect → train StatV3 → train v5 walk-forward →
  bottom predictor → evaluate).
- Launches the Streamlit dashboard at <http://localhost:8501> with the
  **🛡️ v5 Production** tab as the landing page.

Non-interactive flags:

| flag                   | effect                                |
|------------------------|---------------------------------------|
| `./run.sh --fresh`     | force fresh venv + reinstall          |
| `./run.sh --reuse`     | force reuse existing venv             |
| `./run.sh --dashboard-only` | skip the pipeline, open the dashboard |

Requires Python 3.9–3.12 (`brew install python@3.11` on macOS).

---

## Repository layout

```
.
├── run.sh                          # single-command launcher
├── requirements.txt
├── data/
│   ├── alarm_config_v5.json        # frozen v5 hysteresis config
│   ├── experiment_*.json           # research scorecards (A/C/D)
│   ├── v6_kill_verdict.json        # documented v6 negative result
│   └── market_crash.db             # SQLite — populated by the pipeline
├── models/
│   ├── statistical_v3/             # StatV3 saved model
│   └── v5/v5_final.pkl             # production XGBoost classifier
├── scripts/
│   ├── data/                       # collect_data, populate_crash_events,
│   │                               # fetch_v5_features, purge_bad_rows
│   ├── training/                   # train_statistical_model_v3,
│   │                               # train_v5_walkforward (canonical),
│   │                               # train_bottom_predictor
│   ├── utils/                      # generate_predictions_v5,
│   │                               # generate_bottom_predictions,
│   │                               # v5_backtest
│   ├── evaluation/                 # evaluate_crash_detection,
│   │                               # evaluate_bottom_predictions
│   ├── research/                   # experiments A / C / D / D-round2
│   └── database/migrate_to_postgresql.py
├── src/
│   ├── dashboard/                  # Streamlit app + v5_production page
│   ├── models/                     # crash_prediction, bottom_prediction
│   ├── data_collection/, feature_engineering/, alerts/, …
│   └── utils/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   ├── V5_HONEST_SCORECARD.md      # canonical v5 metrics
│   ├── FUTURE_WORK_RESULTS.md      # v5.1 / v6 / v5_multi kill log
│   ├── V6_NEGATIVE_RESULT.md       # documented v6 shelving
│   ├── HISTORICAL_CRASHES_REFERENCE.md
│   ├── INVESTOR_LAWS.md
│   ├── REPRODUCIBILITY_GUIDE.md
│   ├── CHANGELOG.md
│   └── README.md
└── tests/                          # pytest suite
```

---

## The v5 pipeline (what `run.sh` runs)

1. **`scripts/data/collect_data.py`** — pulls Nasdaq, S&P 500, VIX, FRED
   macro indicators and stores them in `data/market_crash.db`.
2. **`scripts/data/populate_crash_events.py`** — labels historical crash
   episodes (peak-back-walked 15% drawdowns ≥ 30 trading days).
3. **`scripts/training/train_statistical_model_v3.py`** — trains StatV3
   risk-factor model and saves to `models/statistical_v3/`.
4. **`scripts/utils/generate_predictions_v5.py`** — runs StatV3 on the full
   indicator history and writes per-day probabilities to `predictions`.
5. **`scripts/training/train_v5_walkforward.py`** — *canonical v5 trainer.*
   4-fold nested walk-forward XGBoost training; tunes alarm hysteresis on
   TUNE folds only, evaluates on held-out TEST folds, then writes the
   blended v5 probability series to the DB and freezes
   `data/alarm_config_v5.json`.
6. **`scripts/training/train_bottom_predictor.py`** — re-entry timing model.
7. **`scripts/utils/generate_bottom_predictions.py`** — writes bottom
   predictions to the DB.
8. **`scripts/evaluation/evaluate_crash_detection.py`** — computes the
   honest BLIND scorecard.
9. **Streamlit dashboard** launches with the **🛡️ v5 Production** tab.

---

## Tests

```bash
venv/bin/pytest -W ignore
```

---

## License

See [LICENSE](LICENSE).
