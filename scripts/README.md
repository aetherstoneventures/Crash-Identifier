# `scripts/` — pipeline & research

The single entry-point is **[`../run.sh`](../run.sh)** at the repo root.
This directory contains the individual pipeline steps it invokes, plus
research scripts kept for reproducibility.

```
scripts/
├── data/
│   ├── collect_data.py            # FRED + Yahoo macro/market pulls
│   ├── fetch_v5_features.py       # daily v5 feature builder
│   ├── populate_crash_events.py   # labels 15%+ drawdown episodes
│   └── purge_bad_rows.py          # data hygiene
├── training/
│   ├── train_statistical_model_v3.py  # StatV3 risk-factor model
│   ├── train_v5_walkforward.py        # CANONICAL v5 trainer
│   └── train_bottom_predictor.py      # re-entry timing model
├── utils/
│   ├── generate_predictions_v5.py     # populate StatV3 predictions
│   ├── generate_bottom_predictions.py # populate bottom predictions
│   └── v5_backtest.py                 # standalone backtest CLI
├── evaluation/
│   ├── evaluate_crash_detection.py    # honest BLIND scorecard
│   └── evaluate_bottom_predictions.py
├── research/                          # frozen, reproducible experiments
│   ├── experiment_A_vol_gate.py       # NULL effect
│   ├── experiment_C_reentry.py        # MA50 oracle ceiling +0.5pp
│   ├── experiment_D_multi_asset.py    # FAIL kill criteria
│   └── experiment_D_round2_retune.py
└── database/
    └── migrate_to_postgresql.py
```

## Running individual steps

Each script is self-contained and runs from the repo root:

```bash
venv/bin/python3 -W ignore scripts/training/train_v5_walkforward.py
```

## Research

The four `experiment_*.py` scripts produced the JSON verdicts in `data/`
(consumed by the dashboard's "Why v5 stays" panel and documented in
[`../docs/FUTURE_WORK_RESULTS.md`](../docs/FUTURE_WORK_RESULTS.md)). They
are kept verbatim for reproducibility but are NOT part of the
production pipeline.
