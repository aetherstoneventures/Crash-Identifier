# Quick-Start Guide

## TL;DR

```bash
./run.sh
```

That single command sets up the environment, runs the full v5 pipeline,
and launches the Streamlit dashboard at <http://localhost:8501>.

When `venv/` already exists you'll be prompted:

```
Choose mode:
  [1] FRESH  — delete venv, recreate, reinstall requirements
  [2] REUSE  — keep existing venv, skip pip install (default)
```

Press Enter (or `2`) for the fast path; pick `1` after upgrading
dependencies.

## Prerequisites

- macOS or Linux
- Python 3.9–3.12 (`brew install python@3.11` on macOS)
- Git
- ~2 GB free disk for the venv + cached data

## What gets run

| Step | Script | Purpose |
| ---- | ------ | ------- |
| 1 | `scripts/data/collect_data.py` | Pull macro + market data → SQLite |
| 2 | `scripts/data/populate_crash_events.py` | Label historical 15%+ DDs |
| 3a | `scripts/training/train_statistical_model_v3.py` | Train StatV3 |
| 3b | `scripts/utils/generate_predictions_v5.py` | StatV3 predictions → DB |
| 3c | `scripts/training/train_v5_walkforward.py` | **Canonical v5 trainer** |
| 3d | `scripts/training/train_bottom_predictor.py` | Re-entry timing |
| 4 | `scripts/utils/generate_bottom_predictions.py` | Bottom predictions → DB |
| 5 | `scripts/evaluation/evaluate_crash_detection.py` | Honest BLIND scorecard |
| 6 | `streamlit run src/dashboard/app.py` | Open dashboard |

## Non-interactive flags

```bash
./run.sh --fresh             # force fresh venv + reinstall
./run.sh --reuse             # force reuse existing venv
./run.sh --dashboard-only    # skip pipeline, open dashboard immediately
```

## Running the dashboard alone

If the database is already populated:

```bash
./run.sh --dashboard-only
# or
venv/bin/streamlit run src/dashboard/app.py
```

The default landing tab is **🛡️ v5 Production**.

## Running individual steps manually

```bash
source venv/bin/activate
python -W ignore scripts/training/train_v5_walkforward.py
```

Each script is idempotent and may be re-run independently after the
data layer (`collect_data.py` + `populate_crash_events.py`) has been
populated at least once.

## Tests

```bash
venv/bin/pytest -W ignore
```

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `Python 3.9–3.12 required` | `brew install python@3.11`, then re-run |
| `module not found` after upgrade | `./run.sh --fresh` |
| Yahoo Finance 401/429 | already mitigated via `curl_cffi`; just retry |
| Dashboard shows "no data" | ensure `data/market_crash.db` exists; rerun `./run.sh` |

## Where to look next

- [`docs/V5_HONEST_SCORECARD.md`](V5_HONEST_SCORECARD.md) — canonical v5 metrics
- [`docs/METHODOLOGY.md`](METHODOLOGY.md) — full validation methodology
- [`docs/FUTURE_WORK_RESULTS.md`](FUTURE_WORK_RESULTS.md) — kill log for v5.1 / v6 / v5_multi
- [`docs/V6_NEGATIVE_RESULT.md`](V6_NEGATIVE_RESULT.md) — why v6 was shelved
- [`docs/REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md) — data sources & seeds
