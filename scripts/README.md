# `scripts/` — pipeline entry points

The single entry point is **[`../run.sh`](../run.sh)** at the repo root. This
directory holds the individual steps it invokes.

```
scripts/
├── data/
│   ├── backfill_fred.py           # point-in-time FRED repair + refresh
│   ├── collect_data.py            # legacy FRED/Yahoo collector
│   └── populate_crash_events.py   # labels drawdown episodes into crash_events
├── v6/
│   └── validate.py                # walk-forward + BLIND + kill criteria
└── database/
    └── migrate_to_postgresql.py   # optional SQLite -> Postgres migration
```

## The pipeline

```bash
# 1. Repair and refresh the data layer (idempotent; prints coverage deltas)
python scripts/data/backfill_fred.py

# 2. Label historical crash episodes
python scripts/data/populate_crash_events.py

# 3. Validate: walk-forward folds, BLIND, and all five kill criteria
python scripts/v6/validate.py both --x 10 --h 63
```

Step 1 needs `FRED_API_KEY` in `.env`. It is the step that keeps the price
series full-history and stamps macro releases on their real publication dates
— see [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

Step 3 writes `data/v6_artifacts/v6_validation_x{X}_h{H}.json` and prints a
PASS/FAIL verdict per fold and for BLIND.

## Useful variations

```bash
# Different crash definition — no retraining required
python scripts/v6/validate.py blind --x 20 --h 126

# Refresh a single series
python scripts/data/backfill_fred.py --only NASDAQCOM

# See what would change without writing to the database
python scripts/data/backfill_fred.py --dry-run
```

## Removed

`scripts/training/`, `scripts/evaluation/`, `scripts/research/`,
`scripts/forward_risk/` and `scripts/utils/` belonged to v5 and earlier. They
were deleted in commit `1e97ed3` and remain available at git tag
`pre-v6-archive` and branch `v5-benchmark-protected`.
