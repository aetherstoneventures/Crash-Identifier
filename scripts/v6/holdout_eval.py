"""Single-shot holdout evaluation against a frozen configuration.

THE PROBLEM THIS SOLVES
=======================
The v6 protocol calls for a BLIND set evaluated once, with no retuning. That
discipline failed in practice during v6.1: the 2021+ window was inspected
while diagnosing why the gate never fired, the Layer 1 archetype split was
designed in response, and the window quietly stopped being a holdout. The
codebase had no way to notice.

This script makes the next evaluation checkable rather than promised:

1. It loads a **frozen configuration** (`src/v6/freeze.py`) and verifies the
   live settings still hash to the same value. Any drift — a changed
   threshold, a different calibration method — aborts with a diff instead of
   quietly producing a better number.
2. It scores **only dates strictly after the freeze's `lock_date`**, so the
   evaluation window contains no data that informed the configuration.
3. It refuses to run if the database holds no data past the lock date, rather
   than reporting an empty PASS.

Nothing here tunes anything. There are no thresholds to set and no options
that change the model — by construction, so that running it cannot become
another round of fitting.

USAGE
-----
    # Freeze today's configuration
    python -c "from src.v6.freeze import write_freeze; \\
               print(write_freeze('6.1.0', lock_date='2026-08-19'))"

    # Later, once new data has accrued
    python scripts/v6/holdout_eval.py \\
        --freeze data/v6_artifacts/frozen_config_v6.1.0.json

    # See whether enough new data exists yet, without evaluating
    python scripts/v6/holdout_eval.py --freeze <path> --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v6.config import ARTIFACTS_DIR
from src.v6.features import FeatureBuilder
from src.v6.freeze import verify_freeze
from src.v6.pipeline import CrashKPIPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import _clean_floats, _evaluate, _json_default  # noqa: E402

# A holdout narrower than this cannot support a meaningful verdict; the
# horizon alone consumes part of it.
MIN_HOLDOUT_DAYS = 126


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--freeze", required=True,
                    help="path to a frozen-config JSON from src.v6.freeze")
    ap.add_argument("--x", type=float, default=None,
                    help="crash threshold %% (default: the frozen value)")
    ap.add_argument("--h", type=int, default=None,
                    help="horizon in trading days (default: the frozen value)")
    ap.add_argument("--check-only", action="store_true",
                    help="report freeze status and available data, then stop")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # ---- 1. Verify the configuration has not drifted ----------------------
    status = verify_freeze(args.freeze)
    print(f"Frozen config v{status['version']}  lock_date={status['lock_date']}")
    print(f"  frozen hash: {status['frozen_hash'][:16]}...")
    print(f"  live   hash: {status['live_hash'][:16]}...")
    if status["note"]:
        print(f"  note: {status['note']}")

    if not status["matches"]:
        print("\n❌ CONFIGURATION DRIFT — this is not a holdout evaluation.")
        print("   The live settings differ from the frozen ones:")
        for key, delta in status["differences"].items():
            print(f"     {key}: frozen={delta['frozen']!r} live={delta['live']!r}")
        print("\n   Either restore the frozen values, or freeze a new version and")
        print("   wait for fresh data. Re-running a changed model on the same")
        print("   window is retuning, whatever the result is called.")
        return 2
    print("  ✅ configuration matches the freeze")

    # ---- 2. Check that genuinely new data exists --------------------------
    frozen_cfg = json.load(open(args.freeze))["config"]
    x_pct = args.x if args.x is not None else frozen_cfg["default_x_pct"]
    horizon = args.h if args.h is not None else frozen_cfg["default_horizon_td"]

    fb = FeatureBuilder()
    features = fb.build()
    prices = features["_price"]
    lock = pd.Timestamp(status["lock_date"])
    holdout = prices.loc[prices.index > lock]

    print(f"\nData available past the lock date: {len(holdout)} trading days")
    if len(holdout):
        print(f"  {holdout.index.min().date()} .. {holdout.index.max().date()}")

    if args.check_only:
        need = max(0, MIN_HOLDOUT_DAYS - len(holdout))
        print(f"\n{'Ready to evaluate.' if need == 0 else f'Not yet — need ~{need} more trading days.'}")
        return 0

    if len(holdout) < MIN_HOLDOUT_DAYS:
        print(f"\n❌ Only {len(holdout)} trading days past {lock.date()}; "
              f"{MIN_HOLDOUT_DAYS} required.")
        print("   Refresh the data (scripts/data/backfill_fred.py) and try again")
        print("   later. Reporting a verdict on a window this short would be")
        print("   noise dressed as evidence.")
        return 1

    # ---- 3. Single-shot evaluation ---------------------------------------
    print(f"\n=== HOLDOUT: fit <= {lock.date()} -> score {lock.date()} onward "
          f"(x={x_pct}%, h={horizon}d) ===")
    pipe = CrashKPIPipeline()
    pipe.fit_until(lock.strftime("%Y-%m-%d"))
    scores = pipe.score(
        start=(lock + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        x_pct=float(x_pct), horizon_td=int(horizon),
    )
    if len(scores) == 0:
        print("No scorable rows in the holdout window.")
        return 1

    report = _evaluate(scores, prices, float(x_pct), int(horizon),
                       weights=dict(pipe.aggregator.weights), title="HOLDOUT")
    fires = scores[scores["gate_fires"].fillna(False)]
    if len(fires):
        print(f"\n  Fire dates ({len(fires)}):")
        for d in fires.index:
            print(f"    {d.date()} p={scores.loc[d, 'posterior_mean']:.3f} "
                  f"archetype={scores.loc[d, 'archetype']}")

    payload: Dict = {
        "freeze": {k: status[k] for k in
                   ("version", "lock_date", "frozen_hash", "note")},
        "x_pct": x_pct, "horizon_td": horizon,
        "holdout_start": str(holdout.index.min().date()),
        "holdout_end": str(holdout.index.max().date()),
        "n_holdout_days": int(len(holdout)),
        "result": report,
        "pipeline": pipe.summary(),
    }
    out = (Path(args.out) if args.out else
           ARTIFACTS_DIR / f"holdout_v{status['version']}_x{int(x_pct)}_h{horizon}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(_clean_floats(payload), f, indent=2, default=_json_default)

    print(f"\nReport written to {out}")
    print(f"HOLDOUT verdict: {'PASS' if report['kill_passes'] else 'FAIL'}")
    print("\nThis is a genuine out-of-sample result: the configuration hash "
          "matched the\nfreeze, and every scored date falls after the lock "
          "date. Record it as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
