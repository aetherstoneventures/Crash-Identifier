"""Walk-forward + BLIND validation harness for the v6 Crash KPI Engine.

Protocol (locked in design doc, section 6):
- Walk-forward folds: re-fit at each cutoff in WALK_FORWARD_FOLDS
  (default: < 1999, < 2005, < 2012, < 2020). For each fold's OOS window,
  use the model that was fit through that cutoff to score the window.
- BLIND: single-shot fit through (BLIND_START - 1) and score from
  BLIND_START forward.

For each (x_pct, horizon_td) combination passed in, we report:
- Total trading days scored
- Number of gate fires
- Number of true crash episodes (extracted with that x_pct over horizon)
- Precision / recall of gate fires vs crash-episode dates (anywhere in
  the next horizon_td days from a fire)
- Calibration: bucket posteriors into deciles, plot empirical hit rate
- Kill-criteria check (config.KillCriteria)

Usage:
    python scripts/v6/walkforward.py --x 10 --h 63
    python scripts/v6/blind_eval.py --x 10 --h 63
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v6.config import CONFIG, WALK_FORWARD_FOLDS, BLIND_START, ARTIFACTS_DIR
from src.v6.pipeline import CrashKPIPipeline
from src.v6.features import FeatureBuilder
from src.v6.features.crash_extractor import extract_crashes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _crash_day_labels(prices: pd.Series, x_pct: float, horizon_td: int) -> pd.Series:
    """Boolean Series: True on date d if there exists a crash episode of
    severity >= x_pct that BEGINS within the next horizon_td trading days."""
    eps = extract_crashes(prices, x_pct=x_pct)
    label = pd.Series(False, index=prices.index)
    if not eps:
        return label
    starts = pd.DatetimeIndex([e.peak_date for e in eps])
    for d in label.index:
        future_idx = label.index.get_indexer([d])[0]
        if future_idx < 0:
            continue
        window_end_pos = min(future_idx + horizon_td, len(label.index) - 1)
        window_end_date = label.index[window_end_pos]
        if ((starts >= d) & (starts <= window_end_date)).any():
            label.iloc[future_idx] = True
    return label


def _confusion(scores: pd.DataFrame, prices: pd.Series, x_pct: float,
               horizon_td: int) -> Dict[str, float]:
    fires = scores["gate_fires"].astype(bool)
    px = prices.loc[scores.index.intersection(prices.index)]
    fires = fires.loc[px.index]
    labels = _crash_day_labels(px, x_pct, horizon_td).loc[px.index]
    tp = int((fires & labels).sum())
    fp = int((fires & ~labels).sum())
    fn = int((~fires & labels).sum())
    tn = int((~fires & ~labels).sum())
    fires_n = int(fires.sum())
    crashes_n = int(labels.sum())
    precision = tp / fires_n if fires_n > 0 else float("nan")
    recall = tp / crashes_n if crashes_n > 0 else float("nan")
    return {
        "n_days": int(len(fires)),
        "n_fires": fires_n,
        "n_crash_days": crashes_n,
        "fire_pct": 100.0 * fires_n / max(len(fires), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def _calibration(scores: pd.DataFrame, prices: pd.Series, x_pct: float,
                 horizon_td: int, n_bins: int = 10) -> pd.DataFrame:
    px = prices.loc[scores.index.intersection(prices.index)]
    labels = _crash_day_labels(px, x_pct, horizon_td).loc[px.index]
    p = scores["posterior_mean"].loc[px.index]
    bins = pd.cut(p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grp = pd.DataFrame({"p": p, "y": labels.astype(int)}).groupby(bins, observed=True)
    out = grp.agg(predicted_mean=("p", "mean"), empirical_rate=("y", "mean"), n=("y", "size"))
    return out


def _kill_check(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    kc = CONFIG.kill
    fails: List[str] = []
    fire_pct = metrics.get("fire_pct", 0.0)
    if fire_pct > kc.max_gate_fire_pct:
        fails.append(f"gate fires {fire_pct:.2f}% > max {kc.max_gate_fire_pct}%")
    if fire_pct < kc.min_gate_fire_pct:
        fails.append(f"gate fires {fire_pct:.2f}% < min {kc.min_gate_fire_pct}%")
    return (len(fails) == 0, fails)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def walk_forward(x_pct: float, horizon_td: int,
                 folds: Optional[List[str]] = None) -> Dict:
    folds = folds or list(WALK_FORWARD_FOLDS)
    fb = FeatureBuilder()
    feats_full = fb.build()
    prices_full = feats_full["_price"]

    all_scores: List[pd.DataFrame] = []
    fold_reports = []
    for i, cutoff in enumerate(folds):
        cutoff_ts = pd.Timestamp(cutoff)
        next_cutoff = pd.Timestamp(folds[i + 1]) if i + 1 < len(folds) else prices_full.index.max()
        print(f"\n=== Walk-forward fold {i+1}: fit < {cutoff_ts.date()} -> score {cutoff_ts.date()} .. {next_cutoff.date()} ===")
        fit_through = (cutoff_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            pipe = CrashKPIPipeline()
            pipe.fit_until(fit_through)
        except RuntimeError as e:
            print(f"  Skipping fold: {e}")
            continue
        scores = pipe.score(
            start=cutoff_ts.strftime("%Y-%m-%d"),
            end=next_cutoff.strftime("%Y-%m-%d"),
            x_pct=x_pct, horizon_td=horizon_td,
        )
        if len(scores) == 0:
            print("  (no rows)")
            continue
        all_scores.append(scores)
        m = _confusion(scores, prices_full, x_pct, horizon_td)
        print(f"  days={m['n_days']} fires={m['n_fires']} ({m['fire_pct']:.2f}%) "
              f"crash_days={m['n_crash_days']} prec={m['precision']:.3f} recall={m['recall']:.3f}")
        fold_reports.append({"fold": i + 1, "cutoff": cutoff, **m})

    pooled = pd.concat(all_scores) if all_scores else pd.DataFrame()
    pooled_metrics = _confusion(pooled, prices_full, x_pct, horizon_td) if len(pooled) else {}
    cal = _calibration(pooled, prices_full, x_pct, horizon_td) if len(pooled) else pd.DataFrame()
    print("\n=== Pooled walk-forward ===")
    if pooled_metrics:
        print(f"  days={pooled_metrics['n_days']} fires={pooled_metrics['n_fires']} "
              f"({pooled_metrics['fire_pct']:.2f}%) precision={pooled_metrics['precision']:.3f} "
              f"recall={pooled_metrics['recall']:.3f}")
    print("\nCalibration:")
    print(cal.round(3).to_string())
    return {
        "x_pct": x_pct, "horizon_td": horizon_td,
        "fold_reports": fold_reports,
        "pooled": pooled_metrics,
        "calibration": cal.reset_index().astype(str).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# BLIND
# ---------------------------------------------------------------------------
def blind_evaluate(x_pct: float, horizon_td: int) -> Dict:
    fb = FeatureBuilder()
    feats_full = fb.build()
    prices_full = feats_full["_price"]
    blind = pd.Timestamp(BLIND_START)
    fit_through = (blind - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n=== BLIND: fit < {blind.date()} -> score {blind.date()} .. {prices_full.index.max().date()} ===")
    pipe = CrashKPIPipeline()
    pipe.fit_until(fit_through)
    scores = pipe.score(
        start=blind.strftime("%Y-%m-%d"),
        x_pct=x_pct, horizon_td=horizon_td,
    )
    if len(scores) == 0:
        print("  No BLIND rows.")
        return {}
    m = _confusion(scores, prices_full, x_pct, horizon_td)
    cal = _calibration(scores, prices_full, x_pct, horizon_td)
    passes, fails = _kill_check(m)
    print(f"  days={m['n_days']} fires={m['n_fires']} ({m['fire_pct']:.2f}%) "
          f"crash_days={m['n_crash_days']} precision={m['precision']:.3f} recall={m['recall']:.3f}")
    print("Calibration:")
    print(cal.round(3).to_string())
    print(f"\nKill-criteria: {'PASS' if passes else 'FAIL'}")
    for f in fails:
        print(f"  - {f}")
    if m["n_fires"] > 0:
        fires = scores[scores["gate_fires"]]
        print(f"\nFire dates ({len(fires)}):")
        for d in fires.index:
            print(f"  {d.date()} p={scores.loc[d, 'posterior_mean']:.3f}")
    return {
        "x_pct": x_pct, "horizon_td": horizon_td,
        "metrics": m,
        "calibration": cal.reset_index().astype(str).to_dict(orient="records"),
        "kill_passes": passes,
        "kill_fails": fails,
        "fire_dates": [str(d.date()) for d in scores[scores["gate_fires"]].index],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["walkforward", "blind", "both"], default="both", nargs="?")
    ap.add_argument("--x", type=float, default=10.0, help="crash threshold (percent)")
    ap.add_argument("--h", type=int, default=63, help="horizon (trading days)")
    ap.add_argument("--out", type=str, default=None, help="json output path")
    args = ap.parse_args()

    report: Dict = {"x_pct": args.x, "horizon_td": args.h}
    if args.mode in ("walkforward", "both"):
        report["walkforward"] = walk_forward(args.x, args.h)
    if args.mode in ("blind", "both"):
        report["blind"] = blind_evaluate(args.x, args.h)

    out_path = Path(args.out) if args.out else ARTIFACTS_DIR / f"v6_validation_x{int(args.x)}_h{args.h}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
