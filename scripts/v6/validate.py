"""Walk-forward + BLIND validation harness for the v6 Crash KPI Engine.

Protocol (design doc §6):

- **Walk-forward folds.** Re-fit at each cutoff in ``WALK_FORWARD_FOLDS``.
  For each fold's out-of-sample window, score with the model fit through
  that cutoff only.
- **BLIND.** Single-shot fit through ``BLIND_START - 1``, score from
  ``BLIND_START`` forward. No retuning.

For each (x_pct, horizon_td) this reports:

- days scored, gate fires, and the **base rate** of the event, so a reader
  can tell a hard problem from an empty label set;
- precision / recall / F1 of gate fires against the forward-drawdown label;
- **lead time** from each fire to the drawdown it warned about;
- calibration: Brier score, log loss, reliability slope, and a bucket table;
- a **decision backtest** — CAGR, MaxDD, Sharpe versus buy-and-hold with
  slippage;
- the **full five-part kill-criteria verdict**.

WHAT CHANGED SINCE THE ALPHA
============================
1. **The label matches the target.** The alpha scored against crash-episode
   *onsets* from the peak-to-trough segmenter while the engines were trained
   on *forward maximum drawdown*. Those are different events (see
   ``src/v6/features/labels.py``); both now come from ``crash_label``.

2. **All five kill criteria are evaluated.** The alpha's ``_kill_check``
   tested only criterion 5 (gate fire rate) and reported PASS/FAIL as though
   that were the whole contract. Criteria 1-4 — calibration slope, CAGR
   versus buy-and-hold, MaxDD ratio, and weight degeneracy — were never
   computed. Criteria 2 and 3 need a backtest, which did not exist.

3. **BLIND results are persisted.** The alpha wrote a `walkforward` key only,
   so the scorecard's headline BLIND numbers could not be reproduced from
   any artefact.

Usage
-----
    python scripts/v6/validate.py both --x 10 --h 63
    python scripts/v6/validate.py blind --x 20 --h 126
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
from src.v6.features.labels import crash_label, forward_maxdd

TRADING_DAYS_PER_YEAR = 252
SLIPPAGE_BPS = 5.0          # per switch, per design §6
EXIT_HOLD_TD = 21           # stay defensive this long after the last fire


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------
def _confusion(scores: pd.DataFrame, prices: pd.Series, x_pct: float,
               horizon_td: int) -> Dict[str, float]:
    """Gate fires vs the forward-drawdown label, on comparable dates."""
    idx = scores.index.intersection(prices.index)
    labels = crash_label(prices, x_pct, horizon_td).reindex(idx)
    fires = scores["gate_fires"].reindex(idx).astype(bool)

    # Only dates with a complete forward window can be scored.
    scorable = labels.notna()
    fires_s = fires[scorable]
    y = labels[scorable].astype(bool)

    tp = int((fires_s & y).sum())
    fp = int((fires_s & ~y).sum())
    fn = int((~fires_s & y).sum())
    tn = int((~fires_s & ~y).sum())
    n_fires = int(fires_s.sum())
    n_pos = int(y.sum())
    precision = tp / n_fires if n_fires else float("nan")
    recall = tp / n_pos if n_pos else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if n_fires and n_pos and (precision + recall) > 0 else float("nan")
    )
    return {
        "n_days": int(len(idx)),
        "n_scorable": int(scorable.sum()),
        "n_fires": int(fires.sum()),
        "n_crash_days": n_pos,
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "fire_pct": 100.0 * fires.sum() / max(len(idx), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Lift over always-firing: >1 means the gate beats the base rate.
        "precision_lift": (
            precision / float(y.mean()) if n_fires and y.mean() > 0 else float("nan")
        ),
    }


def _lead_times(scores: pd.DataFrame, prices: pd.Series, x_pct: float,
                horizon_td: int) -> Dict[str, float]:
    """Trading days from each gate fire to the drawdown crossing x_pct.

    Answers "how early was the warning", which precision alone cannot. A
    negative median would mean the gate fires only once the fall is underway
    — useful as confirmation, not as a forecast.
    """
    idx = scores.index.intersection(prices.index)
    fires = scores["gate_fires"].reindex(idx).astype(bool)
    fire_dates = idx[fires]
    if len(fire_dates) == 0:
        return {"n_fires_with_event": 0, "median_lead_td": float("nan"),
                "mean_lead_td": float("nan")}

    px = prices.reindex(idx).astype(float)
    positions = {d: i for i, d in enumerate(idx)}
    values = px.values
    leads: List[int] = []
    for d in fire_dates:
        i = positions[d]
        window = values[i : i + horizon_td + 1]
        if len(window) < 2 or np.isnan(window).any():
            continue
        running_peak = np.maximum.accumulate(window)
        dd = (window / running_peak - 1.0) * -100.0
        crossed = np.flatnonzero(dd >= x_pct)
        if crossed.size:
            leads.append(int(crossed[0]))
    if not leads:
        return {"n_fires_with_event": 0, "median_lead_td": float("nan"),
                "mean_lead_td": float("nan")}
    return {
        "n_fires_with_event": len(leads),
        "median_lead_td": float(np.median(leads)),
        "mean_lead_td": float(np.mean(leads)),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _calibration(scores: pd.DataFrame, prices: pd.Series, x_pct: float,
                 horizon_td: int, n_bins: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Reliability table plus Brier / log-loss / reliability slope."""
    idx = scores.index.intersection(prices.index)
    labels = crash_label(prices, x_pct, horizon_td).reindex(idx)
    p = pd.to_numeric(scores["posterior_mean"].reindex(idx), errors="coerce")
    ok = labels.notna() & p.notna()
    p, y = p[ok], labels[ok]
    if len(p) == 0:
        return pd.DataFrame(), {}

    bins = pd.cut(p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = (
        pd.DataFrame({"p": p, "y": y})
        .groupby(bins, observed=True)
        .agg(predicted_mean=("p", "mean"), empirical_rate=("y", "mean"), n=("y", "size"))
    )

    brier = float(np.mean((p - y) ** 2))
    eps = 1e-9
    pc = np.clip(p, eps, 1 - eps)
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    # Reliability slope: regress empirical rate on predicted, weighted by bin
    # size. Kill criterion 1 wants this within [0.5, 1.5] — 1.0 is perfect
    # calibration, below 0.5 means the posterior barely tracks reality.
    valid = table.dropna()
    if len(valid) >= 2 and valid["predicted_mean"].std() > 0:
        slope, intercept = np.polyfit(
            valid["predicted_mean"], valid["empirical_rate"], 1, w=valid["n"]
        )
    else:
        slope, intercept = float("nan"), float("nan")
    return table, {
        "brier": brier,
        "log_loss": logloss,
        "reliability_slope": float(slope),
        "reliability_intercept": float(intercept),
        "base_rate": float(y.mean()),
        # Brier skill vs always predicting the base rate: >0 beats the prior.
        "brier_skill_score": float(
            1.0 - brier / np.mean((y.mean() - y) ** 2)
        ) if np.mean((y.mean() - y) ** 2) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Decision backtest (kill criteria 2 and 3)
# ---------------------------------------------------------------------------
def _backtest(scores: pd.DataFrame, prices: pd.Series,
              exit_hold_td: int = EXIT_HOLD_TD,
              slippage_bps: float = SLIPPAGE_BPS) -> Dict[str, float]:
    """Convert gate fires into an exit/re-entry rule and price the result.

    Rule, fixed in advance: hold the index; on a gate fire move to cash and
    stay there while any fire occurred within the last `exit_hold_td` days.
    Each switch costs `slippage_bps`. Cash earns nothing, which is
    conservative for the strategy in a high-rate era.
    """
    idx = scores.index.intersection(prices.index)
    if len(idx) < 2:
        return {}
    px = prices.reindex(idx).astype(float)
    ret = px.pct_change().fillna(0.0)
    fires = scores["gate_fires"].reindex(idx).fillna(False).astype(bool)

    # Defensive while a fire is inside the trailing hold window. Shifted by
    # one day: a signal on day t can only be acted on at t+1's close.
    defensive = fires.rolling(exit_hold_td, min_periods=1).max().astype(bool).shift(1).fillna(False)
    exposure = (~defensive).astype(float)
    switches = exposure.diff().abs().fillna(0.0)
    cost = switches * (slippage_bps / 10_000.0)
    strat_ret = exposure * ret - cost

    def _stats(r: pd.Series, label: str) -> Dict[str, float]:
        curve = (1.0 + r).cumprod()
        years = len(r) / TRADING_DAYS_PER_YEAR
        cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
        vol = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if r.std() > 0 else float("nan")
        downside = r[r < 0].std()
        sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside and downside > 0 else float("nan")
        maxdd = float(((curve / curve.cummax()) - 1.0).min() * 100.0)
        return {f"{label}_cagr_pct": 100 * cagr, f"{label}_vol_pct": 100 * vol,
                f"{label}_sharpe": sharpe, f"{label}_sortino": sortino,
                f"{label}_maxdd_pct": maxdd}

    out: Dict[str, float] = {}
    out.update(_stats(strat_ret, "strategy"))
    out.update(_stats(ret, "buyhold"))
    out["n_switches"] = int(switches.sum())
    out["time_in_market_pct"] = float(100 * exposure.mean())
    out["cagr_delta_pp"] = out["strategy_cagr_pct"] - out["buyhold_cagr_pct"]
    bh_dd = abs(out["buyhold_maxdd_pct"])
    out["maxdd_ratio"] = abs(out["strategy_maxdd_pct"]) / bh_dd if bh_dd > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Kill criteria — all five
# ---------------------------------------------------------------------------
def _kill_check(metrics: Dict[str, float], calib: Dict[str, float],
                backtest: Dict[str, float],
                weights: Optional[Dict[str, float]] = None
                ) -> Tuple[bool, List[str], List[str]]:
    """Evaluate every pre-declared kill criterion. Returns (pass, fails, notes)."""
    kc = CONFIG.kill
    fails: List[str] = []
    notes: List[str] = []

    # 1. Calibration error / reliability slope in [0.5, 1.5].
    slope = calib.get("reliability_slope", float("nan"))
    if np.isfinite(slope):
        if not (0.5 <= slope <= 1.5):
            fails.append(f"[1] reliability slope {slope:.3f} outside [0.5, 1.5]")
        else:
            notes.append(f"[1] reliability slope {slope:.3f} OK")
    else:
        notes.append("[1] reliability slope not computable (too few bins)")

    # 2. CAGR vs buy-and-hold.
    delta = backtest.get("cagr_delta_pp", float("nan"))
    if np.isfinite(delta):
        if delta < kc.min_cagr_delta_vs_bh_pp:
            fails.append(
                f"[2] CAGR {delta:+.2f}pp vs B&H < {kc.min_cagr_delta_vs_bh_pp}pp"
            )
        else:
            notes.append(f"[2] CAGR {delta:+.2f}pp vs B&H OK")
    else:
        notes.append("[2] CAGR delta not computable")

    # 3. MaxDD ratio vs buy-and-hold.
    ratio = backtest.get("maxdd_ratio", float("nan"))
    if np.isfinite(ratio):
        if ratio > kc.max_maxdd_ratio_vs_bh:
            fails.append(
                f"[3] MaxDD ratio {ratio:.3f} > {kc.max_maxdd_ratio_vs_bh}"
            )
        else:
            notes.append(f"[3] MaxDD ratio {ratio:.3f} OK")
    else:
        notes.append("[3] MaxDD ratio not computable")

    # 4. Weight degeneracy — one engine carrying everything.
    if weights:
        top = max(weights.values())
        if top >= 0.90:
            fails.append(f"[4] weight degenerate: max engine weight {top:.3f}")
        else:
            notes.append(f"[4] max engine weight {top:.3f} OK")
    else:
        notes.append("[4] weights unavailable")

    # 5. Gate fire rate inside the admissible band.
    fire_pct = metrics.get("fire_pct", 0.0)
    if fire_pct > kc.max_gate_fire_pct:
        fails.append(f"[5] gate fires {fire_pct:.2f}% > max {kc.max_gate_fire_pct}%")
    elif fire_pct < kc.min_gate_fire_pct:
        fails.append(f"[5] gate fires {fire_pct:.2f}% < min {kc.min_gate_fire_pct}%")
    else:
        notes.append(f"[5] gate fire rate {fire_pct:.2f}% OK")

    return (len(fails) == 0, fails, notes)


def _report_block(title: str, metrics: Dict, calib_tbl: pd.DataFrame,
                  calib: Dict, backtest: Dict, leads: Dict) -> None:
    print(f"\n=== {title} ===")
    if metrics:
        print(f"  days={metrics['n_days']} scorable={metrics['n_scorable']} "
              f"fires={metrics['n_fires']} ({metrics['fire_pct']:.2f}%)")
        print(f"  base_rate={metrics['base_rate']:.3f} "
              f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} lift={metrics['precision_lift']:.2f}x")
    if leads and np.isfinite(leads.get("median_lead_td", float("nan"))):
        print(f"  median lead to -{metrics.get('x_pct', '')}% "
              f"= {leads['median_lead_td']:.0f} trading days "
              f"({leads['n_fires_with_event']} fires led to the event)")
    if calib:
        print(f"  Brier={calib['brier']:.4f} (skill {calib['brier_skill_score']:+.3f}) "
              f"logloss={calib['log_loss']:.4f} slope={calib['reliability_slope']:.3f}")
    if backtest:
        print(f"  BACKTEST strategy CAGR={backtest['strategy_cagr_pct']:.2f}% "
              f"MaxDD={backtest['strategy_maxdd_pct']:.1f}% "
              f"Sharpe={backtest['strategy_sharpe']:.2f}")
        print(f"           buy&hold CAGR={backtest['buyhold_cagr_pct']:.2f}% "
              f"MaxDD={backtest['buyhold_maxdd_pct']:.1f}% "
              f"Sharpe={backtest['buyhold_sharpe']:.2f}")
        print(f"           delta={backtest['cagr_delta_pp']:+.2f}pp  "
              f"maxdd_ratio={backtest['maxdd_ratio']:.3f}  "
              f"in-market={backtest['time_in_market_pct']:.1f}%  "
              f"switches={backtest['n_switches']}")
    if len(calib_tbl):
        print("  Calibration:")
        print(calib_tbl.round(3).to_string().replace("\n", "\n    "))


def _evaluate(scores: pd.DataFrame, prices: pd.Series, x_pct: float,
              horizon_td: int, weights: Optional[Dict[str, float]] = None,
              title: str = "") -> Dict:
    """Full metric bundle for one scored window."""
    metrics = _confusion(scores, prices, x_pct, horizon_td)
    metrics["x_pct"] = x_pct
    leads = _lead_times(scores, prices, x_pct, horizon_td)
    calib_tbl, calib = _calibration(scores, prices, x_pct, horizon_td)
    backtest = _backtest(scores, prices)
    passes, fails, notes = _kill_check(metrics, calib, backtest, weights)
    if title:
        _report_block(title, metrics, calib_tbl, calib, backtest, leads)
        print(f"\n  Kill-criteria: {'PASS' if passes else 'FAIL'}")
        for f in fails:
            print(f"    FAIL {f}")
        for n in notes:
            print(f"    ok   {n}")
    return {
        "metrics": metrics,
        "lead_times": leads,
        "calibration_summary": calib,
        "calibration_table": calib_tbl.reset_index().astype(str).to_dict(orient="records"),
        "backtest": backtest,
        "kill_passes": bool(passes),
        "kill_fails": fails,
        "kill_notes": notes,
    }


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
    last_weights: Optional[Dict[str, float]] = None
    for i, cutoff in enumerate(folds):
        cutoff_ts = pd.Timestamp(cutoff)
        next_cutoff = pd.Timestamp(folds[i + 1]) if i + 1 < len(folds) else prices_full.index.max()
        print(f"\n=== Walk-forward fold {i+1}: fit < {cutoff_ts.date()} "
              f"-> score {cutoff_ts.date()} .. {next_cutoff.date()} ===")
        fit_through = (cutoff_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            pipe = CrashKPIPipeline()
            pipe.fit_until(fit_through)
            scores = pipe.score(
                start=cutoff_ts.strftime("%Y-%m-%d"),
                end=next_cutoff.strftime("%Y-%m-%d"),
                x_pct=x_pct, horizon_td=horizon_td,
            )
        except RuntimeError as exc:
            print(f"  Skipping fold: {exc}")
            fold_reports.append({"fold": i + 1, "cutoff": cutoff,
                                 "skipped": str(exc)})
            continue
        if len(scores) == 0:
            print("  (no rows)")
            continue
        if pipe.engine_errors_:
            print(f"  engine issues: {pipe.engine_errors_}")
        last_weights = dict(pipe.aggregator.weights)
        all_scores.append(scores)
        report = _evaluate(scores, prices_full, x_pct, horizon_td,
                           weights=last_weights, title=f"Fold {i+1} ({cutoff})")
        report.update({"fold": i + 1, "cutoff": cutoff,
                       "pipeline": pipe.summary()})
        fold_reports.append(report)

    # Fold windows share their boundary date, so pooling can duplicate it.
    pooled = pd.concat(all_scores) if all_scores else pd.DataFrame()
    if len(pooled):
        pooled = pooled[~pooled.index.duplicated(keep="first")].sort_index()
    pooled_report = (
        _evaluate(pooled, prices_full, x_pct, horizon_td, weights=last_weights,
                  title="Pooled walk-forward")
        if len(pooled) else {}
    )
    return {
        "x_pct": x_pct, "horizon_td": horizon_td,
        "fold_reports": fold_reports,
        "pooled": pooled_report,
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
    print(f"\n=== BLIND: fit < {blind.date()} -> score {blind.date()} .. "
          f"{prices_full.index.max().date()} ===")

    pipe = CrashKPIPipeline()
    pipe.fit_until(fit_through)
    scores = pipe.score(start=blind.strftime("%Y-%m-%d"),
                        x_pct=x_pct, horizon_td=horizon_td)
    if len(scores) == 0:
        print("  No BLIND rows.")
        return {}

    report = _evaluate(scores, prices_full, x_pct, horizon_td,
                       weights=dict(pipe.aggregator.weights), title="BLIND")
    fires = scores[scores["gate_fires"].fillna(False)]
    if len(fires):
        print(f"\n  Fire dates ({len(fires)}):")
        for d in fires.index:
            print(f"    {d.date()} p={scores.loc[d, 'posterior_mean']:.3f} "
                  f"conf={scores.loc[d, 'confidence']:.3f}")
    report.update({
        "x_pct": x_pct, "horizon_td": horizon_td,
        "fire_dates": [str(d.date()) for d in fires.index],
        "pipeline": pipe.summary(),
    })
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _json_default(obj):
    """Make numpy / pandas scalars JSON-serialisable (and NaN -> null)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return str(obj)


def _clean_floats(obj):
    """Recursively replace non-finite floats with None so the JSON is valid.

    The alpha's artefact contained bare `NaN` tokens, which is not valid JSON
    and made the file unreadable by any strict parser.
    """
    if isinstance(obj, dict):
        return {k: _clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_floats(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["walkforward", "blind", "both"],
                    default="both", nargs="?")
    ap.add_argument("--x", type=float, default=10.0, help="crash threshold (percent)")
    ap.add_argument("--h", type=int, default=63, help="horizon (trading days)")
    ap.add_argument("--out", type=str, default=None, help="json output path")
    args = ap.parse_args()

    report: Dict = {"x_pct": args.x, "horizon_td": args.h}
    if args.mode in ("walkforward", "both"):
        report["walkforward"] = walk_forward(args.x, args.h)
    if args.mode in ("blind", "both"):
        report["blind"] = blind_evaluate(args.x, args.h)

    out_path = (
        Path(args.out) if args.out
        else ARTIFACTS_DIR / f"v6_validation_x{int(args.x)}_h{args.h}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_clean_floats(report), f, indent=2, default=_json_default)
    print(f"\nReport written to {out_path}")

    blind = report.get("blind") or {}
    if blind:
        print(f"BLIND verdict: {'PASS' if blind.get('kill_passes') else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
