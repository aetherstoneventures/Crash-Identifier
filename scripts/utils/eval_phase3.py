"""Re-evaluate Phase 3 predictions at event-detection-optimized threshold."""
import sys
sys.path.insert(0, ".")
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = "data/market_crash.db"

# Load predictions and crash events
conn = sqlite3.connect(DB_PATH)
pred_df = pd.read_sql(
    "SELECT prediction_date, crash_probability FROM predictions ORDER BY prediction_date",
    conn, parse_dates=["prediction_date"]
)
crash_df = pd.read_sql(
    "SELECT start_date, trough_date, max_drawdown FROM crash_events "
    "WHERE ABS(max_drawdown) >= 10.0 ORDER BY start_date",
    conn, parse_dates=["start_date", "trough_date"]
)
ind_df = pd.read_sql(
    "SELECT date, in_crash, pre_crash_60d FROM indicators ORDER BY date",
    conn, parse_dates=["date"]
)
conn.close()

crash_df = crash_df.drop_duplicates(subset=["start_date", "trough_date"])
prob_series = pred_df.set_index("prediction_date")["crash_probability"]

print(f"Predictions: {len(pred_df)} rows")
print(f"Crash events: {len(crash_df)} unique")
print(f"Prob range: {prob_series.min():.3f} - {prob_series.max():.3f}")


def eval_threshold(thresh):
    results = []
    for _, ev in crash_df.iterrows():
        peak = ev["start_date"]
        trough = ev["trough_date"]
        dd = float(ev["max_drawdown"])
        window = (prob_series.index >= peak - pd.Timedelta(days=120)) & \
                 (prob_series.index <= peak)
        pre = prob_series[window]
        if len(pre) == 0:
            results.append({"peak": str(peak.date()), "dd": dd, "max_prob": None, "det": False})
            continue
        max_prob = float(pre.max())
        first_signal = None
        lead_peak = None
        lead_trough = None
        if max_prob >= thresh:
            first_signal = pre[pre >= thresh].index[0]
            lead_peak = (peak - first_signal).days
            lead_trough = (trough - first_signal).days
        results.append({
            "peak": str(peak.date()), "dd": dd,
            "max_prob": round(max_prob, 4),
            "det": max_prob >= thresh,
            "lead_peak": lead_peak,
            "lead_trough": lead_trough
        })
    n_det = sum(r["det"] for r in results)
    return n_det / len(results), n_det, len(results), results


# Threshold scan
print("\n--- Threshold Scan (event detection rate) ---")
print(f"{'Thresh':8s} {'Det%':8s} {'Det/Tot':10s}")
for t in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.17, 0.20, 0.22, 0.25]:
    rate, det, tot, _ = eval_threshold(t)
    print(f"  {t:.2f}   {rate:.0%}    {det}/{tot}")

# Full detail at 0.13
print("\n--- Full detail at threshold=0.13 ---")
rate13, det13, tot13, results13 = eval_threshold(0.13)
res_df = pd.DataFrame(results13).sort_values("peak")
print(f"{'Peak':12s} {'DD':7s} {'MaxProb':9s} {'LeadPeak':10s} {'LeadTrough':12s} {'Det':4s}")
print("-" * 60)
for _, r in res_df.iterrows():
    det_s = "YES" if r["det"] else " NO"
    mp = f"{r['max_prob']:.1%}" if r["max_prob"] is not None else "N/A"
    lp = f"{r['lead_peak']:.0f}d" if r["lead_peak"] is not None else "N/A"
    lt = f"{r['lead_trough']:.0f}d" if r["lead_trough"] is not None else "N/A"
    print(f"{r['peak']:12s} {r['dd']:7.1f}%  {mp:9s} {lp:10s} {lt:12s} {det_s}")

det_lts = [r["lead_trough"] for r in results13 if r["det"] and r["lead_trough"] is not None]
median_lead = float(np.median(det_lts)) if det_lts else 0
min_lead = float(np.min(det_lts)) if det_lts else 0
print(f"\nDetection rate @ 0.13: {rate13:.0%} ({det13}/{tot13})")
print(f"Median lead to trough: {median_lead:.0f}d | Min: {min_lead:.0f}d")

# Day-level precision/recall at 0.13
target = ((ind_df["pre_crash_60d"] == 1) | (ind_df["in_crash"] == 1)).values.astype(int)
proba = prob_series.reindex(ind_df.set_index("date").index).values
proba = np.where(np.isnan(proba), 0, proba)
preds = (proba >= 0.13).astype(int)
tp = ((preds == 1) & (target == 1)).sum()
fp = ((preds == 1) & (target == 0)).sum()
fn = ((preds == 0) & (target == 1)).sum()
tn = ((preds == 0) & (target == 0)).sum()
prec = tp / (tp + fp) if tp + fp > 0 else 0
rec = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
print(f"\nDay-level @ 0.13: precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

# Current signal
latest = prob_series.iloc[-1]
latest_dt = prob_series.index[-1]
print(f"\nCurrent ({latest_dt.date()}): {latest:.1%} crash probability")
if latest >= 0.13:
    print(f"  WARNING ACTIVE (above threshold 0.13)")
else:
    print(f"  Below threshold 0.13")

# Save optimal threshold
Path("data/optimal_threshold.txt").write_text("0.13")
print("\nSaved threshold 0.13 to data/optimal_threshold.txt")
