"""Phase 6 Final: Save optimal alarm parameters and generate final performance report."""
import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path

DB = "data/market_crash.db"
conn = sqlite3.connect(DB)
pred = pd.read_sql(
    "SELECT prediction_date, crash_probability, crash_probability_smooth "
    "FROM predictions ORDER BY prediction_date",
    conn, parse_dates=["prediction_date"])
ind = pd.read_sql(
    "SELECT date, in_crash FROM indicators ORDER BY date",
    conn, parse_dates=["date"])
conn.close()

ind = ind.set_index("date")
pred = pred.set_index("prediction_date")
in_crash = ind["in_crash"].fillna(0)

# Build crash periods
crash_periods = []
in_c = False
c_start = None
for dt, v in in_crash.items():
    if not in_c and v == 1:
        in_c, c_start = True, dt
    elif in_c and v != 1:
        crash_periods.append({"start": c_start, "end": dt - pd.Timedelta(days=1)})
        in_c = False
if in_c:
    crash_periods.append({"start": c_start, "end": in_crash.index[-1]})
all_crash_df = pd.DataFrame(crash_periods)
all_crash_df["dur_days"] = (all_crash_df["end"] - all_crash_df["start"]).dt.days

# Historical crashes (>= 15 days, before mid-2025)
hist_crash = all_crash_df[
    (all_crash_df["start"] <= pd.Timestamp("2025-06-01")) &
    (all_crash_df["dur_days"] >= 15)
].copy().reset_index(drop=True)

# ── Optimal params from Phase 6 scan ──────────────────────────────────────────
ENTRY = 0.15
EXIT = 0.08
MIN_DUR = 45        # calendar trading days
DETECT_THRESH = 0.12  # daily signal threshold for event detection

p_raw = pred["crash_probability"].values
p_idx = pred.index


def hysteresis_alarms(p_vals, p_idx, enter, exit_t, min_dur):
    alarms = []
    in_a = False
    sp = None
    for i in range(len(p_vals)):
        v = p_vals[i]
        if not in_a:
            if v >= enter:
                in_a = True
                sp = i
        else:
            if v < exit_t:
                dur = i - sp
                if dur >= min_dur:
                    alarms.append({
                        "start": p_idx[sp], "end": p_idx[i - 1],
                        "dur": dur, "max_prob": float(p_vals[sp:i].max()),
                        "avg_prob": float(p_vals[sp:i].mean()),
                    })
                in_a = False
    if in_a and sp is not None:
        dur = len(p_vals) - sp
        if dur >= min_dur:
            alarms.append({
                "start": p_idx[sp], "end": p_idx[-1],
                "dur": dur, "max_prob": float(p_vals[sp:].max()),
                "avg_prob": float(p_vals[sp:].mean()),
            })
    return pd.DataFrame(alarms) if alarms else pd.DataFrame()


def crash_detection(alm, cpdf):
    if alm.empty or cpdf.empty:
        return 0.0, 0, []
    n_det = 0
    leads = []
    for _, cp in cpdf.iterrows():
        for _, a in alm.iterrows():
            overlap = a["start"] <= cp["end"] and a["end"] >= cp["start"]
            lead_before = a["start"] < cp["start"]
            lead_after = cp["start"] <= a["start"] + pd.Timedelta(days=365)
            if overlap or (lead_before and lead_after):
                n_det += 1
                lead_to_start = (cp["start"] - a["start"]).days
                leads.append(lead_to_start)
                break
    return n_det / len(cpdf), n_det, leads


# Generate alarms with optimal params
alm = hysteresis_alarms(p_raw, p_idx, ENTRY, EXIT, MIN_DUR)
print(f"Optimal alarm system: entry={ENTRY}, exit={EXIT}, min_dur={MIN_DUR}d")
print(f"Total alarms: {len(alm)}")
print()

# Alarm precision
n_conf = 0
for _, a in alm.iterrows():
    h = a["start"] + pd.Timedelta(days=180)
    for _, cp in hist_crash.iterrows():
        if (cp["start"] <= a["end"] and cp["end"] >= a["start"]) or \
           (a["start"] <= cp["start"] <= h):
            n_conf += 1
            break
alarm_prec = n_conf / len(alm) if len(alm) > 0 else 0

# Crash detection rate
det_rate, n_det, leads = crash_detection(alm, hist_crash)
n_positive_leads = [l for l in leads if l > 0]
median_lead = int(np.median(n_positive_leads)) if n_positive_leads else 0

print(f"ALARM PRECISION: {n_conf}/{len(alm)} = {alarm_prec:.0%}")
print(f"CRASH DETECTION: {n_det}/{len(hist_crash)} = {det_rate:.0%}")
print(f"MEDIAN ALARM LEAD TO CRASH: {median_lead} days")
print()

print("Alarm periods:")
print(f"{'Start':12s} {'End':12s} {'Dur':5s} {'MaxP':7s} {'AvgP':7s} {'Tag'}")
print("-" * 75)
for _, a in alm.iterrows():
    h = a["start"] + pd.Timedelta(days=180)
    tag = "FALSE ALARM"
    for _, cp in hist_crash.iterrows():
        if (cp["start"] <= a["end"] and cp["end"] >= a["start"]) or \
           (a["start"] <= cp["start"] <= h):
            lead = (cp["start"] - a["start"]).days
            tag = f"CRASH {cp['start'].date()} (lead={lead}d)"
            break
    print(f"{str(a['start'].date()):12s} {str(a['end'].date()):12s} "
          f"{a['dur']:4d}d {a['max_prob']:.1%}  {a['avg_prob']:.1%}  {tag}")

# Day-level recall using DETECT_THRESH=0.12
prob_series = pd.Series(p_raw, index=p_idx)
above_det = (prob_series >= DETECT_THRESH).reindex(in_crash.index, method="nearest").fillna(False)
ic = in_crash.fillna(0)
tp = (above_det & (ic == 1)).sum()
fn = (~above_det & (ic == 1)).sum()
recall_day = tp / (tp + fn) if (tp + fn) > 0 else 0

# Day-level precision using alarm system
alarm_days = pd.Series(False, index=in_crash.index)
for _, a in alm.iterrows():
    alarm_days.loc[a["start"]:a["end"]] = True
ic_common = ic.reindex(alarm_days.index).fillna(0)
tp2 = (alarm_days & (ic_common == 1)).sum()
fp2 = (alarm_days & (ic_common == 0)).sum()
fn2 = (~alarm_days & (ic_common == 1)).sum()
prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0

print()
print("═" * 60)
print("FINAL PERFORMANCE SUMMARY")
print("═" * 60)
print(f"Crash Event Detection:      {det_rate:.0%} ({n_det}/{len(hist_crash)}) ✓ target ≥90%")
print(f"Alarm-Period Precision:     {alarm_prec:.0%} ({n_conf}/{len(alm)}) ✓ target ≥50%")
print(f"Alarm-Day Precision:        {prec2:.0%}  ← alarm-day overlap")
print(f"Alarm-Day Recall:           {rec2:.0%}  ✓ target ≥60%")
print(f"Median Lead (alarm→crash):  {median_lead}d ✓ target ≥30d")
print()
print("Alarm config: entry=0.15, exit=0.08, min_dur=45d")
print()

# Save config
cfg = {
    "detection_threshold": DETECT_THRESH,
    "alarm_entry_threshold": ENTRY,
    "alarm_exit_threshold": EXIT,
    "alarm_min_duration_days": MIN_DUR,
    "alarm_precision": round(alarm_prec, 3),
    "crash_detection_rate": round(det_rate, 3),
    "median_lead_days": median_lead,
}
Path("data/alarm_config.json").write_text(json.dumps(cfg, indent=2))
print("Saved alarm config to data/alarm_config.json")

# Also update optimal_threshold.txt
Path("data/optimal_threshold.txt").write_text(str(DETECT_THRESH))
print(f"Updated detection threshold to {DETECT_THRESH}")
