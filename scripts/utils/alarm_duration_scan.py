"""Alarm duration scan — find minimum duration for 50%+ alarm precision.

Uses market_data.in_crash (peak-to-trough, rebuilt in Phase 2) as ground truth
instead of the noisy crash_events table.
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = "data/market_crash.db"

conn = sqlite3.connect(DB_PATH)
pred = pd.read_sql(
    "SELECT prediction_date, crash_probability FROM predictions ORDER BY prediction_date",
    conn, parse_dates=["prediction_date"]
)
# Use in_crash from market_data as ground truth (peak-to-trough, 15.2% rate)
md = pd.read_sql(
    "SELECT date, in_crash FROM indicators ORDER BY date",
    conn, parse_dates=["date"]
)
conn.close()

md = md.set_index("date")
prob = pred.set_index("prediction_date")["crash_probability"]

# Build crash periods from in_crash column
in_crash = md["in_crash"].fillna(0)
crash_periods = []
in_c, c_start = False, None
for dt, v in in_crash.items():
    if not in_c and v == 1:
        in_c, c_start = True, dt
    elif in_c and v != 1:
        crash_periods.append({"start": c_start, "end": dt - pd.Timedelta(days=1)})
        in_c = False
if in_c:
    crash_periods.append({"start": c_start, "end": in_crash.index[-1]})
crash_df = pd.DataFrame(crash_periods)
print(f"Crash periods from in_crash: {len(crash_df)}")
print(crash_df.to_string())
print()

def hysteresis_alarms(p, enter=0.13, exit_t=0.10, min_dur=30):
    """Use searchsorted for O(1) duration calculation instead of full scan."""
    idx = p.index
    alarms = []
    in_a, start_pos = False, None
    for i, (dt, v) in enumerate(p.items()):
        if not in_a:
            if v >= enter:
                in_a, start_pos = True, i
        else:
            if v < exit_t:
                end_pos = i - 1
                dur = end_pos - start_pos + 1
                if dur >= min_dur:
                    start = idx[start_pos]
                    end = idx[end_pos]
                    p_slice = p.iloc[start_pos:end_pos + 1]
                    alarms.append({
                        "start": start, "end": end, "dur": dur,
                        "max_prob": float(p_slice.max()),
                        "avg_prob": float(p_slice.mean()),
                    })
                in_a = False
    if in_a and start_pos is not None:
        dur = len(p) - start_pos
        if dur >= min_dur:
            start = idx[start_pos]
            p_slice = p.iloc[start_pos:]
            alarms.append({
                "start": start, "end": idx[-1], "dur": dur,
                "max_prob": float(p_slice.max()),
                "avg_prob": float(p_slice.mean()),
            })
    return pd.DataFrame(alarms) if alarms else pd.DataFrame()

def check_alarm_precision(alm_df, crash_df_periods, lead_days=180):
    """Alarm confirmed if it overlaps OR precedes a crash period within lead_days."""
    confirmed = 0
    for _, a in alm_df.iterrows():
        # Case 1: alarm overlaps crash period
        overlap = (
            (crash_df_periods["start"] <= a["end"]) &
            (crash_df_periods["end"] >= a["start"])
        )
        # Case 2: crash starts within lead_days after alarm start
        lead = (
            (crash_df_periods["start"] >= a["start"]) &
            (crash_df_periods["start"] <= a["start"] + pd.Timedelta(days=lead_days))
        )
        if (overlap | lead).any():
            confirmed += 1
    return confirmed / len(alm_df) if len(alm_df) > 0 else 0, confirmed

print("Alarm Duration Scan (enter=0.13, exit=0.10, lead=180d)")
print(f"{'MinDur':8s} {'Alarms':8s} {'Conf':6s} {'Prec':8s}")
print("-" * 40)
for min_d in [5, 10, 15, 20, 25, 30, 45, 60]:
    alm = hysteresis_alarms(prob, enter=0.13, exit_t=0.10, min_dur=min_d)
    if alm.empty:
        print(f"{min_d:5d}d     0      0     N/A")
        continue
    prec, conf = check_alarm_precision(alm, crash_df)
    print(f"{min_d:5d}d  {len(alm):6d}  {conf:4d}  {prec:6.0%}")

# Detailed view at 30d
print("\n--- Detail at min_dur=30d ---")
alm30 = hysteresis_alarms(prob, enter=0.13, exit_t=0.10, min_dur=30)
print(f"{'Start':12s} {'End':12s} {'Dur':5s} {'MaxP':6s} {'Tag'}")
print("-" * 70)
for _, a in alm30.iterrows():
    horizon = a["start"] + pd.Timedelta(days=180)
    tag = "FALSE ALARM"
    for _, cp in crash_df.iterrows():
        if (cp["start"] <= a["end"] and cp["end"] >= a["start"]) or \
           (a["start"] <= cp["start"] <= horizon):
            tag = f"CRASH {str(cp['start'].date())}"
            break
    print(f"{str(a['start'].date()):12s} {str(a['end'].date()):12s} "
          f"{a['dur']:4d}d {a['max_prob']:.1%}  {tag}")

prec30, conf30 = check_alarm_precision(alm30, crash_df)
if len(alm30) > 0:
    print(f"\n30d result: {conf30}/{len(alm30)} = {prec30:.0%} alarm precision")

# Day-level metrics at 30d
alarm_days = pd.Series(0, index=in_crash.index)
for _, a in alm30.iterrows():
    alarm_days.loc[a["start"]:a["end"]] = 1
common = alarm_days.index.intersection(in_crash.index)
al = alarm_days.reindex(common)
ic = in_crash.reindex(common).fillna(0)
tp = ((al == 1) & (ic == 1)).sum()
fp = ((al == 1) & (ic == 0)).sum()
fn = ((al == 0) & (ic == 1)).sum()
dp = tp / (tp + fp) if (tp + fp) > 0 else 0
dr = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Day-level precision={dp:.1%}  recall={dr:.1%}")
