"""Phase 6: Entry threshold + min-duration scan to find 50%+ alarm precision."""
import sqlite3
import pandas as pd
import numpy as np

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

# Build crash periods from in_crash ground truth
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

# Keep only historically meaningful crashes (>=15 days, before mid-2025)
all_crash_df["dur_days"] = (all_crash_df["end"] - all_crash_df["start"]).dt.days
hist_crash = all_crash_df[
    (all_crash_df["start"] <= pd.Timestamp("2025-06-01")) &
    (all_crash_df["dur_days"] >= 15)
].reset_index(drop=True)

print(f"Historical crashes to detect ({len(hist_crash)}):")
for _, r in hist_crash.iterrows():
    print(f"  {r['start'].date()} -> {r['end'].date()} ({int(r['dur_days'])}d)")
print()


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
                        "dur": dur, "max_prob": float(p_vals[sp:i].max())
                    })
                in_a = False
    if in_a and sp is not None:
        dur = len(p_vals) - sp
        if dur >= min_dur:
            alarms.append({
                "start": p_idx[sp], "end": p_idx[-1],
                "dur": dur, "max_prob": float(p_vals[sp:].max())
            })
    return pd.DataFrame(alarms) if alarms else pd.DataFrame()


def alarm_precision(alm, cpdf, lead=180):
    if alm.empty:
        return 0.0, 0, 0
    n_conf = 0
    for _, a in alm.iterrows():
        h = a["start"] + pd.Timedelta(days=lead)
        for _, cp in cpdf.iterrows():
            overlap = cp["start"] <= a["end"] and cp["end"] >= a["start"]
            future_lead = a["start"] <= cp["start"] <= h
            if overlap or future_lead:
                n_conf += 1
                break
    return n_conf / len(alm), n_conf, len(alm)


def crash_detection(alm, cpdf):
    if alm.empty or cpdf.empty:
        return 0.0, 0
    n_det = 0
    for _, cp in cpdf.iterrows():
        for _, a in alm.iterrows():
            overlap = a["start"] <= cp["end"] and a["end"] >= cp["start"]
            lead_before = a["start"] < cp["start"]
            lead_after = cp["start"] <= a["start"] + pd.Timedelta(days=180)
            if overlap or (lead_before and lead_after):
                n_det += 1
                break
    return n_det / len(cpdf), n_det


p_raw = pred["crash_probability"].values
p_sm = pred["crash_probability_smooth"].fillna(method="bfill").values
p_idx = pred.index

print("Entry  ExitT MinD Signal   Alarms Conf Prec  Det%")
print("-" * 58)
results = []
for entry in [0.13, 0.15, 0.18, 0.20]:
    for exit_t in [0.08, 0.10, 0.12]:
        for min_d in [10, 20, 30, 45]:
            for label, p in [("raw", p_raw), ("smooth", p_sm)]:
                alm = hysteresis_alarms(p, p_idx, enter=entry, exit_t=exit_t, min_dur=min_d)
                prec, conf, total = alarm_precision(alm, hist_crash)
                det, n_det = crash_detection(alm, hist_crash)
                results.append({
                    "entry": entry, "exit_t": exit_t, "min_d": min_d,
                    "signal": label, "alarms": total, "conf": conf,
                    "prec": prec, "det": det
                })
                flag = " ***" if prec >= 0.50 and det >= 0.75 else ""
                print(f"{entry:.2f}  {exit_t:.2f}  {min_d:2d}d {label:6s} "
                      f"{total:5d}  {conf:3d}  {prec:.0%}  {det:.0%}{flag}")

print()
good = [r for r in results if r["prec"] >= 0.50 and r["det"] >= 0.70]
if good:
    best = max(good, key=lambda x: x["prec"] * 0.4 + x["det"] * 0.6)
    print(f"BEST (prec>=50%, det>=70%): entry={best['entry']}, "
          f"exit={best['exit_t']}, min_d={best['min_d']}d, "
          f"signal={best['signal']}, prec={best['prec']:.0%}, det={best['det']:.0%}")
else:
    best = max(results, key=lambda x: x["prec"] * 0.4 + x["det"] * 0.6)
    print(f"BEST AVAILABLE: entry={best['entry']}, exit={best['exit_t']}, "
          f"min_d={best['min_d']}d, signal={best['signal']}, "
          f"prec={best['prec']:.0%}, det={best['det']:.0%}")
