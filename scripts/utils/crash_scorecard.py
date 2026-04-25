"""Crash-by-crash detection scorecard for v5.

For each historical crash event (rolling-252-max DD>=15%, >=30TD), report:
  - Was an alarm raised between (peak - 90d) and (trough + 30d)?
  - First alarm date relative to peak (negative = before peak = real prediction)
  - Drawdown captured at first alarm (how much loss had already happened)
  - Drawdown saved if you exit on first alarm (relative to trough)
"""
import sqlite3, sys, pickle, json
sys.path.insert(0, '.')
import pandas as pd, numpy as np

DB = 'data/market_crash.db'
cfg = json.loads(open('data/alarm_config_v5.json').read())
ENTRY = cfg['entry']; EXIT = cfg['exit']
MIN_DUR = cfg['min_dur']; MAX_DUR = cfg['max_dur']; MF_MIN = cfg['mf_min']

# Load v5 predictions and indicators
conn = sqlite3.connect(DB)
preds = pd.read_sql("SELECT prediction_date, crash_probability FROM predictions WHERE model_version='v5' ORDER BY prediction_date",
                    conn, parse_dates=['prediction_date']).set_index('prediction_date')['crash_probability']
ind = pd.read_sql('SELECT date, nasdaq_close FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date')
conn.close()

eq = ind['nasdaq_close'].dropna()

# Recompute crashes (same logic as training)
def compute_crash_labels(prices, dd_thresh=0.15, min_td=30):
    prices = prices.ffill(); n=len(prices); pv=prices.values
    rm = prices.rolling(252, min_periods=50).max().values
    below = (pv / rm - 1) <= -dd_thresh
    crash = np.zeros(n, dtype=bool); i=0
    while i<n:
        if not below[i]: i+=1; continue
        rs=i
        while i<n and below[i]: i+=1
        re=i-1
        tp = rs + int(np.argmin(pv[rs:re+1]))
        peak_val = rm[rs]; pp=rs
        for j in range(rs-1, max(0, rs-260), -1):
            if pv[j] >= peak_val * 0.998: pp=j; break
        if tp - pp >= min_td: crash[pp:tp+1]=True
    return pd.Series(crash, index=prices.index)

lbl = compute_crash_labels(eq)

# Need StatV3 elev_counts to apply alarm gate properly
# For scorecard purposes, just use prob >= ENTRY (slightly conservative)
def alarms_in(window_proba, min_dur=MIN_DUR, max_dur=MAX_DUR, enter=ENTRY, exit_t=EXIT):
    alarms=[]; in_a=False; si=None
    arr = window_proba.values; idx = window_proba.index; n=len(arr)
    for i in range(n):
        if not in_a:
            if arr[i] >= enter: in_a=True; si=i
        else:
            d = i - si
            if arr[i] < exit_t or d >= max_dur:
                if d >= min_dur: alarms.append((idx[si], idx[i-1], d))
                in_a=False
    if in_a and d>=min_dur: alarms.append((idx[si], idx[-1], n-si))
    return alarms

# Identify each crash event
periods=[]; in_c=False; cs=None
for dt, v in lbl.items():
    if not in_c and v: in_c, cs = True, dt
    elif in_c and not v:
        periods.append((cs, dt)); in_c=False
if in_c: periods.append((cs, lbl.index[-1]))

events=[]
for s, e in periods:
    seg = eq.loc[s:e]
    peak_dt = seg.idxmax(); peak_val = seg.loc[peak_dt]
    trough_dt = seg.idxmin(); trough_val = seg.loc[trough_dt]
    events.append(dict(start=s, end=e, peak=peak_dt, trough=trough_dt,
                       peak_val=peak_val, trough_val=trough_val,
                       drawdown=trough_val/peak_val-1))

print('='*100)
print(f'CRASH-BY-CRASH SCORECARD (v5, alarm: enter≥{ENTRY:.0%}, exit<{EXIT:.0%}, min_dur={MIN_DUR}d, max_dur={MAX_DUR}d)')
print('='*100)
print(f'\n{"Peak":12s} {"Trough":12s} {"DD":>7s}  {"Alarm 1st":12s} {"d-peak":>7s} {"DD@alarm":>9s} {"Saved":>7s} {"Result":15s}')
print('-'*100)

n_caught_pre = 0   # alarmed BEFORE peak
n_caught_in  = 0   # alarmed during drawdown
n_missed     = 0
total_dd_unhedged = 0.0
total_dd_v5_exit  = 0.0

for ev in events:
    p = ev['peak']; t = ev['trough']
    # Search window: peak - 90 days to trough + 30 days
    win_start = p - pd.Timedelta(days=90)
    win_end   = t + pd.Timedelta(days=30)
    win = preds.loc[win_start:win_end].dropna()
    alms = alarms_in(win)
    if not alms:
        n_missed += 1
        result = 'MISSED'
        d_peak = ''; dd_at = ''; saved = ''
        total_dd_unhedged += ev['drawdown']  # full loss
        total_dd_v5_exit  += ev['drawdown']  # also full loss (no exit)
    else:
        first_alarm = alms[0][0]
        d_peak = (first_alarm - p).days
        # Find equity value on first alarm date (or nearest valid earlier date)
        if first_alarm in eq.index:
            v = eq.loc[first_alarm]
        else:
            v = eq.loc[:first_alarm].iloc[-1]
        dd_at = v / ev['peak_val'] - 1
        # Saved = how much further drop avoided if we exit on alarm
        saved = ev['trough_val'] / v - 1
        if d_peak < 0:
            n_caught_pre += 1; result='CAUGHT (pre-peak)'
        else:
            n_caught_in += 1; result='CAUGHT (in-DD)'
        total_dd_unhedged += ev['drawdown']
        total_dd_v5_exit  += dd_at  # locked in loss at this point
    fa_str = first_alarm.date().isoformat() if alms else '—'
    print(f'{p.date()}  {t.date()}  {ev["drawdown"]:>6.1%}  '
          f'{fa_str:12s} {(str(d_peak)+"d") if alms else "":>7s} '
          f'{(f"{dd_at:>8.1%}") if alms else "":>9s} '
          f'{(f"{saved:>6.1%}") if alms else "":>7s} {result}')

print('-'*100)
print(f'\nDetection summary on {len(events)} historical crashes:')
print(f'  Caught BEFORE peak:    {n_caught_pre}/{len(events)}')
print(f'  Caught DURING drawdown:{n_caught_in}/{len(events)}')
print(f'  MISSED entirely:       {n_missed}/{len(events)}')

if total_dd_unhedged < 0:
    avoided = total_dd_unhedged - total_dd_v5_exit  # negative numbers; v5_exit is less negative if alarms work
    print(f'\nCumulative drawdown if held through all crashes:  {total_dd_unhedged*100:>+7.1f}%')
    print(f'Cumulative drawdown if exited on v5 first alarm:  {total_dd_v5_exit*100:>+7.1f}%')
    print(f'Drawdown avoided by v5 (sum across crashes):       {avoided*100:>+7.1f}% pts')
