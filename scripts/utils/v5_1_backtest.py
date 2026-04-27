"""
V5.1 regime-switching backtest (honest):
  Strategy: LONG NASDAQ default → CASH on alarm → LONG when bottom-finder ≥ THRESH
  Compare:  Buy & Hold, v5 (old), v5.1 (new)

Re-entry by bottom-finder REPLACES the v5 hand-coded "MA50 + 5% rally" rule.
"""
import sqlite3, pickle, json
import pandas as pd, numpy as np

DB = 'data/market_crash.db'
COST_BPS = 5
TUNE_END = pd.Timestamp('2020-12-31')
TEST_START = pd.Timestamp('2021-01-01')

# Load price + signals
conn = sqlite3.connect(DB)
ind  = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds= pd.read_sql('SELECT * FROM predictions', conn, parse_dates=['prediction_date'])
conn.close()
eq = ind['nasdaq_close'].ffill().dropna()
sig_v51 = preds[preds.model_version=='v5.1'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
sig_v5  = preds[preds.model_version=='v5'  ].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
sig_bot = preds[preds.model_version=='v5.1_bot'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()

cfg = json.loads(open('data/alarm_config_v5_1.json').read())
ENTER = cfg['entry']; EXIT = cfg['exit']; MIN_DUR = cfg['min_dur']; MAX_DUR = cfg['max_dur']

def hysteresis(p, idx, enter, exitt, mn, mx):
    a=[]; on=False; si=None; pv=p.values
    for i in range(len(pv)):
        if not on:
            if pv[i] >= enter: on=True; si=i
        else:
            d = i - si
            if pv[i] < exitt or d >= mx:
                if d >= mn: a.append((idx[si], idx[i-1]))
                on = False
    if on and si is not None and (len(pv)-si) >= mn:
        a.append((idx[si], idx[-1]))
    return a

def alarm_days_mask(alarms, idx):
    m = pd.Series(False, index=idx)
    for s,e in alarms: m.loc[s:e] = True
    return m

# --- Find best bottom-finder threshold on TUNE ---
def find_bot_thresh(alarm_mask, prices, bot_signal, tune_end):
    best=(None, -1e18)
    a_in = alarm_mask & (prices.index <= tune_end)
    for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        ret = backtest_two_sided(prices, alarm_mask, bot_signal, th)['equity'].loc[:tune_end]
        score = ret.iloc[-1] / ret.iloc[0]
        if score > best[1]: best = (th, score)
    return best[0]

def backtest_two_sided(prices, alarm_mask, bot_signal, bot_thresh, cost_bps=COST_BPS):
    """Default LONG; on alarm go to CASH; re-enter when alarm OFF for 5 days
    AND (bot_signal >= bot_thresh OR price > 50d MA).  Skip the bot rule pre-2007 if NaN."""
    rets = prices.pct_change().fillna(0)
    ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos = True; cash_idx = None
    eq = [1.0]; pos_track = []
    cost = cost_bps / 10000.0
    days_off_alarm = 0
    for i, dt in enumerate(prices.index):
        if i==0: pos_track.append(True); continue
        prev_pos = in_pos
        if alarm_mask.iloc[i]:
            in_pos = False; days_off_alarm = 0
        else:
            days_off_alarm = days_off_alarm + 1 if not in_pos else days_off_alarm
            if not in_pos and days_off_alarm >= 5:
                bs = bot_signal.iloc[i]; ma_ok = prices.iloc[i] > ma50.iloc[i]
                bot_ok = (not pd.isna(bs)) and (bs >= bot_thresh)
                if bot_ok or ma_ok: in_pos = True
        # apply return
        r = rets.iloc[i] if prev_pos else 0.0
        if prev_pos != in_pos: r -= cost
        eq.append(eq[-1] * (1 + r))
        pos_track.append(in_pos)
    return pd.DataFrame({'equity': eq, 'position': pos_track}, index=prices.index)

def backtest_v5_style(prices, alarm_mask, cost_bps=COST_BPS):
    """Original v5 strategy: re-enter on (alarm-off 10d) AND (price>MA50) AND (5% rally off post-alarm low)."""
    rets = prices.pct_change().fillna(0)
    ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos = True; eq = [1.0]; cost = cost_bps/10000.0
    days_off = 0; post_low = None
    for i, dt in enumerate(prices.index):
        if i==0: continue
        prev = in_pos
        if alarm_mask.iloc[i]:
            in_pos = False; days_off = 0; post_low = prices.iloc[i]
        else:
            days_off = days_off + 1 if not in_pos else days_off
            if not in_pos:
                post_low = min(post_low, prices.iloc[i]) if post_low else prices.iloc[i]
                if days_off >= 10 and prices.iloc[i] > ma50.iloc[i] and prices.iloc[i] >= post_low * 1.05:
                    in_pos = True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eq.append(eq[-1] * (1 + r))
    return pd.Series(eq, index=prices.index)

def backtest_buyhold(prices):
    return prices / prices.iloc[0]

def metrics(eq, name):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/max(yrs,1e-6)) - 1
    rets = eq.pct_change().dropna()
    sharpe = rets.mean()/rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = eq.cummax(); dd = (eq/rolling_max - 1).min()
    return f'{name:10s} CAGR={cagr*100:>5.1f}%  Sharpe={sharpe:.2f}  MaxDD={dd*100:>6.1f}%  Final={eq.iloc[-1]:.2f}x'

# --- v5 alarm reconstruction (load v5 config) ---
v5_cfg = json.loads(open('data/alarm_config_v5.json').read()) if __import__('os').path.exists('data/alarm_config_v5.json') else None
if v5_cfg:
    v5_alms = hysteresis(sig_v5, sig_v5.index, v5_cfg['entry'], v5_cfg['exit'], v5_cfg['min_dur'], v5_cfg['max_dur'])
else:
    v5_alms = hysteresis(sig_v5, sig_v5.index, 0.45, 0.20, 20, 30)
v5_mask = alarm_days_mask(v5_alms, sig_v5.index)

# v5.1 alarms
v51_alms = hysteresis(sig_v51, sig_v51.index, ENTER, EXIT, MIN_DUR, MAX_DUR)
v51_mask = alarm_days_mask(v51_alms, sig_v51.index)

# Find bottom thresh on TUNE only
print('Searching bottom-finder threshold on TUNE (1999-2020) ...')
results = {}
for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    eq_full = backtest_two_sided(eq, v51_mask, sig_bot, th)['equity']
    final_tune = eq_full.loc[:TUNE_END].iloc[-1]
    final_full = eq_full.iloc[-1]
    results[th] = (final_tune, final_full, eq_full)
    print(f'  bot_thresh={th}  tune_end_eq={final_tune:.2f}x  full_eq={final_full:.2f}x')
best_th = max(results, key=lambda k: results[k][0])
print(f'Selected bot_thresh = {best_th} (chosen on tune ONLY)')

# Compute equity curves
v51_eq = backtest_two_sided(eq, v51_mask, sig_bot, best_th)['equity']
v5_eq  = backtest_v5_style(eq, v5_mask)
bh     = backtest_buyhold(eq)

# Windows
for label, start, end in [
    ('FULL  2000-2026', '2000-01-01', None),
    ('TUNE  2000-2020', '2000-01-01', '2020-12-31'),
    ('BLIND 2021-2026', '2021-01-01', None),
    ('Mid-bull 2011-19','2011-01-01','2019-12-31'),
]:
    s = pd.Timestamp(start); e = pd.Timestamp(end) if end else eq.index[-1]
    print(f'\n=== {label} ===')
    print(metrics(bh    .loc[s:e]/bh    .loc[s:e].iloc[0], 'BuyHold'))
    print(metrics(v5_eq .loc[s:e]/v5_eq .loc[s:e].iloc[0], 'v5'))
    print(metrics(v51_eq.loc[s:e]/v51_eq.loc[s:e].iloc[0], 'v5.1'))

# % time in market
print(f'\nv5.1 alarm coverage: {v51_mask.mean()*100:.1f}% of days in CASH')
print(f'v5   alarm coverage: {v5_mask.mean()*100:.1f}% of days in CASH')

# Save
with open('data/v5_1_backtest.json','w') as f:
    out = {
        'bot_thresh': best_th, 'cost_bps': COST_BPS,
        'final': {
            'BH': float(bh.iloc[-1]), 'v5': float(v5_eq.iloc[-1]), 'v5_1': float(v51_eq.iloc[-1]),
        }
    }
    json.dump(out, f, indent=2, default=float)

# Persist last few daily reads for sanity
last30_v51 = v51_eq.iloc[-30:].iloc[[0,-1]].to_dict()
last30_bh  = bh.iloc[-30:].iloc[[0,-1]].to_dict()
print('\nLast 30d equity:'); print('  v5.1:', last30_v51); print('  BH  :', last30_bh)
print('\nDone.')
