"""
Honest test: does the bottom-finder model improve v5's re-entry?

Holds v5 ALARM SIGNAL fixed (already validated). Compares two re-entry strategies:
  A) v5 original: alarm-off 10d AND price>MA50 AND 5% rally off post-alarm low
  B) v5 + bottom-finder: alarm-off 5d AND (bot_signal>=th OR price>MA50)

bot_threshold chosen on TUNE only (1999-2020), reported on BLIND (2021-2026).
"""
import sqlite3, json
import pandas as pd, numpy as np

DB='data/market_crash.db'
COST_BPS = 5

conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds = pd.read_sql('SELECT * FROM predictions', conn, parse_dates=['prediction_date'])
conn.close()

eq = ind['nasdaq_close'].ffill().dropna()
sig_v5 = preds[preds.model_version=='v5'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
sig_bot= preds[preds.model_version=='v5.1_bot'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()

# Use v5's original alarm config
v5cfg = json.loads(open('data/alarm_config_v5.json').read())
ENTER=v5cfg['entry']; EXITT=v5cfg['exit']; MIN_DUR=v5cfg['min_dur']; MAX_DUR=v5cfg['max_dur']
print(f'v5 alarm config: entry={ENTER} exit={EXITT} min_dur={MIN_DUR} max_dur={MAX_DUR}')

def hysteresis(p, idx, en, ex, mn, mx):
    a=[]; on=False; si=None; pv=p.values
    for i in range(len(pv)):
        if not on:
            if pv[i]>=en: on=True; si=i
        else:
            d=i-si
            if pv[i]<ex or d>=mx:
                if d>=mn: a.append((idx[si], idx[i-1]))
                on=False
    if on and si is not None and (len(pv)-si)>=mn: a.append((idx[si], idx[-1]))
    return a

alms = hysteresis(sig_v5, sig_v5.index, ENTER, EXITT, MIN_DUR, MAX_DUR)
am = pd.Series(False, index=sig_v5.index)
for s,e in alms: am.loc[s:e] = True
print(f'v5 alarms: {len(alms)} episodes, {am.sum()} alarm-days ({am.mean()*100:.1f}%)')

# --- Strategy A: v5 original re-entry ---
def strat_v5_orig(prices, am, cost_bps=COST_BPS):
    rets = prices.pct_change().fillna(0)
    ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos=True; days_off=0; post_low=None; eqv=[1.0]; cost=cost_bps/10000.0
    for i in range(1, len(prices)):
        prev=in_pos
        if am.iloc[i]: in_pos=False; days_off=0; post_low=prices.iloc[i]
        else:
            days_off = days_off+1 if not in_pos else days_off
            if not in_pos:
                post_low = min(post_low, prices.iloc[i]) if post_low else prices.iloc[i]
                if days_off>=10 and prices.iloc[i]>ma50.iloc[i] and prices.iloc[i]>=post_low*1.05:
                    in_pos=True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1]*(1+r))
    return pd.Series(eqv, index=prices.index)

# --- Strategy B: v5 + bottom-finder re-entry ---
def strat_v5_bot(prices, am, bot, bot_th, cost_bps=COST_BPS, min_off=5):
    rets = prices.pct_change().fillna(0)
    ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos=True; days_off=0; eqv=[1.0]; cost=cost_bps/10000.0
    for i in range(1, len(prices)):
        prev=in_pos
        if am.iloc[i]:
            in_pos=False; days_off=0
        else:
            days_off = days_off+1 if not in_pos else days_off
            if not in_pos and days_off >= min_off:
                bs = bot.iloc[i]
                if (not pd.isna(bs) and bs >= bot_th) or (prices.iloc[i] > ma50.iloc[i]):
                    in_pos=True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1]*(1+r))
    return pd.Series(eqv, index=prices.index)

def metrics(eqv, start=None, end=None):
    if start: eqv = eqv.loc[pd.Timestamp(start):]
    if end:   eqv = eqv.loc[:pd.Timestamp(end)]
    if len(eqv) < 2: return None
    eqv = eqv / eqv.iloc[0]
    yrs=(eqv.index[-1]-eqv.index[0]).days/365.25
    cagr=(eqv.iloc[-1])**(1/max(yrs,1e-6))-1
    rets=eqv.pct_change().dropna()
    sh = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    mdd=(eqv/eqv.cummax()-1).min()
    return {'cagr':cagr,'sharpe':sh,'mdd':mdd,'final':eqv.iloc[-1]}

eqA = strat_v5_orig(eq, am)
print('\nGrid of bot_threshold (chosen on TUNE 2000-2020):')
print(f'{"th":>5} {"min_off":>8} | {"TUNE CAGR":>10} {"Sharpe":>7} {"MaxDD":>7} | {"BLIND CAGR":>11} {"Sharpe":>7} {"MaxDD":>7}')
results=[]
for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    for mo in [3, 5, 10]:
        eqB = strat_v5_bot(eq, am, sig_bot, th, min_off=mo)
        mt = metrics(eqB, '2000-01-01','2020-12-31')
        mb = metrics(eqB, '2021-01-01', None)
        results.append({'th':th,'min_off':mo,'tune':mt,'blind':mb,'eq':eqB})
        print(f'{th:>5.2f} {mo:>8} | {mt["cagr"]*100:>9.1f}% {mt["sharpe"]:>6.2f} {mt["mdd"]*100:>6.1f}% | {mb["cagr"]*100:>10.1f}% {mb["sharpe"]:>6.2f} {mb["mdd"]*100:>6.1f}%')

# Choose best on TUNE Sharpe
best = max(results, key=lambda r: r['tune']['sharpe'])
print(f'\nChosen on TUNE Sharpe: bot_threshold={best["th"]}  min_off={best["min_off"]}')

# Final comparison
print('\n'+'='*80)
print('FINAL COMPARISON')
print('='*80)
def line(name, eqv):
    print(f'\n{name}')
    for label, s, e in [('FULL  2000-2026','2000-01-01',None),
                         ('TUNE  2000-2020','2000-01-01','2020-12-31'),
                         ('BLIND 2021-2026','2021-01-01',None)]:
        m = metrics(eqv, s, e)
        if m: print(f'  {label:18s}  CAGR={m["cagr"]*100:>5.1f}%  Sharpe={m["sharpe"]:.2f}  MaxDD={m["mdd"]*100:>6.1f}%  Final={m["final"]:.2f}x')

line('A) v5 + ORIGINAL re-entry (MA50 + 5% rally + 10d)', eqA)
line('B) v5 + BOTTOM-FINDER re-entry (bot or MA50, after 5d off)', best['eq'])
line('   Buy & Hold', eq)

# Save best config
with open('data/v5_bot_reentry_config.json','w') as f:
    json.dump({'bot_threshold': best['th'], 'min_off_days': best['min_off']}, f, indent=2)
print('\nSaved data/v5_bot_reentry_config.json')
