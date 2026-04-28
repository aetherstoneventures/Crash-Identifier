"""
EXPERIMENT C — Re-entry rule shootout (bottom-finder vs price-only baselines)

Question: among realistic re-entry rules after a v5 crash exit, which gives
the best risk-adjusted return on BLIND? The current v5 rule is "exit on
alarm, re-enter when price > 50d MA after 5 day off-period."

Compare on the SAME v5 alarm signal, same cost (5bps), same window:
  R1: MA50 (current v5)
  R2: MA30 above MA200 (golden-cross style)
  R3: MA50 above MA200
  R4: signal-only — re-enter when v5 prob drops below exit threshold (0.20) for 5d
  R5: oracle (cheat) — re-enter at trough of subsequent drawdown — UPPER BOUND ONLY

Discipline: rules are ALL pre-specified (no tuning). Compare on TUNE and BLIND.
Kill criterion: a rule "wins" only if it beats MA50 on BOTH TUNE CAGR AND BLIND CAGR
                without dropping ev_det. Otherwise MA50 stays.
"""
import sqlite3, json
import numpy as np, pandas as pd

DB = 'data/market_crash.db'
TUNE_END = pd.Timestamp('2020-12-31'); TEST_START = pd.Timestamp('2021-01-01')
COST_BPS = 5

conn = sqlite3.connect(DB)
ind   = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds = pd.read_sql("SELECT * FROM predictions WHERE model_version='v5'", conn, parse_dates=['prediction_date'])
conn.close()
eq = ind['nasdaq_close'].ffill().dropna()
sig = preds.set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
cfg5 = json.loads(open('data/alarm_config_v5.json').read())

def hyst(p, idx, en, ex, mn, mx):
    a = []; on = False; si = None; pv = p.values
    for i in range(len(pv)):
        if not on:
            if pv[i] >= en: on = True; si = i
        else:
            d = i - si
            if pv[i] < ex or d >= mx:
                if d >= mn: a.append((idx[si], idx[i-1]))
                on = False
    if on and si is not None and (len(pv)-si) >= mn: a.append((idx[si], idx[-1]))
    return a

def alarm_mask(alms, idx):
    m = pd.Series(False, index=idx)
    for s, e in alms: m.loc[s:e] = True
    return m

# v5 alarms
alms = hyst(sig, sig.index, cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])
am   = alarm_mask(alms, sig.index)

# Re-entry rules
def reentry_signal(rule, prices, sig, exit_thr=0.20):
    ma30  = prices.rolling(30,  min_periods=15).mean()
    ma50  = prices.rolling(50,  min_periods=20).mean()
    ma200 = prices.rolling(200, min_periods=100).mean()
    if rule == 'R1_MA50':       return prices > ma50
    if rule == 'R2_MA30x200':   return ma30 > ma200
    if rule == 'R3_MA50x200':   return ma50 > ma200
    if rule == 'R4_SIGNAL':     return sig < exit_thr
    if rule == 'R5_ORACLE':     return None  # special — buy trough
    raise ValueError(rule)

def backtest(prices, am, rule, sig, off_period=5, cost_bps=COST_BPS):
    rets = prices.pct_change().fillna(0); cost = cost_bps/10000.0
    if rule == 'R5_ORACLE':
        # Oracle: during off-period after each alarm-end, find the index of
        # min price up to next alarm-start, and re-enter exactly there.
        in_pos = pd.Series(True, index=prices.index)
        i = 0
        while i < len(prices):
            if am.iloc[i]:
                # find end of this alarm episode
                j = i
                while j < len(prices) and am.iloc[j]: j += 1
                # off region [j, next alarm or end)
                k = j
                while k < len(prices) and not am.iloc[k]: k += 1
                lo_idx = j + int(np.argmin(prices.iloc[j:k].values)) if k > j else j
                # off until lo_idx, then in
                in_pos.iloc[i:lo_idx+1] = False
                in_pos.iloc[lo_idx+1:k] = True
                i = k
            else:
                i += 1
        eqv = [1.0]; pos = True
        for t in range(1, len(prices)):
            new_pos = bool(in_pos.iloc[t])
            r = rets.iloc[t] if pos else 0.0
            if new_pos != pos: r -= cost
            pos = new_pos
            eqv.append(eqv[-1]*(1+r))
        return pd.Series(eqv, index=prices.index)
    cond = reentry_signal(rule, prices, sig)
    in_pos = True; days_off = 0; eqv = [1.0]
    for t in range(1, len(prices)):
        prev = in_pos
        if am.iloc[t]: in_pos = False; days_off = 0
        else:
            days_off = days_off+1 if not in_pos else 0
            if not in_pos and days_off >= off_period and bool(cond.iloc[t]): in_pos = True
        r = rets.iloc[t] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1]*(1+r))
    return pd.Series(eqv, index=prices.index)

def m_window(eqv, s, e):
    if s: eqv = eqv.loc[pd.Timestamp(s):]
    if e: eqv = eqv.loc[:pd.Timestamp(e)]
    eqv = eqv/eqv.iloc[0]
    yrs = (eqv.index[-1] - eqv.index[0]).days/365.25
    cagr = eqv.iloc[-1]**(1/max(yrs, 1e-6)) - 1
    rr = eqv.pct_change().dropna()
    sh = rr.mean()/rr.std() * np.sqrt(252) if rr.std() > 0 else 0
    mdd = (eqv/eqv.cummax() - 1).min()
    return dict(cagr=cagr, sharpe=sh, mdd=mdd)

rules = ['R1_MA50','R2_MA30x200','R3_MA50x200','R4_SIGNAL','R5_ORACLE']
print('='*100); print('RE-ENTRY RULE SHOOTOUT (same v5 alarm; 5bp cost; 5d off-period)'); print('='*100)
print(f'\n{"Rule":<14} {"FULL CAGR":>10} {"FULL Sh":>9} {"FULL MDD":>10}  {"TUNE CAGR":>10} {"TUNE Sh":>9}  {"BLIND CAGR":>10} {"BLIND Sh":>9} {"BLIND MDD":>10}')
print('-'*100)
results = {}
for r in rules:
    eqv = backtest(eq, am, r, sig)
    f = m_window(eqv, '2000-01-01', None)
    t = m_window(eqv, '2000-01-01', '2020-12-31')
    b = m_window(eqv, '2021-01-01', None)
    results[r] = {'full':f, 'tune':t, 'blind':b}
    print(f'{r:<14} {f["cagr"]*100:>9.1f}% {f["sharpe"]:>9.2f} {f["mdd"]*100:>9.1f}%  '
          f'{t["cagr"]*100:>9.1f}% {t["sharpe"]:>9.2f}  '
          f'{b["cagr"]*100:>9.1f}% {b["sharpe"]:>9.2f} {b["mdd"]*100:>9.1f}%')

# Decide vs MA50
ma50 = results['R1_MA50']
print('\n--- KILL TEST (must beat MA50 on BOTH TUNE CAGR and BLIND CAGR) ---')
for r in ['R2_MA30x200','R3_MA50x200','R4_SIGNAL']:
    rr = results[r]
    ok_tune  = rr['tune']['cagr']  >= ma50['tune']['cagr']
    ok_blind = rr['blind']['cagr'] >= ma50['blind']['cagr']
    status = 'WIN' if (ok_tune and ok_blind) else 'LOSS'
    print(f'  {r:<14} tune Δ={(rr["tune"]["cagr"]-ma50["tune"]["cagr"])*100:>+5.1f}pp  blind Δ={(rr["blind"]["cagr"]-ma50["blind"]["cagr"])*100:>+5.1f}pp   {status}')

oracle = results['R5_ORACLE']
print(f'\n  ORACLE upper bound:  full CAGR {oracle["full"]["cagr"]*100:.1f}%  blind CAGR {oracle["blind"]["cagr"]*100:.1f}%')
print(f'  MA50 baseline:       full CAGR {ma50["full"]["cagr"]*100:.1f}%  blind CAGR {ma50["blind"]["cagr"]*100:.1f}%')
print(f'  Headroom over MA50:  full +{(oracle["full"]["cagr"]-ma50["full"]["cagr"])*100:.1f}pp  blind +{(oracle["blind"]["cagr"]-ma50["blind"]["cagr"])*100:.1f}pp')

with open('data/experiment_C_reentry.json','w') as f:
    json.dump(results, f, indent=2, default=float)
print('\nSaved data/experiment_C_reentry.json')
