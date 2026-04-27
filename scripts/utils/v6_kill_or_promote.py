"""
v6 KILL-OR-PROMOTE — applies the four hard kill criteria from train_v6.py.

Kill if ANY of these holds on BLIND (2021-2026):
  v6 day_precision  < v5 day_precision - 5 pp
  v6 ev_det         < v5 ev_det        - 5 pp
  v6 median lead    < v5 median lead   - 5 days
  v6 backtest CAGR  < v5 backtest CAGR - 2 pp

Also computes the standard 5d-MA50 backtest for both models.
"""
import sqlite3, json, sys
sys.path.insert(0, '.')
import pandas as pd, numpy as np

DB = 'data/market_crash.db'
TUNE_END = pd.Timestamp('2020-12-31'); TEST_START = pd.Timestamp('2021-01-01')
COST_BPS = 5

conn = sqlite3.connect(DB)
ind   = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds = pd.read_sql('SELECT * FROM predictions', conn, parse_dates=['prediction_date'])
conn.close()
eq = ind['nasdaq_close'].ffill().dropna()
sig_v5 = preds[preds.model_version=='v5'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
sig_v6 = preds[preds.model_version=='v6'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()

cfg5 = json.loads(open('data/alarm_config_v5.json').read())
cfg6 = json.loads(open('data/alarm_config_v6.json').read())

# Crash labels and events (canonical)
def crash_labels(p, dd=0.15, mtd=30):
    p = p.ffill(); n=len(p); pv=p.values
    rm = p.rolling(252, min_periods=50).max().values
    below = (pv/rm - 1) <= -dd
    cr = np.zeros(n, dtype=bool); i=0
    while i<n:
        if not below[i]: i+=1; continue
        rs=i
        while i<n and below[i]: i+=1
        re=i-1
        tp = rs + int(np.argmin(pv[rs:re+1]))
        peak_val = rm[rs]; pp=rs
        for j in range(rs-1, max(0, rs-260), -1):
            if pv[j] >= peak_val*0.998: pp=j; break
        if tp-pp >= mtd: cr[pp:tp+1]=True
    return pd.Series(cr, index=p.index)
crash = crash_labels(eq)

def events(lbl, prices):
    out=[]; on=False; cs=None
    for d,v in lbl.items():
        if not on and v: on, cs = True, d
        elif on and not v: out.append({'start':cs,'end':d}); on=False
    if on: out.append({'start':cs,'end':lbl.index[-1]})
    df = pd.DataFrame(out)
    if df.empty: return df
    df['peak'] = df.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    df['trough'] = df.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    df['drawdown'] = df.apply(lambda r: prices.loc[r.trough]/prices.loc[r.peak]-1, axis=1)
    return df
EV = events(crash, eq)
ev_test = EV[EV.peak >= TEST_START].reset_index(drop=True)

def hyst(p, idx, en, ex, mn, mx):
    a=[]; on=False; si=None; pv=p.values
    for i in range(len(pv)):
        if not on:
            if pv[i] >= en: on=True; si=i
        else:
            d=i-si
            if pv[i] < ex or d >= mx:
                if d >= mn: a.append((idx[si], idx[i-1]))
                on = False
    if on and si is not None and (len(pv)-si) >= mn: a.append((idx[si], idx[-1]))
    return a

def alarm_mask(alms, idx):
    m = pd.Series(False, index=idx)
    for s,e in alms: m.loc[s:e] = True
    return m

def eval_alms(alms, evts, idx, lbl):
    if not alms: return dict(ev_det=0,ev_prec=0,day_prec=0,day_rec=0,adp=0,lead=0,n=0)
    am = alarm_mask(alms, idx)
    y  = lbl.reindex(idx).fillna(False).astype(bool)
    tp = int((am&y).sum()); fp = int((am&~y).sum()); fn = int((~am&y).sum())
    dp = tp/max(tp+fp,1); dr = tp/max(tp+fn,1); adp = am.sum()/len(idx)
    n_conf = sum(1 for s,e in alms
                 if any(((s <= ce.peak) and (s >= ce.peak-pd.Timedelta(days=180))) or
                        ((s <= ce.end) and (e >= ce.start)) for _, ce in evts.iterrows()))
    leads=[]; n_det=0
    for _, ce in evts.iterrows():
        for s,e in alms:
            if (s <= ce.end+pd.Timedelta(days=30)) and (e >= ce.start-pd.Timedelta(days=180)):
                n_det += 1; leads.append((ce.peak - s).days); break
    return dict(ev_det=n_det/max(len(evts),1), ev_prec=n_conf/len(alms),
                day_prec=dp, day_rec=dr, adp=adp,
                lead=int(np.median(leads)) if leads else 0, n=len(alms))

def backtest_5d_ma50(prices, am, cost_bps=COST_BPS):
    rets = prices.pct_change().fillna(0); ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos=True; days_off=0; eqv=[1.0]; cost=cost_bps/10000.0
    for i in range(1, len(prices)):
        prev=in_pos
        if am.iloc[i]: in_pos=False; days_off=0
        else:
            days_off = days_off+1 if not in_pos else days_off
            if not in_pos and days_off >= 5 and prices.iloc[i] > ma50.iloc[i]: in_pos=True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1]*(1+r))
    return pd.Series(eqv, index=prices.index)

def m_window(eqv, s, e):
    if s: eqv = eqv.loc[pd.Timestamp(s):]
    if e: eqv = eqv.loc[:pd.Timestamp(e)]
    eqv = eqv/eqv.iloc[0]
    yrs = (eqv.index[-1]-eqv.index[0]).days/365.25
    cagr = eqv.iloc[-1]**(1/max(yrs, 1e-6)) - 1
    rr = eqv.pct_change().dropna()
    sh = rr.mean()/rr.std()*np.sqrt(252) if rr.std() > 0 else 0
    mdd = (eqv/eqv.cummax() - 1).min()
    return dict(cagr=cagr, sharpe=sh, mdd=mdd, final=eqv.iloc[-1])

# Run alarms
alm5_test = hyst(sig_v5[sig_v5.index >= TEST_START], sig_v5[sig_v5.index >= TEST_START].index,
                 cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])
alm6_test = hyst(sig_v6[sig_v6.index >= TEST_START], sig_v6[sig_v6.index >= TEST_START].index,
                 cfg6['entry'], cfg6['exit'], cfg6['min_dur'], cfg6['max_dur'])
m5_test = eval_alms(alm5_test, ev_test, sig_v5[sig_v5.index >= TEST_START].index, crash)
m6_test = eval_alms(alm6_test, ev_test, sig_v6[sig_v6.index >= TEST_START].index, crash)

# Full-window alarm masks for backtest
alm5_full = hyst(sig_v5, sig_v5.index, cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])
alm6_full = hyst(sig_v6, sig_v6.index, cfg6['entry'], cfg6['exit'], cfg6['min_dur'], cfg6['max_dur'])
am5 = alarm_mask(alm5_full, sig_v5.index); am6 = alarm_mask(alm6_full, sig_v6.index)
eq5 = backtest_5d_ma50(eq, am5); eq6 = backtest_5d_ma50(eq, am6)

print('='*80); print('v5 vs v6 — BLIND VALIDATION'); print('='*80)
print(f'\n{"Metric":<25} {"v5 (BENCHMARK)":>16} {"v6 (CANDIDATE)":>16}  {"Δ":>10}')
print('-'*80)
print(f'{"BLIND day_precision":<25} {m5_test["day_prec"]:>15.1%} {m6_test["day_prec"]:>15.1%}  {(m6_test["day_prec"]-m5_test["day_prec"])*100:>+9.1f}pp')
print(f'{"BLIND ev_det":<25} {m5_test["ev_det"]:>15.1%} {m6_test["ev_det"]:>15.1%}  {(m6_test["ev_det"]-m5_test["ev_det"])*100:>+9.1f}pp')
print(f'{"BLIND median lead":<25} {m5_test["lead"]:>14}d  {m6_test["lead"]:>14}d  {m6_test["lead"]-m5_test["lead"]:>+9}d')
print(f'{"BLIND alarm-day pct":<25} {m5_test["adp"]:>15.1%} {m6_test["adp"]:>15.1%}  {(m6_test["adp"]-m5_test["adp"])*100:>+9.1f}pp')
print(f'{"BLIND ev_prec":<25} {m5_test["ev_prec"]:>15.1%} {m6_test["ev_prec"]:>15.1%}  {(m6_test["ev_prec"]-m5_test["ev_prec"])*100:>+9.1f}pp')

print('\nBACKTEST (5d + MA50 re-entry, 5bps cost):')
mb5_full = m_window(eq5, '2000-01-01', None);   mb6_full = m_window(eq6, '2000-01-01', None)
mb5_blind= m_window(eq5, '2021-01-01', None);   mb6_blind= m_window(eq6, '2021-01-01', None)
mb5_tune = m_window(eq5, '2000-01-01', '2020-12-31'); mb6_tune = m_window(eq6, '2000-01-01', '2020-12-31')
print(f'{"FULL  CAGR":<25} {mb5_full["cagr"]*100:>15.1f}% {mb6_full["cagr"]*100:>15.1f}%  {(mb6_full["cagr"]-mb5_full["cagr"])*100:>+9.1f}pp')
print(f'{"FULL  Sharpe":<25} {mb5_full["sharpe"]:>16.2f} {mb6_full["sharpe"]:>16.2f}  {mb6_full["sharpe"]-mb5_full["sharpe"]:>+10.2f}')
print(f'{"FULL  MaxDD":<25} {mb5_full["mdd"]*100:>15.1f}% {mb6_full["mdd"]*100:>15.1f}%  {(mb6_full["mdd"]-mb5_full["mdd"])*100:>+9.1f}pp')
print(f'{"TUNE  CAGR":<25} {mb5_tune["cagr"]*100:>15.1f}% {mb6_tune["cagr"]*100:>15.1f}%  {(mb6_tune["cagr"]-mb5_tune["cagr"])*100:>+9.1f}pp')
print(f'{"BLIND CAGR":<25} {mb5_blind["cagr"]*100:>15.1f}% {mb6_blind["cagr"]*100:>15.1f}%  {(mb6_blind["cagr"]-mb5_blind["cagr"])*100:>+9.1f}pp')
print(f'{"BLIND Sharpe":<25} {mb5_blind["sharpe"]:>16.2f} {mb6_blind["sharpe"]:>16.2f}  {mb6_blind["sharpe"]-mb5_blind["sharpe"]:>+10.2f}')
print(f'{"BLIND MaxDD":<25} {mb5_blind["mdd"]*100:>15.1f}% {mb6_blind["mdd"]*100:>15.1f}%  {(mb6_blind["mdd"]-mb5_blind["mdd"])*100:>+9.1f}pp')

# KILL CRITERIA
print('\n--- KILL CRITERIA (all must PASS to promote) ---')
checks = [
    ('day_precision', m6_test['day_prec'] - m5_test['day_prec'], -0.05, 'pp'),
    ('ev_det',        m6_test['ev_det']   - m5_test['ev_det'],   -0.05, 'pp'),
    ('median lead',   m6_test['lead']     - m5_test['lead'],     -5,    'd'),
    ('blind CAGR',    mb6_blind['cagr']   - mb5_blind['cagr'],   -0.02, 'pp'),
]
all_pass = True
for name, delta, threshold, unit in checks:
    delta_disp = delta*100 if unit == 'pp' else delta
    threshold_disp = threshold*100 if unit == 'pp' else threshold
    ok = delta >= threshold
    if not ok: all_pass = False
    status = 'PASS' if ok else 'FAIL'
    print(f'  {name:<18s} v6 - v5 = {delta_disp:>+7.1f}{unit}   threshold {threshold_disp:>+5.1f}{unit}   {status}')

print('\n'+'='*80)
if all_pass:
    print('VERDICT: v6 PASSES kill criteria — candidate for promotion.')
    print('  However, MAINTAIN DISCIPLINE: re-verify by manual inspection before any swap.')
else:
    print('VERDICT: v6 FAILS — SHELVED. v5 remains production benchmark.')
print('='*80)

# Persist verdict
out = {'verdict':'PASS' if all_pass else 'FAIL',
       'v5':{'tune':mb5_tune,'blind':mb5_blind,'full':mb5_full,'alarm':m5_test},
       'v6':{'tune':mb6_tune,'blind':mb6_blind,'full':mb6_full,'alarm':m6_test},
       'kill_criteria':[{'metric':n,'delta':d,'threshold':t,'pass':d>=t} for n,d,t,_ in checks]}
with open('data/v6_kill_verdict.json','w') as f: json.dump(out, f, indent=2, default=float)
print('Saved data/v6_kill_verdict.json')
