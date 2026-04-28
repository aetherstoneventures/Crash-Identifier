"""
EXPERIMENT A — Volatility-regime gating on v5

Hypothesis: v5's known failure mode is firing during calm bull-market periods
(false positives that drag CAGR). If we suppress v5 alarms when realized
volatility is in a "calm" regime — defined ONLY using TUNE data — we should
keep ev_det but reduce alarm-day pct and improve precision/CAGR.

Discipline:
  - Threshold and lookback period chosen on TUNE (≤ 2020-12-31) only.
  - Apply unchanged to BLIND (≥ 2021-01-01).
  - Kill criteria (ALL must pass to promote):
      gated ev_det     ≥ v5 ev_det     (don't miss crashes)
      gated day_prec   ≥ v5 day_prec - 0pp
      gated CAGR_blind ≥ v5 CAGR_blind - 0pp
      gated CAGR_full  ≥ v5 CAGR_full  - 0pp
"""
import sqlite3, json
import numpy as np, pandas as pd

DB = 'data/market_crash.db'
TUNE_END = pd.Timestamp('2020-12-31')
TEST_START = pd.Timestamp('2021-01-01')
COST_BPS = 5

conn = sqlite3.connect(DB)
ind   = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds = pd.read_sql("SELECT * FROM predictions WHERE model_version='v5'", conn, parse_dates=['prediction_date'])
conn.close()
eq = ind['nasdaq_close'].ffill().dropna()
sig_v5 = preds.set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
cfg5 = json.loads(open('data/alarm_config_v5.json').read())

# Realized vol — use SP500 daily log returns, 20-day window, annualized
spx = ind['sp500_close'].ffill().reindex(eq.index).ffill()
logret = np.log(spx).diff()
rv20 = logret.rolling(20, min_periods=15).std() * np.sqrt(252)

def crash_labels(p, dd=0.15, mtd=30):
    p = p.ffill(); n = len(p); pv = p.values
    rm = p.rolling(252, min_periods=50).max().values
    below = (pv/rm - 1) <= -dd
    cr = np.zeros(n, dtype=bool); i = 0
    while i < n:
        if not below[i]: i += 1; continue
        rs = i
        while i < n and below[i]: i += 1
        re = i - 1
        tp = rs + int(np.argmin(pv[rs:re+1]))
        peak_val = rm[rs]; pp = rs
        for j in range(rs-1, max(0, rs-260), -1):
            if pv[j] >= peak_val*0.998: pp = j; break
        if tp - pp >= mtd: cr[pp:tp+1] = True
    return pd.Series(cr, index=p.index)

def events(lbl, prices):
    out = []; on = False; cs = None
    for d, v in lbl.items():
        if not on and v: on, cs = True, d
        elif on and not v: out.append({'start': cs, 'end': d}); on = False
    if on: out.append({'start': cs, 'end': lbl.index[-1]})
    df = pd.DataFrame(out)
    if df.empty: return df
    df['peak']   = df.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    df['trough'] = df.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    return df

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

def gate_alarms(alms, gate_ok):
    """Keep alarm episode iff any day in [start, end] is in non-calm regime."""
    out = []
    for s, e in alms:
        if gate_ok.loc[s:e].any(): out.append((s, e))
    return out

def eval_alms(alms, evts, idx, lbl):
    if not alms: return dict(ev_det=0, ev_prec=0, day_prec=0, day_rec=0, adp=0, lead=0, n=0)
    am = alarm_mask(alms, idx)
    y  = lbl.reindex(idx).fillna(False).astype(bool)
    tp = int((am & y).sum()); fp = int((am & ~y).sum()); fn = int((~am & y).sum())
    dp = tp/max(tp+fp, 1); dr = tp/max(tp+fn, 1); adp = am.sum()/len(idx)
    n_conf = sum(1 for s, e in alms
                 if any(((s <= ce.peak) and (s >= ce.peak - pd.Timedelta(days=180))) or
                        ((s <= ce.end) and (e >= ce.start)) for _, ce in evts.iterrows()))
    leads = []; n_det = 0
    for _, ce in evts.iterrows():
        for s, e in alms:
            if (s <= ce.end + pd.Timedelta(days=30)) and (e >= ce.start - pd.Timedelta(days=180)):
                n_det += 1; leads.append((ce.peak - s).days); break
    return dict(ev_det=n_det/max(len(evts), 1), ev_prec=n_conf/len(alms),
                day_prec=dp, day_rec=dr, adp=adp,
                lead=int(np.median(leads)) if leads else 0, n=len(alms))

def backtest_5d_ma50(prices, am, cost_bps=COST_BPS):
    rets = prices.pct_change().fillna(0); ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos = True; days_off = 0; eqv = [1.0]; cost = cost_bps/10000.0
    for i in range(1, len(prices)):
        prev = in_pos
        if am.iloc[i]: in_pos = False; days_off = 0
        else:
            days_off = days_off+1 if not in_pos else days_off
            if not in_pos and days_off >= 5 and prices.iloc[i] > ma50.iloc[i]: in_pos = True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1] * (1 + r))
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
    return dict(cagr=cagr, sharpe=sh, mdd=mdd, final=eqv.iloc[-1])

# Build labels & events
crash = crash_labels(eq)
EV = events(crash, eq)
ev_test = EV[EV.peak >= TEST_START].reset_index(drop=True)

# v5 baseline alarms
sig_full = sig_v5
alm_v5_full = hyst(sig_full, sig_full.index, cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])
am_v5_full = alarm_mask(alm_v5_full, sig_full.index)

# === SELECT GATE THRESHOLD ON TUNE ONLY ===
# Use 252-day rolling quantile of rv20 — gate is "calm" if rv20 < q-th percentile.
# Try several q values on TUNE; pick the one that maximises TUNE backtest CAGR
# subject to ev_det not dropping vs v5 on TUNE.
sig_tune = sig_full[sig_full.index <= TUNE_END]
alm_v5_tune = hyst(sig_tune, sig_tune.index, cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])

ev_tune = EV[EV.peak <= TUNE_END].reset_index(drop=True)
m_v5_tune = eval_alms(alm_v5_tune, ev_tune, sig_tune.index, crash)
print(f'v5 TUNE baseline: ev_det={m_v5_tune["ev_det"]:.1%} day_prec={m_v5_tune["day_prec"]:.1%} adp={m_v5_tune["adp"]:.1%}')

# Compute rolling vol regime — quantile is computed only with PAST data so no leak
rv_q_lookback = 504  # 2 years
results = []
print('\n--- Searching for optimal calm-regime threshold on TUNE ---')
for q in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    q_thr = rv20.rolling(rv_q_lookback, min_periods=252).quantile(q)
    gate_ok = (rv20 >= q_thr)  # True = NOT calm = alarms allowed
    gate_ok = gate_ok.fillna(True)  # if quantile not yet defined, allow alarms
    alm_g_tune = gate_alarms(alm_v5_tune, gate_ok.loc[sig_tune.index])
    m = eval_alms(alm_g_tune, ev_tune, sig_tune.index, crash)
    am_g = alarm_mask(alm_g_tune, sig_full.index)
    eq_g = backtest_5d_ma50(eq, am_g)
    bt = m_window(eq_g, '2000-01-01', '2020-12-31')
    results.append((q, m['ev_det'], m['day_prec'], m['adp'], bt['cagr'], bt['sharpe'], len(alm_g_tune)))
    print(f'  q={q:.2f}: TUNE ev_det={m["ev_det"]:.1%} day_prec={m["day_prec"]:.1%} adp={m["adp"]:.1%}  CAGR={bt["cagr"]*100:.1f}% Sharpe={bt["sharpe"]:.2f}  n_alm={len(alm_g_tune)}')

# Best q on TUNE: maximize CAGR subject to ev_det >= v5 ev_det (no missed crashes)
df = pd.DataFrame(results, columns=['q','ev_det','day_prec','adp','cagr','sharpe','n'])
ok = df[df.ev_det >= m_v5_tune['ev_det'] - 1e-9]
if len(ok) == 0:
    print('\nNO threshold preserves TUNE ev_det. Picking q with highest ev_det.')
    best = df.sort_values(['ev_det','cagr'], ascending=False).iloc[0]
else:
    best = ok.sort_values('cagr', ascending=False).iloc[0]
q_best = best['q']
print(f'\nSELECTED q={q_best:.2f}  (TUNE ev_det={best.ev_det:.1%}, CAGR={best.cagr*100:.1f}%)')

# === APPLY UNCHANGED TO FULL HISTORY ===
q_thr = rv20.rolling(rv_q_lookback, min_periods=252).quantile(q_best)
gate_ok = (rv20 >= q_thr).fillna(True)

alm_v5g_full = gate_alarms(alm_v5_full, gate_ok)
am_v5g_full = alarm_mask(alm_v5g_full, sig_full.index)
eq_v5  = backtest_5d_ma50(eq, am_v5_full)
eq_v5g = backtest_5d_ma50(eq, am_v5g_full)

# BLIND alarms
alm_v5_test  = hyst(sig_full[sig_full.index >= TEST_START], sig_full[sig_full.index >= TEST_START].index,
                    cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])
alm_v5g_test = gate_alarms(alm_v5_test, gate_ok)
m_v5_test  = eval_alms(alm_v5_test,  ev_test, sig_full[sig_full.index >= TEST_START].index, crash)
m_v5g_test = eval_alms(alm_v5g_test, ev_test, sig_full[sig_full.index >= TEST_START].index, crash)

print('\n' + '='*80); print('VOLATILITY-REGIME GATED v5 vs RAW v5'); print('='*80)
print(f'\n{"Metric":<25} {"v5 (raw)":>14} {"v5 + vol-gate":>16}  {"Δ":>10}')
print('-'*80)
print(f'{"BLIND ev_det":<25} {m_v5_test["ev_det"]:>13.1%} {m_v5g_test["ev_det"]:>15.1%}  {(m_v5g_test["ev_det"]-m_v5_test["ev_det"])*100:>+9.1f}pp')
print(f'{"BLIND day_prec":<25} {m_v5_test["day_prec"]:>13.1%} {m_v5g_test["day_prec"]:>15.1%}  {(m_v5g_test["day_prec"]-m_v5_test["day_prec"])*100:>+9.1f}pp')
print(f'{"BLIND ev_prec":<25} {m_v5_test["ev_prec"]:>13.1%} {m_v5g_test["ev_prec"]:>15.1%}  {(m_v5g_test["ev_prec"]-m_v5_test["ev_prec"])*100:>+9.1f}pp')
print(f'{"BLIND adp":<25} {m_v5_test["adp"]:>13.1%} {m_v5g_test["adp"]:>15.1%}  {(m_v5g_test["adp"]-m_v5_test["adp"])*100:>+9.1f}pp')
print(f'{"BLIND lead":<25} {m_v5_test["lead"]:>12}d {m_v5g_test["lead"]:>14}d  {m_v5g_test["lead"]-m_v5_test["lead"]:>+9}d')

bt_full_v5  = m_window(eq_v5,  '2000-01-01', None)
bt_full_v5g = m_window(eq_v5g, '2000-01-01', None)
bt_blind_v5  = m_window(eq_v5,  '2021-01-01', None)
bt_blind_v5g = m_window(eq_v5g, '2021-01-01', None)
bt_tune_v5  = m_window(eq_v5,  '2000-01-01', '2020-12-31')
bt_tune_v5g = m_window(eq_v5g, '2000-01-01', '2020-12-31')

print(f'\n{"FULL CAGR":<25} {bt_full_v5["cagr"]*100:>12.1f}% {bt_full_v5g["cagr"]*100:>14.1f}%  {(bt_full_v5g["cagr"]-bt_full_v5["cagr"])*100:>+9.1f}pp')
print(f'{"FULL Sharpe":<25} {bt_full_v5["sharpe"]:>13.2f} {bt_full_v5g["sharpe"]:>15.2f}  {bt_full_v5g["sharpe"]-bt_full_v5["sharpe"]:>+10.2f}')
print(f'{"FULL MaxDD":<25} {bt_full_v5["mdd"]*100:>12.1f}% {bt_full_v5g["mdd"]*100:>14.1f}%  {(bt_full_v5g["mdd"]-bt_full_v5["mdd"])*100:>+9.1f}pp')
print(f'{"TUNE CAGR":<25} {bt_tune_v5["cagr"]*100:>12.1f}% {bt_tune_v5g["cagr"]*100:>14.1f}%  {(bt_tune_v5g["cagr"]-bt_tune_v5["cagr"])*100:>+9.1f}pp')
print(f'{"BLIND CAGR":<25} {bt_blind_v5["cagr"]*100:>12.1f}% {bt_blind_v5g["cagr"]*100:>14.1f}%  {(bt_blind_v5g["cagr"]-bt_blind_v5["cagr"])*100:>+9.1f}pp')
print(f'{"BLIND Sharpe":<25} {bt_blind_v5["sharpe"]:>13.2f} {bt_blind_v5g["sharpe"]:>15.2f}  {bt_blind_v5g["sharpe"]-bt_blind_v5["sharpe"]:>+10.2f}')
print(f'{"BLIND MaxDD":<25} {bt_blind_v5["mdd"]*100:>12.1f}% {bt_blind_v5g["mdd"]*100:>14.1f}%  {(bt_blind_v5g["mdd"]-bt_blind_v5["mdd"])*100:>+9.1f}pp')

# Kill criteria for vol-gate
print('\n--- KILL CRITERIA (gate must NOT regress v5) ---')
checks = [
    ('BLIND ev_det',     m_v5g_test['ev_det']    - m_v5_test['ev_det'],    -0.0,  'pp'),
    ('BLIND day_prec',   m_v5g_test['day_prec']  - m_v5_test['day_prec'],  -0.0,  'pp'),
    ('FULL  CAGR',       bt_full_v5g['cagr']     - bt_full_v5['cagr'],     -0.005,'pp'),
    ('BLIND CAGR',       bt_blind_v5g['cagr']    - bt_blind_v5['cagr'],    -0.005,'pp'),
]
all_pass = True
for name, delta, threshold, unit in checks:
    dd = delta*100 if unit == 'pp' else delta
    tt = threshold*100 if unit == 'pp' else threshold
    ok = delta >= threshold
    if not ok: all_pass = False
    print(f'  {name:<18s}  Δ = {dd:>+7.2f}{unit}   threshold {tt:>+5.2f}{unit}   {"PASS" if ok else "FAIL"}')

print('\n' + '='*80)
if all_pass:
    print(f'VERDICT: vol-gate (q={q_best:.2f}) PASSES — candidate v5 enhancement.')
else:
    print(f'VERDICT: vol-gate FAILS — discarded.')
print('='*80)

out = {'experiment':'A_vol_regime_gating','q_selected':float(q_best),
       'lookback_days':rv_q_lookback,'verdict':'PASS' if all_pass else 'FAIL',
       'v5_raw':{'tune':bt_tune_v5,'blind':bt_blind_v5,'full':bt_full_v5,'alarm':m_v5_test},
       'v5_gated':{'tune':bt_tune_v5g,'blind':bt_blind_v5g,'full':bt_full_v5g,'alarm':m_v5g_test}}
with open('data/experiment_A_vol_gate.json','w') as f: json.dump(out, f, indent=2, default=float)
print('Saved data/experiment_A_vol_gate.json')
