"""
EXPERIMENT D — round 2: re-tune alarm config on US TUNE for multi-asset signal.

Round 1 used v5's threshold which is unfair (different calibration).
Round 2: independent grid search on US TUNE ≤ 2020-12-31, then evaluate BLIND.
"""
import json, sqlite3, os
import numpy as np, pandas as pd
import yfinance as yf
from curl_cffi import requests as _crequests
import xgboost as xgb
from sklearn.metrics import roc_auc_score

DB = 'data/market_crash.db'
TUNE_END = pd.Timestamp('2020-12-31'); TEST_START = pd.Timestamp('2021-01-01')
COST_BPS = 5
CACHE = 'data/cache/multi_asset_prices.parquet'
_session = _crequests.Session(impersonate='chrome')

INDICES = {'US_IXIC':'^IXIC','EU_STOXX':'^STOXX50E','EU_DAX':'^GDAXI',
           'EU_FTSE':'^FTSE','AS_N225':'^N225','AS_HSI':'^HSI'}

px = pd.read_parquet(CACHE)
conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds_v5 = pd.read_sql("SELECT * FROM predictions WHERE model_version='v5'", conn, parse_dates=['prediction_date'])
preds_stat = pd.read_sql("SELECT * FROM predictions WHERE model_version='StatV3_phase3'", conn, parse_dates=['prediction_date'])
conn.close()

def crash_labels(p, dd=0.15, mtd=30):
    p = p.ffill().dropna(); n = len(p); pv = p.values
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

def per_index_features(price):
    p = price.ffill().dropna()
    out = pd.DataFrame(index=p.index)
    out['equity_drawdown']    = p / p.rolling(252, min_periods=50).max() - 1
    out['equity_return_5d']   = p.pct_change(5)
    out['equity_return_20d']  = p.pct_change(20)
    out['equity_return_63d']  = p.pct_change(63)
    out['equity_return_252d'] = p.pct_change(252)
    logr = np.log(p).diff()
    out['equity_vol_20d']  = logr.rolling(20).std() * np.sqrt(252)
    out['equity_vol_60d']  = logr.rolling(60).std() * np.sqrt(252)
    out['equity_vol_ratio'] = out['equity_vol_20d'] / out['equity_vol_60d']
    out['equity_ma50_dist']  = p / p.rolling(50, min_periods=20).mean() - 1
    out['equity_ma200_dist'] = p / p.rolling(200, min_periods=100).mean() - 1
    out['equity_momentum']   = (p.pct_change(20) - p.pct_change(60)).fillna(0)
    return out

us_macro = pd.DataFrame(index=ind.index)
for c in ['vix_close','regime_num','credit_stress','sentiment','economic',
          'financial_conditions','market_momentum','momentum_shock','total_risk_v3','hy_credit']:
    if c in ind.columns: us_macro[c] = ind[c]
us_macro = us_macro.ffill()

panels = []; labels = []; idx_arr = []
for name in INDICES:
    if name not in px.columns: continue
    p = px[name].dropna()
    if len(p) < 1000: continue
    f = per_index_features(p)
    macro = us_macro.reindex(f.index, method='ffill')
    feats = pd.concat([f, macro], axis=1).dropna()
    lab = crash_labels(p).reindex(feats.index).fillna(False).astype(int)
    panels.append(feats); labels.append(lab); idx_arr.append(np.array([name]*len(feats)))

X_all = pd.concat(panels, axis=0)
y_all = pd.concat(labels, axis=0)
idx_all = np.concatenate(idx_arr)
dates_all = X_all.index

mask_tune = (dates_all <= TUNE_END)
spw = (1 - y_all[mask_tune]).sum() / max(int(y_all[mask_tune].sum()), 1)
clf = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                       subsample=0.75, colsample_bytree=0.8, min_child_weight=15,
                       reg_alpha=0.1, reg_lambda=2.0, scale_pos_weight=spw,
                       eval_metric='auc', random_state=42, tree_method='hist', n_jobs=-1)
clf.fit(X_all[mask_tune], y_all[mask_tune])

proba_all = clf.predict_proba(X_all)[:, 1]
prob_us = pd.Series(proba_all[idx_all == 'US_IXIC'],
                    index=dates_all[idx_all == 'US_IXIC']).sort_index()
prob_us = prob_us[~prob_us.index.duplicated(keep='last')]

sv3 = preds_stat.set_index('prediction_date')['crash_probability'].sort_index().reindex(prob_us.index).ffill().fillna(0)
sig_d = 0.5 * prob_us + 0.5 * sv3
sig_v5 = preds_v5.set_index('prediction_date')['crash_probability'].sort_index().reindex(prob_us.index).ffill()
eq_us = px['US_IXIC'].ffill().reindex(prob_us.index).ffill()
crash_us = crash_labels(eq_us)

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
    return df

EV = events(crash_us, eq_us)
ev_tune = EV[EV.peak <= TUNE_END].reset_index(drop=True)
ev_test = EV[EV.peak >= TEST_START].reset_index(drop=True)

def eval_alms(alms, evts, idx, lbl):
    if not alms: return dict(ev_det=0, ev_prec=0, day_prec=0, day_rec=0, adp=0, lead=0, n=0)
    am = alarm_mask(alms, idx)
    y = lbl.reindex(idx).fillna(False).astype(bool)
    tp = int((am & y).sum()); fp = int((am & ~y).sum()); fn = int((~am & y).sum())
    dp = tp/max(tp+fp, 1); dr = tp/max(tp+fn, 1); adp = am.sum()/len(idx)
    n_conf = sum(1 for s, e in alms
                 if any(((s <= ce.peak) and (s >= ce.peak - pd.Timedelta(days=180))) or
                        ((s <= ce.end) and (e >= ce.start)) for _, ce in evts.iterrows()))
    leads=[]; n_det=0
    for _, ce in evts.iterrows():
        for s, e in alms:
            if (s <= ce.end + pd.Timedelta(days=30)) and (e >= ce.start - pd.Timedelta(days=180)):
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
            days_off = days_off+1 if not in_pos else 0
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
    cagr = eqv.iloc[-1]**(1/max(yrs,1e-6)) - 1
    rr = eqv.pct_change().dropna()
    sh = rr.mean()/rr.std()*np.sqrt(252) if rr.std()>0 else 0
    mdd = (eqv/eqv.cummax() - 1).min()
    return dict(cagr=cagr, sharpe=sh, mdd=mdd)

# === GRID SEARCH ON TUNE ONLY ===
sig_d_tune = sig_d[sig_d.index <= TUNE_END]
print('Grid search alarm config on TUNE only...')
best = None
results = []
for entry in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    for exit_thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        if exit_thr >= entry: continue
        for min_dur in [10, 15, 20, 25]:
            alms = hyst(sig_d_tune, sig_d_tune.index, entry, exit_thr, min_dur, 30)
            m = eval_alms(alms, ev_tune, sig_d_tune.index, crash_us)
            am_full = alarm_mask(hyst(sig_d, sig_d.index, entry, exit_thr, min_dur, 30), sig_d.index)
            eqv = backtest_5d_ma50(eq_us, am_full)
            bt_tune = m_window(eqv, '2000-01-01', '2020-12-31')
            results.append({'entry':entry,'exit':exit_thr,'min_dur':min_dur,
                            'tune_ev_det':m['ev_det'],'tune_day_prec':m['day_prec'],
                            'tune_adp':m['adp'],'tune_cagr':bt_tune['cagr'],
                            'tune_sharpe':bt_tune['sharpe']})
df = pd.DataFrame(results)
# Pick best by TUNE CAGR subject to ev_det >= 0.90 and day_prec >= 0.80
ok = df[(df.tune_ev_det >= 0.90) & (df.tune_day_prec >= 0.80)]
if len(ok) == 0:
    print('No config meets ev_det>=0.90 AND day_prec>=0.80; falling back to ev_det>=0.85')
    ok = df[df.tune_ev_det >= 0.85]
print(f'\n{len(ok)}/{len(df)} configs meet TUNE quality bar.')
print('Top 5 by TUNE Sharpe:')
print(ok.sort_values('tune_sharpe', ascending=False).head(5).to_string(index=False))

best = ok.sort_values('tune_sharpe', ascending=False).iloc[0]
ENTRY, EXIT, MD = float(best.entry), float(best.exit), int(best.min_dur)
print(f'\nSELECTED: entry={ENTRY} exit={EXIT} min_dur={MD}')

# === EVALUATE ON BLIND ===
sig_d_test = sig_d[sig_d.index >= TEST_START]
sig_5_test = sig_v5[sig_v5.index >= TEST_START]
cfg5 = json.loads(open('data/alarm_config_v5.json').read())

alms_d_test = hyst(sig_d_test, sig_d_test.index, ENTRY, EXIT, MD, 30)
alms_5_test = hyst(sig_5_test, sig_5_test.index, cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur'])

m_d_test = eval_alms(alms_d_test, ev_test, sig_d_test.index, crash_us)
m_5_test = eval_alms(alms_5_test, ev_test, sig_5_test.index, crash_us)

am_d_full = alarm_mask(hyst(sig_d, sig_d.index, ENTRY, EXIT, MD, 30), sig_d.index)
am_5_full = alarm_mask(hyst(sig_v5, sig_v5.index, cfg5['entry'], cfg5['exit'], cfg5['min_dur'], cfg5['max_dur']), sig_v5.index)
eq_d = backtest_5d_ma50(eq_us, am_d_full)
eq_5 = backtest_5d_ma50(eq_us, am_5_full)

print('\n' + '='*80); print('MULTI-ASSET v5 (re-tuned alarm) vs v5 BENCHMARK'); print('='*80)
print(f'\n{"Metric":<25} {"v5":>14} {"v5_multi":>16}  {"Δ":>10}')
print('-'*80)
print(f'{"BLIND ev_det":<25} {m_5_test["ev_det"]:>13.1%} {m_d_test["ev_det"]:>15.1%}  {(m_d_test["ev_det"]-m_5_test["ev_det"])*100:>+9.1f}pp')
print(f'{"BLIND day_prec":<25} {m_5_test["day_prec"]:>13.1%} {m_d_test["day_prec"]:>15.1%}  {(m_d_test["day_prec"]-m_5_test["day_prec"])*100:>+9.1f}pp')
print(f'{"BLIND ev_prec":<25} {m_5_test["ev_prec"]:>13.1%} {m_d_test["ev_prec"]:>15.1%}  {(m_d_test["ev_prec"]-m_5_test["ev_prec"])*100:>+9.1f}pp')
print(f'{"BLIND adp":<25} {m_5_test["adp"]:>13.1%} {m_d_test["adp"]:>15.1%}  {(m_d_test["adp"]-m_5_test["adp"])*100:>+9.1f}pp')
print(f'{"BLIND lead":<25} {m_5_test["lead"]:>12}d {m_d_test["lead"]:>14}d  {m_d_test["lead"]-m_5_test["lead"]:>+9}d')

bf5 = m_window(eq_5, '2000-01-01', None);     bfd = m_window(eq_d, '2000-01-01', None)
bt5 = m_window(eq_5, '2000-01-01','2020-12-31'); btd = m_window(eq_d, '2000-01-01','2020-12-31')
bb5 = m_window(eq_5, '2021-01-01', None);     bbd = m_window(eq_d, '2021-01-01', None)
print(f'\n{"FULL CAGR":<25} {bf5["cagr"]*100:>12.1f}% {bfd["cagr"]*100:>14.1f}%  {(bfd["cagr"]-bf5["cagr"])*100:>+9.1f}pp')
print(f'{"FULL Sharpe":<25} {bf5["sharpe"]:>13.2f} {bfd["sharpe"]:>15.2f}  {bfd["sharpe"]-bf5["sharpe"]:>+10.2f}')
print(f'{"FULL MaxDD":<25} {bf5["mdd"]*100:>12.1f}% {bfd["mdd"]*100:>14.1f}%  {(bfd["mdd"]-bf5["mdd"])*100:>+9.1f}pp')
print(f'{"TUNE CAGR":<25} {bt5["cagr"]*100:>12.1f}% {btd["cagr"]*100:>14.1f}%  {(btd["cagr"]-bt5["cagr"])*100:>+9.1f}pp')
print(f'{"BLIND CAGR":<25} {bb5["cagr"]*100:>12.1f}% {bbd["cagr"]*100:>14.1f}%  {(bbd["cagr"]-bb5["cagr"])*100:>+9.1f}pp')
print(f'{"BLIND Sharpe":<25} {bb5["sharpe"]:>13.2f} {bbd["sharpe"]:>15.2f}  {bbd["sharpe"]-bb5["sharpe"]:>+10.2f}')
print(f'{"BLIND MaxDD":<25} {bb5["mdd"]*100:>12.1f}% {bbd["mdd"]*100:>14.1f}%  {(bbd["mdd"]-bb5["mdd"])*100:>+9.1f}pp')

print('\n--- KILL CRITERIA ---')
checks = [
    ('BLIND ev_det',     m_d_test['ev_det']    - m_5_test['ev_det'],    -0.05),
    ('BLIND day_prec',   m_d_test['day_prec']  - m_5_test['day_prec'],  -0.05),
    ('FULL  CAGR',       bfd['cagr']           - bf5['cagr'],           -0.02),
    ('BLIND CAGR',       bbd['cagr']           - bb5['cagr'],           -0.02),
    ('FULL  MaxDD',      bfd['mdd']            - bf5['mdd'],            -0.05),
]
all_pass = True
for name, delta, threshold in checks:
    ok_check = delta >= threshold
    if not ok_check: all_pass = False
    print(f'  {name:<18s}  Δ = {delta*100:>+7.2f}pp   threshold {threshold*100:>+5.2f}pp   {"PASS" if ok_check else "FAIL"}')

print('\n' + '='*80)
print(f'VERDICT: multi-asset v5 (re-tuned) {"PASSES" if all_pass else "FAILS"}.')
print('='*80)

with open('data/experiment_D_round2.json','w') as f:
    json.dump({'verdict':'PASS' if all_pass else 'FAIL',
               'config_selected':{'entry':ENTRY,'exit':EXIT,'min_dur':MD},
               'v5':{'tune':bt5,'blind':bb5,'full':bf5,'alarm':m_5_test},
               'v5_multi':{'tune':btd,'blind':bbd,'full':bfd,'alarm':m_d_test}},
              f, indent=2, default=float)
print('Saved data/experiment_D_round2.json')
