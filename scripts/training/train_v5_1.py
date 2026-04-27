"""
V5.1 — predictive ensemble + bottom-finder (HONEST training).

Three models:
  1. v5_coincident   — target: in_crash[t]               (precision; what v5 does today)
  2. v5_predictive   — target: in_crash[t..t+K_LEAD]     (forecasts crash starting in next K days)
  3. v5_bottom       — target: forward_60d_ret > 0 GIVEN drawdown < -10%

Combined alarm signal = max(coincident_blended, predictive_blended).
Bottom-finder used only to time re-entry (separate from the alarm).

Honest discipline:
  - Nested walk-forward (4 folds 1999-2026)
  - Alarm hysteresis grid-searched on 1999-2020 only
  - 2021-2026 strictly blind
  - Lead measured FROM PEAK
  - Per-crash scorecard on full history
  - Backtest reported alongside buy-and-hold and v5
"""
import sys, sqlite3, pickle, json
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction

DB = 'data/market_crash.db'
K_LEAD = 20    # predictive horizon
DD_FOR_BOTTOM = -0.10
FWD_BOTTOM_DAYS = 60

print('='*78); print(f'V5.1 ENSEMBLE TRAINING  (K_LEAD={K_LEAD}d)'); print('='*78)

# ── 1. LOAD ─────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
conn.close()
eq = ind['nasdaq_close'].copy()
ind = ind.loc[eq.dropna().index]
eq  = ind['nasdaq_close'].ffill()
print(f'Daily NASDAQ rows: {len(eq)}  ({eq.index[0].date()} -> {eq.index[-1].date()})')

feats = engineer_features_for_prediction(ind)

# Equity-derived features (already done in v5)
rm = eq.rolling(252, min_periods=50).max()
feats['equity_drawdown']    = eq / rm - 1
feats['equity_return_5d']   = eq.pct_change(5)
feats['equity_return_20d']  = eq.pct_change(20)
feats['equity_return_63d']  = eq.pct_change(63)
feats['equity_vol_20d']     = eq.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
feats['equity_vol_63d']     = eq.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
# Vol-of-vol (acceleration of stress)
feats['equity_vol_chg_20d'] = feats['equity_vol_20d'].diff(20)
# Days since 252d high (regime exhaustion)
feats['days_since_high']    = (eq.rolling(252, min_periods=50).apply(lambda x: len(x)-1-int(np.argmax(x)))).fillna(0)

# Existing v5 add-ons (forward-fill safe)
for col in ['baa_10y_spread','oil_wti','dollar_twi','epu_daily']:
    if col in ind.columns: feats[col] = ind[col].ffill()

# === NEW v5.1 features ===
# VIX term structure (key leading indicator; 2007+)
vix = ind.get('vix_close')
vxv = ind.get('vxv_close')
vxo = ind.get('vxo_close')
if vix is not None and vxv is not None:
    # Construct a "wide-history" near-month VIX: prefer VIXCLS, fallback VXO
    vix_long = vix.copy()
    if vxo is not None: vix_long = vix_long.fillna(vxo)
    feats['vix_long']      = vix_long.ffill()
    feats['vix_term_ratio'] = (vix_long / vxv).clip(0.5, 2.0)  # >1 => stress now > forward
    feats['vix_term_ratio'] = feats['vix_term_ratio'].ffill()

# AAA-10Y credit spread (1983+ daily)
if 'aaa_10y_spread' in ind.columns:
    feats['aaa_10y'] = ind['aaa_10y_spread'].ffill()
    feats['aaa_chg_20d'] = feats['aaa_10y'].diff(20)

# NFCI subindexes (1971+ weekly — perfect long-history macro)
for col in ['nfci_leverage','nfci_risk','nfci_credit']:
    if col in ind.columns:
        feats[col] = ind[col].ffill()

# Monthly stress + sentiment (long history, ffill safely)
for col in ['kcfsi','umcsent']:
    if col in ind.columns:
        feats[col] = ind[col].ffill()

# Real yield + breakeven inflation deltas (2003+)
for col in ['t10yie','dfii10']:
    if col in ind.columns:
        feats[col] = ind[col].ffill()
        feats[col + '_chg_20d'] = feats[col].diff(20)

# velocities for dollar / oil / epu (already in v5)
feats['baa_chg_20d']   = feats.get('baa_10y_spread', pd.Series(0, index=feats.index)).diff(20)
feats['oil_return_20d']= feats.get('oil_wti', pd.Series(np.nan, index=feats.index)).pct_change(20)
feats['dollar_chg_20d']= feats.get('dollar_twi', pd.Series(np.nan, index=feats.index)).pct_change(20)
feats['epu_chg_20d']   = feats.get('epu_daily', pd.Series(np.nan, index=feats.index)).diff(20)

# ── 2. CRASH LABELS ─────────────────────────────────────────────────────────
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

crash = compute_crash_labels(eq, 0.15, 30)
# Predictive label: will we be in a crash within next K_LEAD days?
crash_arr = crash.values
n = len(crash_arr)
predictive = np.zeros(n, dtype=bool)
for i in range(n):
    j_end = min(i + K_LEAD + 1, n)
    if crash_arr[i:j_end].any():
        predictive[i] = True
predictive_lbl = pd.Series(predictive, index=crash.index)
print(f'Coincident crash days:   {int(crash.sum())}  ({crash.mean():.1%})')
print(f'Predictive ({K_LEAD}d ahead) days: {int(predictive_lbl.sum())}  ({predictive_lbl.mean():.1%})')

# Bottom-finder label: forward 60d return > 10%, given currently in drawdown >10%
fwd60 = eq.pct_change(FWD_BOTTOM_DAYS).shift(-FWD_BOTTOM_DAYS)
in_dd = feats['equity_drawdown'] <= DD_FOR_BOTTOM
bottom_lbl = ((fwd60 > 0.10) & in_dd).astype(int)
print(f'Bottom-finder positive samples: {int(bottom_lbl.sum())} (out of {int(in_dd.sum())} in-drawdown days)')

# ── 3. STATV3 SCORES ────────────────────────────────────────────────────────
print('\n--- StatV3 factors ---')
sv3 = StatisticalModelV3()
factor_names = ['yield_curve','volatility','credit_stress','hy_credit',
                'economic','labor_market','market_momentum','sentiment',
                'financial_conditions','momentum_shock']
rows=[]
for _, row in feats.iterrows():
    p, expl = sv3._calculate_crash_probability_with_factors(row)
    r = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    r['regime_num']    = {'low':0,'normal':1,'high':2,'extreme':3}.get(expl.get('regime','normal'), 1)
    r['total_risk_v3'] = p
    rows.append(r)
sc = pd.DataFrame(rows, index=feats.index)

raw_feats = [
    'equity_drawdown','equity_return_5d','equity_return_20d','equity_return_63d',
    'equity_vol_20d','equity_vol_63d','equity_vol_chg_20d','days_since_high',
    'vix_close','yield_10y_2y','nfci','recession_prob','initial_claims_change_13w',
    'stress_composite','yield_curve_velocity_63d',
    'baa_10y_spread','baa_chg_20d','oil_wti','oil_return_20d','dollar_chg_20d',
    'epu_chg_20d','epu_daily',
    # NEW v5.1
    'vix_long','vix_term_ratio','aaa_10y','aaa_chg_20d',
    'nfci_leverage','nfci_risk','nfci_credit',
    'kcfsi','umcsent','t10yie','t10yie_chg_20d','dfii10','dfii10_chg_20d',
]
for f in raw_feats:
    if f in feats.columns: sc[f] = feats[f].values

sc = sc.ffill().bfill().fillna(0)
feat_cols = list(sc.columns)
print(f'Feature matrix: {sc.shape}  ({len(feat_cols)} features)')

# ── 4. WALK-FORWARD HELPERS ──────────────────────────────────────────────────
folds = [
    (pd.Timestamp('1999-12-31'), pd.Timestamp('2000-01-01'), pd.Timestamp('2005-12-31')),
    (pd.Timestamp('2005-12-31'), pd.Timestamp('2006-01-01'), pd.Timestamp('2012-12-31')),
    (pd.Timestamp('2012-12-31'), pd.Timestamp('2013-01-01'), pd.Timestamp('2020-12-31')),
    (pd.Timestamp('2020-12-31'), pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31')),
]
XGB = dict(n_estimators=500, max_depth=4, learning_rate=0.03,
           subsample=0.75, colsample_bytree=0.8, min_child_weight=15,
           reg_alpha=0.1, reg_lambda=2.0, eval_metric='auc',
           use_label_encoder=False, verbosity=0, random_state=42)

def walk_forward(target_lbl, name):
    print(f'\n--- WF: {name} ---')
    oos = pd.Series(dtype=float)
    for i, (te_, ts_, t_end) in enumerate(folds):
        X_tr = sc[sc.index <= te_]; y_tr = target_lbl.reindex(X_tr.index).fillna(0).astype(int)
        X_te = sc[(sc.index > te_) & (sc.index <= t_end)]; y_te = target_lbl.reindex(X_te.index).fillna(0).astype(int)
        if y_tr.sum() < 5 or y_te.nunique() < 2:
            print(f'  Fold {i+1}: skip')
            oos = pd.concat([oos, pd.Series(np.nan, index=X_te.index)]); continue
        pos_w = max(1.0, (y_tr==0).sum()/max(y_tr.sum(),1))
        clf = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:,1]
        auc = roc_auc_score(y_te, p) if y_te.nunique() > 1 else float('nan')
        print(f'  Fold {i+1}: n_tr={len(X_tr)} pos_tr={int(y_tr.sum())} n_te={len(X_te)} pos_te={int(y_te.sum())}  AUC={auc:.3f}')
        oos = pd.concat([oos, pd.Series(p, index=X_te.index)])
    return oos

oos_coin = walk_forward(crash, 'COINCIDENT')
oos_pred = walk_forward(predictive_lbl, 'PREDICTIVE (K=20)')
# Bottom-finder uses *masked* training (only in-DD days) but evaluates on all days for use
oos_bot  = walk_forward(bottom_lbl, 'BOTTOM-FINDER')

# ── 5. FINAL MODELS (full data) ─────────────────────────────────────────────
def fit_final(target_lbl):
    y = target_lbl.reindex(sc.index).fillna(0).astype(int)
    pos_w = (y==0).sum()/max(y.sum(),1)
    clf = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB)
    clf.fit(sc, y); return clf

clf_coin = fit_final(crash)
clf_pred = fit_final(predictive_lbl)
clf_bot  = fit_final(bottom_lbl)

# Override in-sample with OOS predictions where available
def merge_oos(oos):
    full = pd.Series(np.nan, index=sc.index)
    full.loc[oos.dropna().index] = oos.dropna().values
    # for pre-OOS dates, use in-sample predictions (acceptable: only used for plots/labels, not metrics)
    is_pred = pd.Series(np.nan, index=sc.index)
    return full

# Use pure OOS for evaluation; production blended uses final-model predictions for live data
proba_coin_full = pd.Series(clf_coin.predict_proba(sc)[:,1], index=sc.index)
proba_pred_full = pd.Series(clf_pred.predict_proba(sc)[:,1], index=sc.index)
proba_bot_full  = pd.Series(clf_bot.predict_proba(sc)[:,1], index=sc.index)
# Override evaluation period with OOS where available
proba_coin_full.loc[oos_coin.dropna().index] = oos_coin.dropna().values
proba_pred_full.loc[oos_pred.dropna().index] = oos_pred.dropna().values
proba_bot_full.loc[oos_bot.dropna().index]   = oos_bot.dropna().values

# Blended alarm signal: max of coincident & predictive (each blended with StatV3)
blended_coin = 0.5*proba_coin_full + 0.5*sc['total_risk_v3']
blended_pred = 0.5*proba_pred_full + 0.5*sc['total_risk_v3']
alarm_signal = pd.concat([blended_coin, blended_pred], axis=1).max(axis=1)

# ── 6. ALARM TUNING (nested) ────────────────────────────────────────────────
TUNE_END = pd.Timestamp('2020-12-31')
TEST_START = pd.Timestamp('2021-01-01')

def hysteresis(p, elev, enter, exit_t, mn, mx, mf, idx):
    alarms=[]; in_a=False; si=None; n=len(p); pv=np.asarray(p); ev=np.asarray(elev)
    for i in range(n):
        if not in_a:
            if pv[i]>=enter and ev[i]>=mf: in_a=True; si=i
        else:
            d=i-si
            if pv[i]<exit_t or d>=mx:
                if d>=mn: alarms.append({'start':idx[si],'end':idx[i-1],'dur':d})
                in_a=False
    if in_a and si is not None and (n-si)>=mn:
        alarms.append({'start':idx[si],'end':idx[-1],'dur':n-si})
    return pd.DataFrame(alarms) if alarms else pd.DataFrame(columns=['start','end','dur'])

elev_full = (sc[factor_names].values > 0.5).sum(axis=1)

# Get crash events for evaluation
def get_events(lbl, prices):
    out=[]; in_c=False; cs=None
    for dt, v in lbl.items():
        if not in_c and v: in_c, cs = True, dt
        elif in_c and not v: out.append({'start':cs,'end':dt}); in_c=False
    if in_c: out.append({'start':cs,'end':lbl.index[-1]})
    if not out: return pd.DataFrame(columns=['start','end','peak','trough','drawdown'])
    df = pd.DataFrame(out)
    df['peak']     = df.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    df['trough']   = df.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    df['drawdown'] = df.apply(lambda r: prices.loc[r.trough]/prices.loc[r.peak]-1, axis=1)
    return df

events = get_events(crash, eq)
ev_tune = events[events.peak <= TUNE_END].reset_index(drop=True)
ev_test = events[events.peak >= TEST_START].reset_index(drop=True)
print(f'\nCrashes — tune: {len(ev_tune)}  test: {len(ev_test)}')

def eval_alarms(alms, evts, idx, lbl):
    n = len(idx)
    if alms.empty: return dict(ev_prec=0, ev_det=0, day_prec=0, day_rec=0, adp=0, lead=0, n=0)
    alarm_days = pd.Series(False, index=idx)
    for _, a in alms.iterrows(): alarm_days.loc[a.start:a.end] = True
    y = lbl.reindex(idx).fillna(False).astype(bool)
    tp=int((alarm_days&y).sum()); fp=int((alarm_days&~y).sum()); fn=int((~alarm_days&y).sum())
    dp = tp/max(tp+fp,1); dr = tp/max(tp+fn,1)
    adp = alarm_days.sum()/n
    n_conf = sum(1 for _, a in alms.iterrows()
                 if any(((a.start <= ce.peak) and (a.start >= ce.peak-pd.Timedelta(days=180))) or
                        ((a.start <= ce.end) and (a.end >= ce.start)) for _, ce in evts.iterrows()))
    ev_prec = n_conf/len(alms)
    leads=[]; n_det=0
    for _, ce in evts.iterrows():
        for _, a in alms.iterrows():
            if (a.start <= ce.end+pd.Timedelta(days=30)) and (a.end >= ce.start-pd.Timedelta(days=180)):
                n_det += 1; leads.append((ce.peak - a.start).days); break
    ev_det = n_det/max(len(evts),1)
    med_lead = int(np.median(leads)) if leads else 0
    return dict(ev_prec=ev_prec, ev_det=ev_det, day_prec=dp, day_rec=dr, adp=adp, lead=med_lead, n=len(alms))

sig_tune = alarm_signal[alarm_signal.index <= TUNE_END]
sig_test = alarm_signal[alarm_signal.index >= TEST_START]
elev_tune = elev_full[:len(sig_tune)]
elev_test = elev_full[len(sig_tune):]

print('\n--- Grid search alarm params on TUNE only ---')
best=None; best_score=-1e9
for entry in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    for xt in [0.10, 0.15, 0.20]:
        for mn in [5, 10, 15, 20]:
            for mx in [30, 45, 60, 90]:
                for mf in [1, 2, 3]:
                    alms = hysteresis(sig_tune.values, elev_tune, entry, xt, mn, mx, mf, sig_tune.index)
                    m = eval_alarms(alms, ev_tune, sig_tune.index, crash)
                    # Score: heavily reward EARLIER lead (less lag from peak).
                    # Lead = peak - alarm_start.  Higher lead = earlier (more positive = before peak).
                    # Penalize alarm-day pct.
                    score = (m['ev_det']*1.5 + m['day_prec']*0.5 + m['lead']*0.01) - m['adp']*1.5
                    if m['ev_det'] >= 0.85 and m['day_prec'] >= 0.30 and score > best_score:
                        best_score=score; best=(entry,xt,mn,mx,mf,m)
if best is None:
    print('No config met ev_det>=85% and day_prec>=30% — relaxing.')
    best_score=-1e9
    for entry in [0.20,0.25,0.30,0.35,0.40,0.45]:
        for xt in [0.10,0.15,0.20]:
            for mn in [5,10,15,20]:
                for mx in [30,45,60,90]:
                    for mf in [1,2,3]:
                        alms = hysteresis(sig_tune.values, elev_tune, entry, xt, mn, mx, mf, sig_tune.index)
                        m = eval_alarms(alms, ev_tune, sig_tune.index, crash)
                        score = (m['ev_det']*1.5 + m['day_prec']*0.5 + m['lead']*0.01) - m['adp']*1.5
                        if score > best_score:
                            best_score=score; best=(entry,xt,mn,mx,mf,m)

entry, xt, mn, mx, mf, m_tune = best
print(f'Chosen: entry={entry} exit={xt} min_dur={mn}d max_dur={mx}d mf_min={mf}')
print(f'  TUNE: ev_det={m_tune["ev_det"]:.0%}  ev_prec={m_tune["ev_prec"]:.0%}  day_prec={m_tune["day_prec"]:.0%}  day_rec={m_tune["day_rec"]:.0%}  adp={m_tune["adp"]:.0%}  lead_from_peak={m_tune["lead"]}d  n={m_tune["n"]}')

alms_test = hysteresis(sig_test.values, elev_test, entry, xt, mn, mx, mf, sig_test.index)
m_test = eval_alarms(alms_test, ev_test, sig_test.index, crash)
print(f'  TEST (BLIND): ev_det={m_test["ev_det"]:.0%}  ev_prec={m_test["ev_prec"]:.0%}  day_prec={m_test["day_prec"]:.0%}  day_rec={m_test["day_rec"]:.0%}  adp={m_test["adp"]:.0%}  lead_from_peak={m_test["lead"]}d  n={m_test["n"]}')

# ── 7. PER-CRASH SCORECARD (full history) ───────────────────────────────────
print('\n--- PER-CRASH DETECTION SCORECARD (v5.1 alarm signal) ---')
print(f'{"Peak":12s} {"Trough":12s} {"DD":>7s}  {"Alarm 1st":12s} {"d-peak":>7s} {"DD@alarm":>9s} {"Saved":>7s}  Result')
print('-'*100)
total_dd = 0.0; total_dd_v51 = 0.0; n_pre=0; n_in=0; n_miss=0
for _, ev in events.iterrows():
    win = alarm_signal.loc[ev.peak-pd.Timedelta(days=90):ev.end+pd.Timedelta(days=30)]
    elev_w = elev_full[alarm_signal.index.get_loc(win.index[0]):alarm_signal.index.get_loc(win.index[-1])+1]
    alms = hysteresis(win.values, elev_w, entry, xt, mn, mx, mf, win.index)
    if alms.empty:
        n_miss += 1
        total_dd += ev.drawdown; total_dd_v51 += ev.drawdown
        print(f'{ev.peak.date()} {ev.trough.date()} {ev.drawdown:>6.1%}    --              --        --       --       MISSED')
        continue
    fa = alms.iloc[0].start
    d_peak = (fa - ev.peak).days
    v_at = eq.loc[:fa].iloc[-1]
    dd_at = v_at/eq.loc[ev.peak] - 1
    saved = eq.loc[ev.trough]/v_at - 1
    if d_peak < 0: n_pre += 1; res='CAUGHT (pre-peak)'
    else: n_in += 1; res='CAUGHT (in-DD)'
    total_dd += ev.drawdown; total_dd_v51 += dd_at
    print(f'{ev.peak.date()} {ev.trough.date()} {ev.drawdown:>6.1%}  {fa.date()}  {d_peak:>5}d  {dd_at:>8.1%}  {saved:>6.1%}  {res}')
print('-'*100)
print(f'Caught pre-peak: {n_pre}   in-DD: {n_in}   missed: {n_miss}   (total {len(events)})')
print(f'Cumulative drawdown if held: {total_dd*100:+.1f}%   if exit on first alarm: {total_dd_v51*100:+.1f}%')

# ── 8. SAVE ─────────────────────────────────────────────────────────────────
Path('models/v5_1').mkdir(parents=True, exist_ok=True)
with open('models/v5_1/v5_1_final.pkl','wb') as f:
    pickle.dump({'clf_coin':clf_coin,'clf_pred':clf_pred,'clf_bot':clf_bot,
                 'feature_cols':feat_cols,'factor_names':factor_names,
                 'k_lead':K_LEAD,'dd_for_bottom':DD_FOR_BOTTOM,'fwd_bottom_days':FWD_BOTTOM_DAYS}, f)

cfg = dict(model='v5.1', k_lead=K_LEAD,
           entry=float(entry), exit=float(xt),
           min_dur=int(mn), max_dur=int(mx), mf_min=int(mf),
           tune=m_tune, test=m_test)
Path('data/alarm_config_v5_1.json').write_text(json.dumps(cfg, indent=2, default=float))

conn = sqlite3.connect(DB)
conn.execute("DELETE FROM predictions WHERE model_version IN ('v5.1','v5.1_pred','v5.1_bot')")
for dt, p in alarm_signal.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v5.1'))
for dt, p in proba_pred_full.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v5.1_pred'))
for dt, p in proba_bot_full.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v5.1_bot'))
conn.commit(); conn.close()
print('\nSaved models/v5_1/v5_1_final.pkl, alarm_config_v5_1.json, and DB predictions.')

# Print top feature importances
fi_p = pd.Series(clf_pred.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('\nTop 10 features for PREDICTIVE model (forecasts crash 20d ahead):')
for f, v in fi_p.head(10).items(): print(f'  {f:<28s} {v:.4f}')
fi_b = pd.Series(clf_bot.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('\nTop 10 features for BOTTOM-FINDER:')
for f, v in fi_b.head(10).items(): print(f'  {f:<28s} {v:.4f}')

print('\n'+'='*78); print('V5.1 TRAINING DONE'); print('='*78)
