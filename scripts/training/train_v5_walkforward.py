"""
V5 Crash Predictor — HONEST walk-forward training.

Key improvements over v4:
  1. Uses NASDAQCOM daily (1987+) as primary equity series — real daily data, not monthly stub.
  2. Adds 4 new daily features: BAA-10Y credit spread, WTI oil, Dollar TWI, EPU daily.
  3. Nested walk-forward: alarm hysteresis params are tuned on TRAIN folds only,
     then evaluated on a held-out TEST fold. No grid-search leakage.
  4. Honest metrics:
       - day-level precision & recall (no event-level inflation)
       - lead measured from MARKET PEAK (not trough)
       - separate TRAIN vs TEST reporting
  5. Head-to-head vs 3 naive baselines (VIX>25, DD>5%, combined). v5 must beat all three.
  6. Data sanity guards (purged holidays/placeholders).
"""
import sys, sqlite3, json, pickle
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve
import xgboost as xgb

from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction

DB = 'data/market_crash.db'
print('='*72); print('V5 HONEST WALK-FORWARD TRAINING'); print('='*72)

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
conn.close()
print(f'\nLoaded {len(ind)} indicator rows  ({ind.index[0].date()} → {ind.index[-1].date()})')

# Use NASDAQ as primary equity series (daily since 1987, 250+ unique values/yr)
eq = ind['nasdaq_close'].copy()
# Restrict to dates where NASDAQ exists — this is our real daily universe
ind = ind.loc[eq.dropna().index]
eq  = ind['nasdaq_close'].ffill()

# Baseline features via existing pipeline (sp500_drawdown etc. computed from sp500_close)
# Note: this still uses sp500_close for some features. Override the drawdown with NASDAQ-based one.
feats = engineer_features_for_prediction(ind)

# CLEAN drawdown from NASDAQ daily
rolling_max = eq.rolling(252, min_periods=50).max()
feats['equity_drawdown']   = (eq / rolling_max - 1)
feats['equity_return_20d'] = eq.pct_change(20)
feats['equity_return_63d'] = eq.pct_change(63)
feats['equity_vol_20d']    = eq.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
feats['equity_vol_63d']    = eq.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)

# New daily FRED features (forward-fill weekly/monthly gaps, bounded by ffill-never-ahead)
for col in ['baa_10y_spread','oil_wti','dollar_twi','epu_daily']:
    if col in ind.columns:
        feats[col] = ind[col].ffill()

# Derived: velocity of key stress signals
feats['baa_chg_20d']   = feats.get('baa_10y_spread', pd.Series(0, index=feats.index)).diff(20)
feats['oil_return_20d']= feats.get('oil_wti', pd.Series(np.nan, index=feats.index)).pct_change(20)
feats['dollar_chg_20d']= feats.get('dollar_twi', pd.Series(np.nan, index=feats.index)).pct_change(20)
feats['epu_chg_20d']   = feats.get('epu_daily', pd.Series(np.nan, index=feats.index)).diff(20)

# ── 2. CRASH LABELS from NASDAQ ──────────────────────────────────────────────
def compute_crash_labels(prices, dd_thresh=0.15, min_td=30):
    prices = prices.ffill(); n = len(prices); pv = prices.values
    rm = prices.rolling(252, min_periods=50).max().values
    below = (pv / rm - 1) <= -dd_thresh
    crash = np.zeros(n, dtype=bool); i = 0
    while i < n:
        if not below[i]: i += 1; continue
        rs = i
        while i < n and below[i]: i += 1
        re = i - 1
        tp = rs + int(np.argmin(pv[rs:re+1]))
        peak_val = rm[rs]; pp = rs
        for j in range(rs-1, max(0, rs-260), -1):
            if pv[j] >= peak_val * 0.998: pp = j; break
        if tp - pp >= min_td: crash[pp:tp+1] = True
    return pd.Series(crash, index=prices.index)

crash_label = compute_crash_labels(eq, dd_thresh=0.15, min_td=30)
print(f'\nCrash days (NASDAQ, ≥15% DD, ≥30TD): {int(crash_label.sum())} / {len(crash_label)} ({crash_label.mean():.1%})')

def get_periods(lbl, prices):
    periods = []; in_c=False; cs=None
    for dt, v in lbl.items():
        if not in_c and v: in_c, cs = True, dt
        elif in_c and not v: periods.append({'start':cs,'end':dt}); in_c=False
    if in_c: periods.append({'start':cs,'end':lbl.index[-1]})
    if not periods: return pd.DataFrame(columns=['start','end','peak','trough','dur_td','drawdown'])
    dfp = pd.DataFrame(periods)
    dfp['peak']     = dfp.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    dfp['trough']   = dfp.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    dfp['dur_td']   = dfp.apply(lambda r: int(lbl.loc[r.start:r.end].sum()), axis=1)
    dfp['drawdown'] = dfp.apply(lambda r: prices.loc[r.trough] / prices.loc[r.peak] - 1, axis=1)
    return dfp

crash_events = get_periods(crash_label, eq)
print(f'Rigorous crash events: {len(crash_events)}')
print(f'{"Peak":>12s}  {"Trough":>12s}  {"Days":>5s}  {"Drawdown":>9s}')
for _, r in crash_events.iterrows():
    print(f'  {r.peak.date()}  {r.trough.date()}  {int(r.dur_td):>5}  {r.drawdown:>8.1%}')

# ── 3. STATV3 FACTOR EXTRACTION ─────────────────────────────────────────────
print('\n--- Extracting StatV3 factor scores (per-row, lookahead-safe) ---')
sv3 = StatisticalModelV3()
factor_names = ['yield_curve','volatility','credit_stress','hy_credit',
                'economic','labor_market','market_momentum','sentiment',
                'financial_conditions','momentum_shock']
score_rows = []
for _, row in feats.iterrows():
    prob, expl = sv3._calculate_crash_probability_with_factors(row)
    sr = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    sr['regime_num']    = {'low':0,'normal':1,'high':2,'extreme':3}.get(expl.get('regime','normal'), 1)
    sr['total_risk_v3'] = prob
    score_rows.append(sr)
sc = pd.DataFrame(score_rows, index=feats.index)

raw_feats = [
    'equity_drawdown','equity_return_20d','equity_return_63d','equity_vol_20d','equity_vol_63d',
    'vix_close','yield_10y_2y','nfci','recession_prob','initial_claims_change_13w',
    'stress_composite','yield_curve_velocity_63d',
    'baa_10y_spread','baa_chg_20d','oil_wti','oil_return_20d','dollar_chg_20d','epu_chg_20d','epu_daily',
]
for f in raw_feats:
    if f in feats.columns:
        sc[f] = feats[f].values

sc = sc.ffill().bfill().fillna(0)
feat_cols = list(sc.columns)
print(f'Feature matrix: {sc.shape}  ({len(feat_cols)} features)')

# ── 4. WALK-FORWARD (honest) ─────────────────────────────────────────────────
# Use NASDAQ history starting 1990 (VIX starts 1990); train expanding.
# Evaluation windows chosen to cover distinct regimes:
folds = [
    (pd.Timestamp('1999-12-31'), pd.Timestamp('2000-01-01'), pd.Timestamp('2005-12-31')),  # dotcom crash in test
    (pd.Timestamp('2005-12-31'), pd.Timestamp('2006-01-01'), pd.Timestamp('2012-12-31')),  # GFC in test
    (pd.Timestamp('2012-12-31'), pd.Timestamp('2013-01-01'), pd.Timestamp('2020-12-31')),  # COVID in test
    (pd.Timestamp('2020-12-31'), pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31')),  # 2022 bear + 2025 + current
]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.75, colsample_bytree=0.8, min_child_weight=15,
    reg_alpha=0.1, reg_lambda=2.0,
    eval_metric='auc', use_label_encoder=False, verbosity=0, random_state=42,
)

oos_proba = pd.Series(dtype=float)
print('\n--- Walk-forward CV ---')
for i, (tr_end, te_start, te_end) in enumerate(folds):
    X_tr = sc[sc.index <= tr_end]; y_tr = crash_label[crash_label.index <= tr_end].astype(int)
    X_te = sc[(sc.index > tr_end) & (sc.index <= te_end)]
    y_te = crash_label[(crash_label.index > tr_end) & (crash_label.index <= te_end)].astype(int)
    if y_tr.sum() < 5 or y_te.nunique() < 2:
        print(f'  Fold {i+1}: skipped (train_crash={y_tr.sum()}, test_unique={y_te.nunique()})')
        oos_proba = pd.concat([oos_proba, pd.Series(np.nan, index=X_te.index)])
        continue
    pos_w = max(1.0, (y_tr == 0).sum() / max(y_tr.sum(), 1))
    clf = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB_PARAMS)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te) if y_te.nunique() > 1 else float('nan')
    print(f'  Fold {i+1}: train→{tr_end.year}  test {te_start.year}-{te_end.year}'
          f'  n_tr={len(X_tr)}  n_te={len(X_te)}  crash_te={y_te.sum()}  AUC={auc:.3f}')
    oos_proba = pd.concat([oos_proba, pd.Series(p_te, index=X_te.index)])

# Final model on all data (for production)
print('\n--- Final model on full history ---')
y_full = crash_label.astype(int)
pos_w  = (y_full == 0).sum() / max(y_full.sum(), 1)
clf_final = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB_PARAMS)
clf_final.fit(sc, y_full)
full_proba = pd.Series(clf_final.predict_proba(sc)[:, 1], index=sc.index)
# Override with OOS where available
valid_oos = oos_proba.dropna()
full_proba.loc[valid_oos.index] = valid_oos.values

# Overall OOS AUC (day-level)
y_oos = y_full.loc[valid_oos.index]
overall_auc = roc_auc_score(y_oos, valid_oos) if y_oos.nunique() > 1 else float('nan')
print(f'\nOverall OOS day-level AUC (1999-2026): {overall_auc:.3f}')

# Blended signal (StatV3 for macro drift + XGBoost for non-linear patterns)
blended = 0.5 * full_proba + 0.5 * sc['total_risk_v3']

# ── 5. NESTED ALARM TUNING (no leakage) ─────────────────────────────────────
# Tune alarm hysteresis on folds 1-3 (1999-2020); evaluate blind on fold 4 (2021-2026).
print('\n--- Nested alarm tuning (train: 1999-2020, test: 2021-2026) ---')
TUNE_END  = pd.Timestamp('2020-12-31')
TEST_START= pd.Timestamp('2021-01-01')

sig_tune = blended[blended.index <= TUNE_END]
sig_test = blended[blended.index >= TEST_START]
sc_tune  = sc.loc[sig_tune.index]; sc_test = sc.loc[sig_test.index]
ev_tune  = crash_events[crash_events.peak <= TUNE_END].reset_index(drop=True)
ev_test  = crash_events[crash_events.peak >= TEST_START].reset_index(drop=True)
print(f'  Tune window crashes: {len(ev_tune)}  |  Test window crashes: {len(ev_test)}')
for _, r in ev_test.iterrows():
    print(f'    TEST crash: peak={r.peak.date()}  trough={r.trough.date()}  dd={r.drawdown:.0%}')

def hysteresis(p, elev, enter, exit_t, min_dur, max_dur, mf_min, idx):
    alarms=[]; in_a=False; si=None; n=len(p)
    pv = np.asarray(p); ev = np.asarray(elev)
    for i in range(n):
        if not in_a:
            if pv[i] >= enter and ev[i] >= mf_min: in_a=True; si=i
        else:
            d = i - si
            if pv[i] < exit_t or d >= max_dur:
                if d >= min_dur: alarms.append({'start':idx[si],'end':idx[i-1],'dur':d})
                in_a=False
    if in_a and si is not None:
        d = n - si
        if d >= min_dur: alarms.append({'start':idx[si],'end':idx[-1],'dur':d})
    return pd.DataFrame(alarms) if alarms else pd.DataFrame(columns=['start','end','dur'])

def eval_alarms(alms, evts, idx):
    """Return (event_precision, event_detection, day_precision, day_recall, alarm_day_pct, median_peak_lead_days)."""
    n_days = len(idx)
    if alms.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    # Day-level
    alarm_days = pd.Series(False, index=idx)
    for _, a in alms.iterrows():
        alarm_days.loc[a.start:a.end] = True
    # Reindex crash label to this window
    y = crash_label.reindex(idx).fillna(False).astype(bool)
    tp = int((alarm_days & y).sum()); fp = int((alarm_days & ~y).sum()); fn = int((~alarm_days & y).sum())
    day_prec = tp / max(tp+fp, 1); day_rec = tp / max(tp+fn, 1)
    alarm_day_pct = alarm_days.sum() / n_days
    # Event-level: an alarm confirms if it overlaps any crash or starts within 180 days before peak
    n_conf = 0
    for _, a in alms.iterrows():
        for _, ce in evts.iterrows():
            pre_peak_window = (a.start <= ce.peak) and (a.start >= ce.peak - pd.Timedelta(days=180))
            overlap = (a.start <= ce.end) and (a.end >= ce.start)
            if pre_peak_window or overlap:
                n_conf += 1; break
    ev_prec = n_conf / len(alms)
    # Detection + lead FROM PEAK (not trough)
    n_det = 0; peak_leads = []
    for _, ce in evts.iterrows():
        for _, a in alms.iterrows():
            if (a.start <= ce.end + pd.Timedelta(days=30)) and (a.end >= ce.start - pd.Timedelta(days=180)):
                n_det += 1
                peak_leads.append((ce.peak - a.start).days)
                break
    ev_det = n_det / max(len(evts), 1)
    med_lead = int(np.median(peak_leads)) if peak_leads else 0
    return ev_prec, ev_det, day_prec, day_rec, alarm_day_pct, med_lead

elev_full = (sc[factor_names].values > 0.5).sum(axis=1)
elev_tune = elev_full[:len(sc_tune)]
elev_test = elev_full[len(sc_tune):]

# Grid search on TUNE period only
best = None; best_score = -1e9; scans = []
for entry in [0.25, 0.30, 0.35, 0.40, 0.45]:
    for xt in [0.10, 0.15, 0.20]:
        for mn in [5, 10, 15, 20]:
            for mx in [30, 45, 60, 90]:
                for mf in [1, 2, 3]:
                    alms = hysteresis(sig_tune.values, elev_tune, entry, xt, mn, mx, mf, sig_tune.index)
                    ep, ed, dp, dr, adp, lead = eval_alarms(alms, ev_tune, sig_tune.index)
                    # Objective: maximize detection, minimize alarm-day %, prefer precision
                    score = (ed * 2.0) + dp - (adp * 2.0)
                    scans.append(dict(entry=entry, xt=xt, mn=mn, mx=mx, mf=mf,
                                       ev_prec=ep, ev_det=ed, day_prec=dp, day_rec=dr,
                                       adp=adp, lead=lead, score=score))
                    if score > best_score and ed >= 0.75 and dp >= 0.30:
                        best_score = score
                        best = (entry, xt, mn, mx, mf)

if best is None:
    # Fallback: best by score unconstrained
    rec = max(scans, key=lambda r: r['score'])
    best = (rec['entry'], rec['xt'], rec['mn'], rec['mx'], rec['mf'])
    print(f'  No config met ev_det≥75% & day_prec≥30%; using best-by-score fallback.')

entry, xt, mn, mx, mf = best
print(f'  Chosen alarm config (from TUNE grid): entry={entry} exit={xt} min={mn}d max={mx}d mf_min={mf}')

# Evaluate on TUNE (in-sample for alarm) and TEST (blind)
alms_tune = hysteresis(sig_tune.values, elev_tune, *best, sig_tune.index)
alms_test = hysteresis(sig_test.values, elev_test, *best, sig_test.index)
m_tune = eval_alarms(alms_tune, ev_tune, sig_tune.index)
m_test = eval_alarms(alms_test, ev_test, sig_test.index)

def print_metrics(tag, m, n_alms):
    ep, ed, dp, dr, adp, lead = m
    print(f'  {tag}:  ev_prec={ep:.0%}  ev_det={ed:.0%}  day_prec={dp:.0%}  day_rec={dr:.0%}  alarm_days={adp:.0%}  lead_from_peak={lead}d  n_alarms={n_alms}')

print('\n=== ALARM METRICS (honest nested-CV) ===')
print_metrics('TUNE (1999-2020, in-sample for alarm params)', m_tune, len(alms_tune))
print_metrics('TEST (2021-2026, BLIND — first look)         ', m_test, len(alms_test))

# ── 6. HEAD-TO-HEAD VS NAIVE BASELINES (on TEST window) ─────────────────────
print('\n=== HEAD-TO-HEAD vs NAIVE BASELINES (TEST 2021-2026) ===')
y_test_days = crash_label.loc[sig_test.index].astype(bool).values
idx_test = sig_test.index

vix = feats['vix_close'].reindex(idx_test).ffill().values
dd  = feats['equity_drawdown'].reindex(idx_test).ffill().values

# Construct v5 alarm day array
v5_days = pd.Series(False, index=idx_test)
for _, a in alms_test.iterrows():
    v5_days.loc[a.start:a.end] = True

vix_sustained = (pd.Series(vix > 25).rolling(10, min_periods=10).sum() >= 10).fillna(False).values

baselines = {
    'VIX>25 (10d sustained)': vix_sustained,
    'DD>5%':                  (dd < -0.05),
    'VIX>25 OR DD>5%':        ((vix > 25) | (dd < -0.05)),
    'VIX>30 AND DD>10%':      ((vix > 30) & (dd < -0.10)),
    f'v5 hysteresis':         v5_days.values,
}

print(f'  {"Rule":28s}  {"alarm_days%":>11s}  {"day_prec":>9s}  {"day_rec":>8s}  {"F1":>6s}')
for name, bs in baselines.items():
    bs = np.asarray(bs).astype(bool)
    adp = bs.mean()
    tp = int((bs & y_test_days).sum()); fp = int((bs & ~y_test_days).sum()); fn = int((~bs & y_test_days).sum())
    p = tp / max(tp+fp,1); r = tp / max(tp+fn,1); f1 = 2*p*r/max(p+r, 1e-9)
    print(f'  {name:28s}  {adp:>10.1%}  {p:>8.1%}  {r:>7.1%}  {f1:>6.2f}')

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
Path('models/v5').mkdir(parents=True, exist_ok=True)
with open('models/v5/v5_final.pkl','wb') as f:
    pickle.dump({'clf':clf_final,'feature_cols':feat_cols,'factor_names':factor_names}, f)

alarm_cfg = dict(
    model='v5', entry=float(entry), exit=float(xt),
    min_dur=int(mn), max_dur=int(mx), mf_min=int(mf),
    overall_oos_auc=round(float(overall_auc), 3),
    tune_metrics=dict(zip(['ev_prec','ev_det','day_prec','day_rec','alarm_day_pct','lead_from_peak'], m_tune)),
    test_metrics=dict(zip(['ev_prec','ev_det','day_prec','day_rec','alarm_day_pct','lead_from_peak'], m_test)),
)
Path('data/alarm_config_v5.json').write_text(json.dumps(alarm_cfg, indent=2, default=float))
print('\nSaved models/v5/v5_final.pkl  &  data/alarm_config_v5.json')

# Write v5 predictions to DB
conn = sqlite3.connect(DB)
conn.execute("DELETE FROM predictions WHERE model_version='v5'")
for dt, p in blended.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v5'))
conn.commit(); conn.close()
print(f'Stored {len(blended)} v5 predictions in DB.')

# Feature importance
fi = pd.Series(clf_final.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('\n--- Top 15 feature importances (final model) ---')
for f, v in fi.head(15).items():
    print(f'  {f:<32s} {v:.4f}')

print('\n'+'='*72); print('V5 TRAINING COMPLETE — see metrics above for honest performance.'); print('='*72)
