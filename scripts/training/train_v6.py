"""
v6 — predictive crash detector + bottom-finder, with hard kill-criterion vs v5.

Adds these mathematically motivated features on top of v5's set:

  TAIL-RISK PREMIUM (options-implied; SHOULD lead the market)
    skew_level           = SKEW (1990+ daily)
    skew_chg_20d         = SKEW - SKEW.shift(20)
    skew_x_vix           = SKEW * VIXCLS / 100  (joint tail-vol stress)
    vix9d_vix_ratio      = VIX9D / VIXCLS  (>1 => stress NOW > forward; backwardation)

  BREADTH / LEADERSHIP (cross-sectional, leadership narrowing => tops)
    rsp_spy_60d          = RSP/SPY return spread 60d (eq-weight vs cap-weight)
    iwm_spy_60d          = IWM/SPY return spread 60d (small vs large)

  SECTOR ROTATION (defensives outperform => risk-off, often coincident or slight lead)
    defensive_ratio_20d  = (XLU+XLP+XLV) / (XLK+XLY+XLF), 20d change

  CROSS-ASSET STRESS
    hyg_spy_20d          = HYG return - SPY return (20d) — credit cracks first
    tlt_spy_60d          = TLT return - SPY return (60d) — flight to bonds
    gld_spy_60d          = GLD return - SPY return (60d) — flight to gold

Discipline:
  - Walk-forward identical to v5 (4 folds; 1999-2020 tune; 2021-2026 BLIND)
  - All preprocessing inside fold
  - Final v6 alarm signal blended same way as v5: 0.5 * xgb + 0.5 * statv3
  - KILL-CRITERION (any of these => v6 is shelved):
      blind day_precision   < v5_benchmark blind day_precision  - 5 pp
      blind ev_det          < v5_benchmark blind ev_det         - 5 pp
      blind median lead     < v5_benchmark blind median lead    - 5 days
      blind backtest CAGR   < v5_benchmark blind CAGR           - 2 pp
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
K_LEAD          = 20
DD_FOR_BOTTOM   = -0.10
FWD_BOT_DAYS    = 30      # tighter than v5.1's 60
BOT_FWD_RET     = 0.05    # require +5% in 30d (vs +10% in 60d)

print('='*80); print(f'v6 TRAINING (K_LEAD={K_LEAD}d, bottom: {FWD_BOT_DAYS}d/+{BOT_FWD_RET:.0%})'); print('='*80)

# 1. LOAD ─────────────────────────────────────────────
conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
conn.close()

eq = ind['nasdaq_close'].copy()
ind = ind.loc[eq.dropna().index]
eq = ind['nasdaq_close'].ffill()

feats = engineer_features_for_prediction(ind)

# v5 equity features
rm = eq.rolling(252, min_periods=50).max()
feats['equity_drawdown']   = eq / rm - 1
feats['equity_return_5d']  = eq.pct_change(5)
feats['equity_return_20d'] = eq.pct_change(20)
feats['equity_return_63d'] = eq.pct_change(63)
feats['equity_vol_20d']    = eq.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
feats['equity_vol_63d']    = eq.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)

for col in ['baa_10y_spread','oil_wti','dollar_twi','epu_daily']:
    if col in ind.columns: feats[col] = ind[col].ffill()

# === v6 NEW FEATURES ===
# Tail-risk premium
skew = ind['v6_skew'].ffill() if 'v6_skew' in ind.columns else pd.Series(np.nan, index=eq.index)
vix  = ind['vix_close'].ffill() if 'vix_close' in ind.columns else pd.Series(np.nan, index=eq.index)
feats['skew_level']     = skew
feats['skew_chg_20d']   = skew - skew.shift(20)
feats['skew_x_vix']     = (skew * vix / 100).where(skew.notna() & vix.notna())

vix9d = ind['v6_vix9d'].ffill() if 'v6_vix9d' in ind.columns else pd.Series(np.nan, index=eq.index)
ratio = (vix9d / vix).where(vix9d.notna() & vix.notna() & (vix > 0))
# Pre-2011 fill with 1.0 (neutral) since VIX9D didn't exist; XGB can use NA-handling
feats['vix9d_vix_ratio'] = ratio

# Breadth proxies
def lret(s, n):
    s = s.ffill(); return np.log(s/s.shift(n))
spy = ind.get('v6_spy', pd.Series(np.nan, index=eq.index))
rsp = ind.get('v6_rsp', pd.Series(np.nan, index=eq.index))
iwm = ind.get('v6_iwm', pd.Series(np.nan, index=eq.index))
feats['rsp_spy_60d'] = (lret(rsp, 60) - lret(spy, 60))
feats['iwm_spy_60d'] = (lret(iwm, 60) - lret(spy, 60))

# Sector rotation (defensives vs cyclicals)
xlu = ind.get('v6_xlu'); xlp = ind.get('v6_xlp'); xlv = ind.get('v6_xlv')
xlk = ind.get('v6_xlk'); xly = ind.get('v6_xly'); xlf = ind.get('v6_xlf')
def avg_lret(syms, n):
    series_list = [lret(s, n) for s in syms if s is not None]
    if not series_list: return pd.Series(np.nan, index=eq.index)
    return pd.concat(series_list, axis=1).mean(axis=1)
def_ret_20d = avg_lret([xlu, xlp, xlv], 20)
cyc_ret_20d = avg_lret([xlk, xly, xlf], 20)
feats['defensive_ratio_20d'] = def_ret_20d - cyc_ret_20d

# Cross-asset stress
hyg = ind.get('v6_hyg'); tlt = ind.get('v6_tlt'); gld = ind.get('v6_gld')
feats['hyg_spy_20d'] = (lret(hyg, 20) - lret(spy, 20)) if hyg is not None else pd.Series(np.nan, index=eq.index)
feats['tlt_spy_60d'] = (lret(tlt, 60) - lret(spy, 60)) if tlt is not None else pd.Series(np.nan, index=eq.index)
feats['gld_spy_60d'] = (lret(gld, 60) - lret(spy, 60)) if gld is not None else pd.Series(np.nan, index=eq.index)

# 2. LABELS ─────────────────────────────────────────────
def compute_crash_labels(prices, dd=0.15, min_td=30):
    p = prices.ffill(); n = len(p); pv = p.values
    rm = p.rolling(252, min_periods=50).max().values
    below = (pv/rm - 1) <= -dd
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
    return pd.Series(crash, index=p.index)

crash = compute_crash_labels(eq)

# Predictive label
crash_arr = crash.values; n = len(crash_arr)
predictive = np.zeros(n, dtype=bool)
for i in range(n):
    j_end = min(i + K_LEAD + 1, n)
    if crash_arr[i:j_end].any(): predictive[i] = True
predictive_lbl = pd.Series(predictive, index=crash.index)

# Bottom label: in DD ≥ 10%, fwd 30d return > +5%
fwd_ret = eq.pct_change(FWD_BOT_DAYS).shift(-FWD_BOT_DAYS)
in_dd = feats['equity_drawdown'] <= DD_FOR_BOTTOM
bottom_lbl = ((fwd_ret > BOT_FWD_RET) & in_dd).astype(int)

print(f'Coincident crash days:   {int(crash.sum())}  ({crash.mean():.1%})')
print(f'Predictive (K=20) days:  {int(predictive_lbl.sum())}')
print(f'Bottom positives:        {int(bottom_lbl.sum())}  (in-DD days: {int(in_dd.sum())})')

# 3. STATV3 ─────────────────────────────────────────────
print('\n--- StatV3 factors ---')
sv3 = StatisticalModelV3()
factor_names = ['yield_curve','volatility','credit_stress','hy_credit','economic',
                'labor_market','market_momentum','sentiment','financial_conditions','momentum_shock']
rows = []
for _, row in feats.iterrows():
    p, expl = sv3._calculate_crash_probability_with_factors(row)
    r = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    r['regime_num'] = {'low':0,'normal':1,'high':2,'extreme':3}.get(expl.get('regime','normal'), 1)
    r['total_risk_v3'] = p
    rows.append(r)
sc = pd.DataFrame(rows, index=feats.index)

raw_feats = [
    'equity_drawdown','equity_return_5d','equity_return_20d','equity_return_63d',
    'equity_vol_20d','equity_vol_63d',
    'vix_close','yield_10y_2y','nfci','recession_prob','initial_claims_change_13w',
    'stress_composite','yield_curve_velocity_63d',
    'baa_10y_spread','oil_wti','dollar_twi','epu_daily',
    # v6 new
    'skew_level','skew_chg_20d','skew_x_vix','vix9d_vix_ratio',
    'rsp_spy_60d','iwm_spy_60d','defensive_ratio_20d',
    'hyg_spy_20d','tlt_spy_60d','gld_spy_60d',
]
for f in raw_feats:
    if f in feats.columns: sc[f] = feats[f].values

# IMPORTANT: do NOT ffill/bfill v6 features beyond their natural availability,
# because XGBoost handles NaN natively. Only fill StatV3 factor cols.
for c in factor_names + ['total_risk_v3','regime_num']:
    sc[c] = sc[c].ffill().bfill().fillna(0)
feat_cols = list(sc.columns)
print(f'Feature matrix: {sc.shape}  ({len(feat_cols)} features)')

# 4. WALK-FORWARD ─────────────────────────────────────────────
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

def wf(target_lbl, name):
    print(f'\n--- WF: {name} ---')
    oos = pd.Series(dtype=float)
    for i, (te_, ts_, t_end) in enumerate(folds):
        Xtr = sc[sc.index <= te_]; ytr = target_lbl.reindex(Xtr.index).fillna(0).astype(int)
        Xte = sc[(sc.index > te_) & (sc.index <= t_end)]; yte = target_lbl.reindex(Xte.index).fillna(0).astype(int)
        if ytr.sum() < 5 or yte.nunique() < 2:
            print(f'  Fold {i+1}: skip'); oos = pd.concat([oos, pd.Series(np.nan, index=Xte.index)]); continue
        pos_w = max(1.0, (ytr == 0).sum() / max(ytr.sum(), 1))
        clf = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p) if yte.nunique() > 1 else float('nan')
        print(f'  Fold {i+1}: n_tr={len(Xtr)} pos_tr={int(ytr.sum())} n_te={len(Xte)} pos_te={int(yte.sum())}  AUC={auc:.3f}')
        oos = pd.concat([oos, pd.Series(p, index=Xte.index)])
    return oos

oos_coin = wf(crash, 'COINCIDENT')
oos_pred = wf(predictive_lbl, 'PREDICTIVE (K=20)')
oos_bot  = wf(bottom_lbl, 'BOTTOM-FINDER')

# Final fit (full)
def fit_final(lbl):
    y = lbl.reindex(sc.index).fillna(0).astype(int)
    pw = (y == 0).sum() / max(y.sum(), 1)
    clf = xgb.XGBClassifier(scale_pos_weight=pw, **XGB)
    clf.fit(sc, y); return clf

clf_coin = fit_final(crash)
clf_pred = fit_final(predictive_lbl)
clf_bot  = fit_final(bottom_lbl)

# Production probas: use OOS where available, in-sample elsewhere
def merge(clf, oos_series):
    p = pd.Series(clf.predict_proba(sc)[:,1], index=sc.index)
    p.loc[oos_series.dropna().index] = oos_series.dropna().values
    return p
proba_coin = merge(clf_coin, oos_coin)
proba_pred = merge(clf_pred, oos_pred)
proba_bot  = merge(clf_bot,  oos_bot)

# v6 alarm signal: max of (predictive blended, coincident blended)
blended_coin = 0.5 * proba_coin + 0.5 * sc['total_risk_v3']
blended_pred = 0.5 * proba_pred + 0.5 * sc['total_risk_v3']
alarm_signal = pd.concat([blended_coin, blended_pred], axis=1).max(axis=1)

# 5. ALARM TUNING — same hysteresis grid as v5 ─────────────────────────────
TUNE_END   = pd.Timestamp('2020-12-31')
TEST_START = pd.Timestamp('2021-01-01')

def hyst(p, idx, en, ex, mn, mx, mf=None, elev=None):
    a = []; on = False; si = None; pv = np.asarray(p)
    for i in range(len(pv)):
        if not on:
            if pv[i] >= en and (elev is None or elev[i] >= mf): on = True; si = i
        else:
            d = i - si
            if pv[i] < ex or d >= mx:
                if d >= mn: a.append((idx[si], idx[i-1]))
                on = False
    if on and si is not None and (len(pv) - si) >= mn: a.append((idx[si], idx[-1]))
    return a

def get_events(lbl, prices):
    out = []; on = False; cs = None
    for d, v in lbl.items():
        if not on and v: on, cs = True, d
        elif on and not v: out.append({'start': cs, 'end': d}); on = False
    if on: out.append({'start': cs, 'end': lbl.index[-1]})
    df = pd.DataFrame(out)
    if df.empty: return df
    df['peak']     = df.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    df['trough']   = df.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    df['drawdown'] = df.apply(lambda r: prices.loc[r.trough]/prices.loc[r.peak] - 1, axis=1)
    return df

events = get_events(crash, eq)
ev_tune = events[events.peak <= TUNE_END].reset_index(drop=True)
ev_test = events[events.peak >= TEST_START].reset_index(drop=True)

elev_full = (sc[factor_names].values > 0.5).sum(axis=1)

def eval_alms(alms, evts, idx, lbl):
    if not alms: return dict(ev_det=0, ev_prec=0, day_prec=0, day_rec=0, adp=0, lead=0, n=0)
    am = pd.Series(False, index=idx)
    for s, e in alms: am.loc[s:e] = True
    y = lbl.reindex(idx).fillna(False).astype(bool)
    tp = int((am & y).sum()); fp = int((am & ~y).sum()); fn = int((~am & y).sum())
    dp = tp / max(tp + fp, 1); dr = tp / max(tp + fn, 1); adp = am.sum() / len(idx)
    n_conf = sum(1 for s, e in alms
                 if any(((s <= ce.peak) and (s >= ce.peak - pd.Timedelta(days=180))) or
                        ((s <= ce.end) and (e >= ce.start)) for _, ce in evts.iterrows()))
    leads = []; n_det = 0
    for _, ce in evts.iterrows():
        for s, e in alms:
            if (s <= ce.end + pd.Timedelta(days=30)) and (e >= ce.start - pd.Timedelta(days=180)):
                n_det += 1; leads.append((ce.peak - s).days); break
    return dict(ev_det=n_det / max(len(evts), 1), ev_prec=n_conf / len(alms),
                day_prec=dp, day_rec=dr, adp=adp,
                lead=int(np.median(leads)) if leads else 0, n=len(alms))

sig_tune = alarm_signal[alarm_signal.index <= TUNE_END]
sig_test = alarm_signal[alarm_signal.index >= TEST_START]
elev_tune = elev_full[:len(sig_tune)]
elev_test = elev_full[len(sig_tune):]

print('\n--- Grid search (TUNE only) ---')
best = None; best_score = -1e18
for entry in [0.30, 0.35, 0.40, 0.45, 0.50]:
    for ex in [0.15, 0.20, 0.25]:
        for mn in [10, 15, 20]:
            for mx in [30, 45, 60]:
                for mf in [1, 2]:
                    alms = hyst(sig_tune.values, sig_tune.index, entry, ex, mn, mx, mf, elev_tune)
                    m = eval_alms(alms, ev_tune, sig_tune.index, crash)
                    score = (m['ev_det']*1.5 + m['day_prec']*1.0 + m['lead']*0.005) - m['adp']*1.0
                    if m['ev_det'] >= 0.85 and m['day_prec'] >= 0.50 and score > best_score:
                        best_score = score
                        best = (entry, ex, mn, mx, mf, m)

if best is None:
    print('No config met ev_det>=85% & day_prec>=50%. Relaxing.')
    for entry in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for ex in [0.15, 0.20, 0.25]:
            for mn in [10, 15, 20]:
                for mx in [30, 45, 60]:
                    for mf in [1, 2]:
                        alms = hyst(sig_tune.values, sig_tune.index, entry, ex, mn, mx, mf, elev_tune)
                        m = eval_alms(alms, ev_tune, sig_tune.index, crash)
                        score = (m['ev_det']*1.5 + m['day_prec']*1.0 + m['lead']*0.005) - m['adp']*1.0
                        if score > best_score: best_score, best = score, (entry, ex, mn, mx, mf, m)

entry, ex, mn, mx, mf, m_tune = best
print(f'Chosen: entry={entry} exit={ex} min_dur={mn}d max_dur={mx}d mf_min={mf}')
print(f'  TUNE: ev_det={m_tune["ev_det"]:.0%}  ev_prec={m_tune["ev_prec"]:.0%}  day_prec={m_tune["day_prec"]:.0%}  day_rec={m_tune["day_rec"]:.0%}  adp={m_tune["adp"]:.0%}  lead={m_tune["lead"]}d  n={m_tune["n"]}')

alms_test = hyst(sig_test.values, sig_test.index, entry, ex, mn, mx, mf, elev_test)
m_test = eval_alms(alms_test, ev_test, sig_test.index, crash)
print(f'  TEST: ev_det={m_test["ev_det"]:.0%}  ev_prec={m_test["ev_prec"]:.0%}  day_prec={m_test["day_prec"]:.0%}  day_rec={m_test["day_rec"]:.0%}  adp={m_test["adp"]:.0%}  lead={m_test["lead"]}d  n={m_test["n"]}')

# 6. SAVE artifacts (NOT YET PROMOTED) ──────────────────────────────────
Path('models/v6').mkdir(parents=True, exist_ok=True)
with open('models/v6/v6_final.pkl', 'wb') as f:
    pickle.dump({'clf_coin':clf_coin,'clf_pred':clf_pred,'clf_bot':clf_bot,
                 'feature_cols':feat_cols,'factor_names':factor_names,
                 'k_lead':K_LEAD,'fwd_bot_days':FWD_BOT_DAYS,'bot_fwd_ret':BOT_FWD_RET}, f)

cfg = dict(model='v6', entry=float(entry), exit=float(ex), min_dur=int(mn),
           max_dur=int(mx), mf_min=int(mf), tune=m_tune, test=m_test)
Path('data/alarm_config_v6.json').write_text(json.dumps(cfg, indent=2, default=float))

conn = sqlite3.connect(DB)
conn.execute("DELETE FROM predictions WHERE model_version IN ('v6','v6_pred','v6_bot')")
for dt, p in alarm_signal.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v6'))
for dt, p in proba_pred.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v6_pred'))
for dt, p in proba_bot.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v6_bot'))
conn.commit(); conn.close()

# Top features
fi_p = pd.Series(clf_pred.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('\nTop 15 features for PREDICTIVE model:')
for f, v in fi_p.head(15).items(): print(f'  {f:<26s} {v:.4f}')
fi_b = pd.Series(clf_bot.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('\nTop 15 features for BOTTOM-FINDER:')
for f, v in fi_b.head(15).items(): print(f'  {f:<26s} {v:.4f}')

print('\n'+'='*80); print('v6 TRAINING DONE — Run scripts/utils/v6_kill_or_promote.py to validate vs v5'); print('='*80)
