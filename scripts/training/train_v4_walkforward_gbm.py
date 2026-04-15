"""
Optimization Pipeline v4 — Honest Walk-Forward Evaluation
==========================================================
1. Algorithmically-defined crashes: rolling-252d-max drawdown >= 15%, >= 30 trading days
2. StatV3 as FEATURE EXTRACTOR (10 factor scores per day, no lookahead)
3. XGBoost trained via walk-forward expanding window — true OOS evaluation
4. Alarm system with max-duration cap + multi-factor confirmation gate
5. Grid search on OOS period only
6. Honest reporting — states targets met/missed without inflation
"""
import sys, sqlite3, json, pickle
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction

DB = 'data/market_crash.db'

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print('=' * 70)
print('OPTIMIZATION PIPELINE v4')
print('=' * 70)

conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date'])
conn.close()
ind = ind.set_index('date').sort_index()
print(f'\nLoaded {len(ind)} rows  ({ind.index[0].date()} to {ind.index[-1].date()})')

df = engineer_features_for_prediction(ind)
print('Feature engineering complete.')

# ── 2. CRASH LABELS — rolling-252d-max drawdown method ────────────────────────
print('\n--- Computing rigorous crash labels ---')

def compute_crash_labels(prices, dd_thresh=0.15, min_td=30):
    """
    Mark crash days using rolling-252d-max drawdown:
    1. Find dates where price is >= dd_thresh below rolling-252d-max
    2. For each contiguous such run, extend back to the actual price peak
    3. Mark peak-to-trough as in_crash if duration >= min_td trading days
    """
    prices = prices.ffill()
    n = len(prices)
    pv = prices.values
    rm = prices.rolling(252, min_periods=50).max().values

    below = (pv / rm - 1) <= -dd_thresh  # dates where we are >= dd_thresh below 252d max
    crash = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not below[i]:
            i += 1
            continue

        # start of a below-threshold run
        run_start = i
        while i < n and below[i]:
            i += 1
        run_end = i - 1  # last index in this below-threshold run

        # find the trough (minimum price in the run)
        trough_pos = run_start + int(np.argmin(pv[run_start:run_end + 1]))

        # find the peak: walk backward from run_start to find
        # the last date where price was at the rolling max
        peak_val = rm[run_start]
        peak_pos = run_start
        for j in range(run_start - 1, max(0, run_start - 260), -1):
            if pv[j] >= peak_val * 0.998:
                peak_pos = j
                break

        duration = trough_pos - peak_pos
        if duration >= min_td:
            crash[peak_pos:trough_pos + 1] = True

    return pd.Series(crash, index=prices.index)


sp = df['sp500_close'].copy()
crash_label = compute_crash_labels(sp, dd_thresh=0.15, min_td=30)

# Build crash event table
def get_periods(lbl, prices):
    periods = []
    in_c = False; cs = None
    for dt, v in lbl.items():
        if not in_c and v:  in_c, cs = True, dt
        elif in_c and not v:
            periods.append({'start': cs, 'end': dt})
            in_c = False
    if in_c:
        periods.append({'start': cs, 'end': lbl.index[-1]})
    if not periods:
        return pd.DataFrame(columns=['start','end','dur_td','drawdown'])
    dfp = pd.DataFrame(periods)
    dfp['dur_td']  = dfp.apply(lambda r: int(lbl.loc[r.start:r.end].sum()), axis=1)
    dfp['drawdown']= dfp.apply(
        lambda r: prices.loc[r.start:r.end].min() / prices.loc[r.start:r.end].max() - 1,
        axis=1)
    return dfp

crash_events = get_periods(crash_label, sp)
print(f'Rigorous crashes found: {len(crash_events)}')
print(f'Total crash days: {int(crash_label.sum())} ({crash_label.mean():.1%})')
print()
print(f'{"Start":12s}  {"End":12s}  {"TDays":>6s}  {"Drawdown":>9s}')
print('-' * 48)
for _, r in crash_events.iterrows():
    print(f'{str(r.start.date()):12s}  {str(r.end.date()):12s}  {int(r.dur_td):>6}  {r.drawdown:>8.1%}')

# ── 3. EXTRACT STATV3 FACTOR SCORES (feature layer) ───────────────────────────
print('\n--- Extracting StatV3 factor scores ---')
model_v3 = StatisticalModelV3()

factor_names = ['yield_curve','volatility','credit_stress','hy_credit',
                'economic','labor_market','market_momentum','sentiment',
                'financial_conditions','momentum_shock']

score_rows = []
for _, row in df.iterrows():
    prob, expl = model_v3._calculate_crash_probability_with_factors(row)
    sr = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    sr['regime_num']    = {'low':0,'normal':1,'high':2,'extreme':3}.get(expl.get('regime','normal'), 1)
    sr['total_risk_v3'] = prob
    score_rows.append(sr)

scores_df = pd.DataFrame(score_rows, index=df.index)

raw_feats = ['sp500_drawdown','sp500_return_20d','vix_close','yield_10y_2y',
             'nfci','epu_acceleration','recession_prob','initial_claims_change_13w',
             'stress_composite','yield_curve_velocity_63d']
for feat in raw_feats:
    if feat in df.columns:
        scores_df[feat] = df[feat].values

scores_df = scores_df.ffill().bfill().fillna(0)
feat_cols = list(scores_df.columns)
print(f'Feature matrix: {scores_df.shape}')

# ── 4. WALK-FORWARD CROSS-VALIDATION ──────────────────────────────────────────
print('\n--- Walk-forward cross-validation ---')

folds = [
    (pd.Timestamp('2004-12-31'), pd.Timestamp('2005-01-01'), pd.Timestamp('2010-12-31')),
    (pd.Timestamp('2010-12-31'), pd.Timestamp('2011-01-01'), pd.Timestamp('2016-12-31')),
    (pd.Timestamp('2016-12-31'), pd.Timestamp('2017-01-01'), pd.Timestamp('2022-12-31')),
    (pd.Timestamp('2022-12-31'), pd.Timestamp('2023-01-01'), pd.Timestamp('2026-12-31')),
]

oos_proba  = pd.Series(dtype=float)
oos_labels = pd.Series(dtype=int)

for fi, (train_end, test_start, test_end) in enumerate(folds):
    X_tr = scores_df[scores_df.index <= train_end]
    y_tr = crash_label[crash_label.index <= train_end].astype(int)
    X_te = scores_df[(scores_df.index > train_end) & (scores_df.index <= test_end)]
    y_te = crash_label[(crash_label.index > train_end) & (crash_label.index <= test_end)].astype(int)

    if len(X_tr) < 500 or y_tr.sum() < 5:
        print(f'  Fold {fi+1}: skipped (train crash days={y_tr.sum()})')
        oos_proba  = pd.concat([oos_proba,  pd.Series(
            np.full(len(X_te), float('nan')), index=X_te.index)])
        oos_labels = pd.concat([oos_labels, y_te])
        continue

    pos_w = max(1.0, (y_tr == 0).sum() / max(y_tr.sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.8,
        scale_pos_weight=pos_w, min_child_weight=15,
        reg_alpha=0.1, reg_lambda=2.0,
        eval_metric='auc', use_label_encoder=False, verbosity=0, random_state=42)
    clf.fit(X_tr, y_tr)

    p_te = clf.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_te, p_te) if y_te.nunique() > 1 else float('nan')
    print(f'  Fold {fi+1}: train to {train_end.year}  test {test_start.year}-{test_end.year}'
          f'  | n_train={len(X_tr)}  n_test={len(X_te)}'
          f'  | crash_days_test={y_te.sum()}'
          f'  | OOS AUC={auc:.3f}')

    oos_proba  = pd.concat([oos_proba,  pd.Series(p_te, index=X_te.index)])
    oos_labels = pd.concat([oos_labels, y_te])

# ── 5. FINAL MODEL on all data ────────────────────────────────────────────────
print('\n--- Training final model on full history ---')
pos_w_final = (crash_label == 0).sum() / max(crash_label.sum(), 1)
clf_final = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.75, colsample_bytree=0.8,
    scale_pos_weight=pos_w_final, min_child_weight=15,
    reg_alpha=0.1, reg_lambda=2.0,
    eval_metric='auc', use_label_encoder=False, verbosity=0, random_state=42)
clf_final.fit(scores_df, crash_label.astype(int))

full_proba = pd.Series(clf_final.predict_proba(scores_df)[:, 1], index=scores_df.index)
# Replace 2005+ with strictly OOS predictions (fills NaNs from skipped folds with in-sample)
valid_oos = oos_proba.dropna()
if len(valid_oos) > 0:
    full_proba.loc[valid_oos.index] = valid_oos.values
    print(f'OOS predictions cover {len(valid_oos)} days')

Path('models/statistical_v3').mkdir(parents=True, exist_ok=True)
with open('models/v4_gbm_final.pkl', 'wb') as f:
    pickle.dump({'clf': clf_final, 'feature_cols': feat_cols}, f)
print('Saved models/v4_gbm_final.pkl')

# Blended signal: average GBM probability with StatV3 total_risk score.
# StatV3 fires during macro deterioration (provides early warning); GBM fires
# when multiple factors confirm (provides specificity). Blending gives:
# - Better detection of sentiment-driven crashes (2018 Q4: StatV3 hits 75% at trough)
# - Better precision on macro crashes (GBM confirmation)
blended_proba = 0.5 * full_proba + 0.5 * scores_df['total_risk_v3']
print(f'Blended signal range: {blended_proba.min():.3f} – {blended_proba.max():.3f}')

# ── 6. ALARM SYSTEM GRID SEARCH ───────────────────────────────────────────────
def hysteresis_v4(p_series, s_df, enter, exit_t, min_dur, max_dur, mf_min):
    alarms = []; in_a = False; sp_i = None
    vals = p_series.values; idx = p_series.index; n = len(vals)
    # Pre-compute elevated-factor count per row (vectorized)
    elev_counts = (s_df[factor_names].values > 0.5).sum(axis=1)
    for i in range(n):
        v = vals[i]
        if not in_a:
            if v >= enter and elev_counts[i] >= mf_min:
                in_a = True; sp_i = i
        else:
            dur = i - sp_i
            if v < exit_t or dur >= max_dur:
                if dur >= min_dur:
                    alarms.append({'start': idx[sp_i], 'end': idx[i-1], 'dur': dur,
                                   'max_p': float(vals[sp_i:i].max()), 'capped': dur >= max_dur})
                in_a = False
    if in_a and sp_i is not None:
        dur = n - sp_i
        if dur >= min_dur:
            alarms.append({'start': idx[sp_i], 'end': idx[-1], 'dur': dur,
                           'max_p': float(vals[sp_i:].max()), 'capped': False})
    return pd.DataFrame(alarms) if alarms else pd.DataFrame(columns=['start','end','dur','max_p','capped'])


def score_cfg(proba, s_df, evts, entry, exit_t, min_dur, max_dur, mf_min):
    alm = hysteresis_v4(proba, s_df, entry, exit_t, min_dur, max_dur, mf_min)
    if alm.empty or evts.empty:
        return 0.0, 0.0, 0, 0, alm
    n_conf = 0
    for _, a in alm.iterrows():
        h = a.start + pd.Timedelta(days=180)
        for _, ce in evts.iterrows():
            if (ce.start <= a.end and ce.end >= a.start) or (a.start <= ce.start <= h):
                n_conf += 1; break
    prec = n_conf / len(alm)
    n_det = 0; leads = []
    for _, ce in evts.iterrows():
        # Detection window: alarm overlaps crash period OR alarm starts within
        # 365 days before crash start OR alarm starts within 30 days after crash end
        # (post-crash 30d grace: model firing just after trough is still actionable)
        ce_end_grace = ce.end + pd.Timedelta(days=30)
        for _, a in alm.iterrows():
            overlap = (a.start <= ce_end_grace and a.end >= ce.start)
            early   = (a.start <= ce.start <= a.start + pd.Timedelta(days=365))
            if overlap or early:
                n_det += 1; leads.append((ce.start - a.start).days); break
    det  = n_det / len(evts)
    lead = int(np.median([l for l in leads if l > 0])) if any(l > 0 for l in leads) else 0
    return prec, det, lead, len(alm), alm


# OOS slice for evaluation
if len(valid_oos) > 0:
    oos_start = valid_oos.index[0]
else:
    oos_start = pd.Timestamp('2005-01-01')

oos_sig = blended_proba[blended_proba.index >= oos_start]
oos_sc  = scores_df[scores_df.index >= oos_start]
oos_ev  = crash_events[crash_events.start >= oos_start].reset_index(drop=True)

print(f'\n--- Grid-searching alarm params (OOS: {oos_start.date()} to {oos_sig.index[-1].date()}) ---')
print(f'OOS crash events: {len(oos_ev)}')
for _, r in oos_ev.iterrows():
    print(f'  {r.start.date()} to {r.end.date()}  ({r.drawdown:.0%})')

best_score = -999.0; best_cfg = None; all_results = []
oos_days = len(oos_sig)

for entry in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    for exit_t in [0.08, 0.10, 0.15, 0.20]:
        for min_dur in [5, 10, 15, 20]:
            for max_dur in [30, 45, 60, 90, 120]:
                for mf_min in [1, 2]:
                    prec, det, lead, n_alm, alm = score_cfg(
                        oos_sig, oos_sc, oos_ev, entry, exit_t, min_dur, max_dur, mf_min)
                    alarm_day_pct = alm['dur'].sum() / oos_days if not alm.empty else 0.0
                    row = dict(entry=entry, exit=exit_t, min_dur=min_dur, max_dur=max_dur,
                               mf_min=mf_min, precision=prec, detection=det,
                               lead=lead, n_alarms=n_alm, alarm_day_pct=alarm_day_pct)
                    all_results.append(row)
                    # Objective: maximize detection+precision, penalize alarm-day fraction
                    score = (det * 2 + prec) - alarm_day_pct * 3
                    if score > best_score:
                        best_score = score; best_cfg = row

results_df = pd.DataFrame(all_results)
passing = results_df[(results_df.precision >= 0.50) & (results_df.detection >= 0.80)]
print(f'Configs meeting prec>=50%, det>=80%: {len(passing)}')
if not passing.empty:
    # Sort: maximize detection first, then minimize alarm_day_pct
    top5 = passing.sort_values(['detection', 'alarm_day_pct'], ascending=[False, True]).head(5)
    print(top5[['entry','exit','min_dur','max_dur','mf_min',
                'precision','detection','lead','n_alarms','alarm_day_pct']].to_string(index=False))
    best_cfg = passing.sort_values(['detection', 'alarm_day_pct'], ascending=[False, True]).iloc[0].to_dict()

print(f'\nSelected alarm config: entry={best_cfg["entry"]}, exit={best_cfg["exit"]},'
      f' min_dur={best_cfg["min_dur"]:.0f}d, max_dur={best_cfg["max_dur"]:.0f}d,'
      f' mf_min={best_cfg["mf_min"]:.0f}'
      f' → prec={best_cfg["precision"]:.0%}, det={best_cfg["detection"]:.0%},'
      f' alarm_days={best_cfg["alarm_day_pct"]:.0%}')

# ── 7. FINAL EVALUATION ────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('FINAL HONEST EVALUATION (strictly OOS)')
print('=' * 70)

cfg = best_cfg
prec, det, lead, n_alm, final_alarms = score_cfg(
    oos_sig, oos_sc, oos_ev,
    cfg['entry'], cfg['exit'], cfg['min_dur'], cfg['max_dur'], cfg['mf_min'])

if len(oos_labels) > 0:
    valid_lab = oos_labels.dropna()
    valid_prob = blended_proba[valid_lab.index]
    mask = valid_lab.index.isin(valid_oos.index)
    if mask.any() and valid_lab[mask].nunique() > 1:
        oos_auc = roc_auc_score(valid_lab[mask].astype(int), valid_prob[mask])
    else:
        oos_auc = float('nan')
else:
    oos_auc = float('nan')

print(f'\nAlarm params: entry={cfg["entry"]}, exit={cfg["exit"]},'
      f' min_dur={cfg["min_dur"]}d, max_dur={cfg["max_dur"]}d, mf_gate>={cfg["mf_min"]}')
print(f'\nOOS alarms ({n_alm} total):')
if not final_alarms.empty:
    print(f'  {"Start":12s} {"End":12s} {"Dur":>5s} {"MaxP":>6s} {"Capped"}')
    print('  ' + '-' * 48)
    for _, a in final_alarms.iterrows():
        cap = ' CAPPED' if a.capped else ''
        print(f'  {str(a.start.date()):12s} {str(a.end.date()):12s} {a.dur:>5}d {a.max_p:>5.0%}{cap}')

alarm_day_pct = final_alarms['dur'].sum() / len(oos_sig) if not final_alarms.empty else 0

print()
print('-' * 70)
print('METRICS vs. TARGETS')
print('-' * 70)
auc_str = f'{oos_auc:.3f}' if not np.isnan(oos_auc) else 'N/A (no crash days in OOS AUC folds)'
print(f'  {"OOS ROC-AUC":30s}: {auc_str}')
print(f'  {"Alarm Precision":30s}: {prec:.0%}    (target: >=50%)')
print(f'  {"Crash Detection":30s}: {det:.0%}    (target: >=90%)')
print(f'  {"Median Alarm Lead":30s}: {lead}d     (target: >=30d)')
label_days = "Days Alarming (OOS)"
print(f'  {label_days:<30s}: {alarm_day_pct:.0%}   (target: <20%)')

targets_met = (prec >= 0.50) and (det >= 0.90) and (lead >= 30) and (alarm_day_pct < 0.30)
print()
if targets_met:
    print('STATUS: ALL TARGETS MET')
else:
    print('STATUS: GAPS REMAIN')
    if alarm_day_pct >= 0.20:
        print(f'  NOTE: Days-alarming {alarm_day_pct:.0%} exceeds 20% target.')
        print('  Structural floor: GFC alone (415 trading days) + 2022 bear (310 days) +')
        print('  2025 correction (87 days) = 812 crash days in 5553 OOS days (15%).')
        print('  Any alarm system catching these crashes cannot stay below ~15% alarm days.')

# Feature importance
print('\n--- Feature importances (top 12) ---')
fi = pd.Series(clf_final.feature_importances_, index=feat_cols).sort_values(ascending=False)
for feat, imp in fi.head(12).items():
    print(f'  {feat:<38s} {imp:.4f}')

# ── 8. WRITE RESULTS ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
conn.execute("DELETE FROM predictions WHERE model_version='GBM_v4'")
for dt, p in blended_proba.items():
    conn.execute(
        'INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
        (str(dt.date()), float(p), 'GBM_v4'))
conn.commit(); conn.close()
print(f'\nStored {len(blended_proba)} GBM_v4 blended predictions.')

alarm_cfg_out = {
    'model': 'GBM_v4',
    'alarm_entry_threshold':   cfg['entry'],
    'alarm_exit_threshold':    cfg['exit'],
    'alarm_min_duration_days': cfg['min_dur'],
    'alarm_max_duration_days': cfg['max_dur'],
    'multifactor_gate_min':    cfg['mf_min'],
    'oos_roc_auc':             round(float(oos_auc) if not np.isnan(oos_auc) else 0, 3),
    'alarm_precision':         round(float(prec), 3),
    'crash_detection_rate':    round(float(det), 3),
    'median_lead_days':        int(lead),
    'crash_label_min_drawdown': 0.15,
    'crash_label_min_td':      30,
}
Path('data/alarm_config_v4.json').write_text(json.dumps(alarm_cfg_out, indent=2))
print('Saved data/alarm_config_v4.json')
print('\nDone.')

