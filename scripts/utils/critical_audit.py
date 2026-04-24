"""CRITICAL AUDIT — stress-test v4 pipeline claims."""
import sqlite3, sys, pickle
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction

DB = 'data/market_crash.db'
conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date')

# =============================================================================
# AUDIT 1 — DATA QUALITY: hunt for bad rows
# =============================================================================
print("="*80)
print("AUDIT 1 — DATA QUALITY CHECKS")
print("="*80)

# Find SP500 day-over-day moves > 10% (highly suspicious outside crashes)
sp = ind['sp500_close'].dropna()
ret = sp.pct_change()
big = ret[ret.abs() > 0.10].sort_values()
print(f"\nDaily SP500 moves > 10% ({len(big)} days):")
for dt, r in big.head(20).items():
    prev = sp.loc[sp.index < dt].iloc[-1] if (sp.index < dt).any() else np.nan
    print(f"  {dt.date()}  move={r:+.1%}  prev_close={prev:.1f}  this_close={sp.loc[dt]:.1f}")

# Feb 16, 2026 anomaly
if pd.Timestamp('2026-02-16') in ind.index:
    r = ind.loc['2026-02-16']
    print(f"\n⚠ Feb 16, 2026 (Presidents Day): SP500={r['sp500_close']}  VIX={r['vix_close']}  EPU={r['epu_index']}")
    print(f"  → Market was CLOSED. This value (4516) is a FRED placeholder/error.")
    print(f"  → Pipeline fired 56.8% GBM_v4 alarm on this day. False alarm caused by bad data.")


# =============================================================================
# AUDIT 2 — WAS THE MARCH 2026 "DETECTION" ACTUALLY PREDICTION?
# =============================================================================
print("\n" + "="*80)
print("AUDIT 2 — MARCH 2026 CORRECTION: LEAD OR LAG?")
print("="*80)

preds = pd.read_sql("SELECT prediction_date, crash_probability FROM predictions WHERE model_version='GBM_v4' ORDER BY prediction_date",
                    conn, parse_dates=['prediction_date']).set_index('prediction_date')
window = pd.concat([ind[['sp500_close','vix_close','epu_index']].loc['2026-02-20':'2026-04-05'],
                    preds.rename(columns={'crash_probability':'gbm_v4'}).loc['2026-02-20':'2026-04-05']], axis=1)
peak = window['sp500_close'].idxmax()
trough = window['sp500_close'].idxmin()
peak_val = window['sp500_close'].loc[peak]
trough_val = window['sp500_close'].loc[trough]
drawdown = trough_val / peak_val - 1
print(f"\nMarch 2026 correction:")
print(f"  Peak:    {peak.date()}  SP500 = {peak_val:.0f}")
print(f"  Trough:  {trough.date()}  SP500 = {trough_val:.0f}")
print(f"  Drawdown: {drawdown:.1%}")
print(f"  This is NOT a 'crash' under the pipeline's own 15% threshold.")

first_alarm = window[window['gbm_v4'] >= 0.40]
if not first_alarm.empty:
    first = first_alarm.index[0]
    lead = (trough - first).days
    sp_drop_to_first = window['sp500_close'].loc[first] / peak_val - 1
    print(f"\n  First full alarm (GBM>=40%): {first.date()}")
    print(f"  By the time alarm fires, SP500 already down {sp_drop_to_first:.1%} from peak.")
    print(f"  'Lead' to trough: {lead} days — but this is LAG from peak by {(first-peak).days} days.")

# Was there an early-warning signal BEFORE the peak?
pre_peak = window.loc[:peak][['gbm_v4']].tail(20)
print(f"\n  GBM probability in 20 days BEFORE March peak:")
print(f"    min={pre_peak['gbm_v4'].min():.1%}  max={pre_peak['gbm_v4'].max():.1%}  mean={pre_peak['gbm_v4'].mean():.1%}")


# =============================================================================
# AUDIT 3 — NAIVE BASELINES (how does v4 compare to trivial rules?)
# =============================================================================
print("\n" + "="*80)
print("AUDIT 3 — NAIVE BASELINES vs v4 (OOS 2005-2026)")
print("="*80)

feats = engineer_features_for_prediction(ind)
# Rebuild rigorous crash labels
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

crash = compute_crash_labels(feats['sp500_close'])
oos_mask = crash.index >= pd.Timestamp('2005-01-01')
y = crash[oos_mask].astype(int).values

# Baseline 1: VIX > 25 → alarm
vix_high = (feats['vix_close'] > 25).astype(int)[oos_mask].values
# Baseline 2: drawdown > 5% → alarm
dd5 = (feats['sp500_drawdown'] < -0.05).astype(int)[oos_mask].values
# Baseline 3: VIX > 25 OR drawdown > 5%
combo = (vix_high | dd5)

for name, bs in [('VIX>25', vix_high), ('Drawdown>5%', dd5), ('VIX>25 OR DD>5%', combo)]:
    n_alarm_days = bs.sum()
    tp = int((bs & y).sum())
    fp = int((bs & ~y.astype(bool)).sum())
    prec = tp / max(n_alarm_days, 1)
    recall = tp / max(y.sum(), 1)
    print(f"  {name:20s}: alarm_days={n_alarm_days:>4}  prec={prec:.1%}  recall={recall:.1%}")

# GBM_v4 blended day-level precision/recall
gbm = preds[preds.index >= pd.Timestamp('2005-01-01')]['crash_probability'].reindex(feats.index[oos_mask]).values
for thresh in [0.25, 0.30, 0.40, 0.50]:
    bs = (gbm >= thresh)
    n_alarm_days = int(bs.sum())
    tp = int((bs & y.astype(bool)).sum())
    prec = tp / max(n_alarm_days, 1)
    recall = tp / max(y.sum(), 1)
    print(f"  GBM_v4 >= {thresh:.2f}   : alarm_days={n_alarm_days:>4}  prec={prec:.1%}  recall={recall:.1%}")

auc = roc_auc_score(y, gbm)
print(f"\n  GBM_v4 day-level ROC-AUC (OOS): {auc:.3f}")


# =============================================================================
# AUDIT 4 — ALARM-PARAMETER TUNING: IS IT LEAKED?
# =============================================================================
print("\n" + "="*80)
print("AUDIT 4 — ALARM PARAMETERS TUNED ON SAME WINDOW AS EVALUATED")
print("="*80)
print("""
  The XGBoost model uses walk-forward validation — ✅ honest.
  BUT the alarm parameters (entry=0.40, exit=0.20, min_dur=10d, max_dur=45d, mf_min=2)
  were GRID-SEARCHED on the full 2005-2026 OOS window, then EVALUATED on that same window.

  → 'Configs meeting prec>=50%, det>=80%: 59' — out of 960 tested configs, only 6% passed.
  → The reported 50% precision / 100% detection is a MAX over 960 combinations on the
    same data they're reported on. This is classic parameter-overfitting.

  → TRUE HONEST TEST: train alarm params on 2005-2020, evaluate on 2021-2026.""")

# Split-window re-test of alarm tuning
factor_names = ['yield_curve','volatility','credit_stress','hy_credit',
                'economic','labor_market','market_momentum','sentiment',
                'financial_conditions','momentum_shock']
sv3 = StatisticalModelV3()
rows_ = []
for _, rr in feats.iterrows():
    p, expl = sv3._calculate_crash_probability_with_factors(rr)
    sr = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    rows_.append(sr)
sc_full = pd.DataFrame(rows_, index=feats.index)
elev_counts = (sc_full[factor_names].values > 0.5).sum(axis=1)

def hyst(p, elev, enter, exit_t, min_dur, max_dur, mf_min, idx):
    alarms=[]; in_a=False; si=None; n=len(p)
    for i in range(n):
        if not in_a:
            if p[i] >= enter and elev[i] >= mf_min: in_a=True; si=i
        else:
            d = i - si
            if p[i] < exit_t or d >= max_dur:
                if d >= min_dur: alarms.append((idx[si], idx[i-1], d))
                in_a=False
    if in_a and si is not None:
        d = n - si
        if d >= min_dur: alarms.append((idx[si], idx[-1], d))
    return alarms

def score(alarms, evs, sig_idx):
    if not alarms or evs.empty: return 0,0,0
    conf=0
    for a_s, a_e, _ in alarms:
        h = a_s + pd.Timedelta(days=180)
        for _, ce in evs.iterrows():
            if (ce.start <= a_e and ce.end >= a_s) or (a_s <= ce.start <= h):
                conf += 1; break
    prec = conf/len(alarms)
    det_n=0
    for _, ce in evs.iterrows():
        for a_s, a_e, _ in alarms:
            if (a_s <= ce.end+pd.Timedelta(days=30) and a_e >= ce.start) or (a_s <= ce.start <= a_s+pd.Timedelta(days=365)):
                det_n += 1; break
    det = det_n/len(evs)
    return prec, det, sum(d for _,_,d in alarms)/len(sig_idx)

# Build events
def events(lbl):
    periods=[]; in_c=False; cs=None
    for dt, v in lbl.items():
        if not in_c and v: in_c, cs = True, dt
        elif in_c and not v: periods.append({'start':cs,'end':dt}); in_c=False
    if in_c: periods.append({'start':cs,'end':lbl.index[-1]})
    return pd.DataFrame(periods) if periods else pd.DataFrame(columns=['start','end'])

gbm_full = preds['crash_probability'].reindex(feats.index).values

# TRAIN window: 2005-2020
train_mask = (feats.index >= pd.Timestamp('2005-01-01')) & (feats.index <= pd.Timestamp('2020-12-31'))
eval_mask  = feats.index >= pd.Timestamp('2021-01-01')
evs_train = events(crash[train_mask])
evs_eval  = events(crash[eval_mask])

print(f"\n  Crashes in TRAIN window (2005-2020): {len(evs_train)}")
print(f"  Crashes in EVAL  window (2021-2026): {len(evs_eval)}")

# Grid search on TRAIN
best=None; best_sc=-1
for e in [0.25,0.30,0.35,0.40,0.45]:
    for xt in [0.10,0.15,0.20]:
        for mn in [10,15,20]:
            for mx in [30,45,60,90]:
                for mf in [1,2]:
                    al = hyst(gbm_full[train_mask], elev_counts[train_mask], e,xt,mn,mx,mf, feats.index[train_mask])
                    p,d,apc = score(al, evs_train, feats.index[train_mask])
                    sc = d*2 + p - apc*3
                    if p >= 0.50 and d >= 0.80 and sc > best_sc:
                        best_sc = sc; best = (e,xt,mn,mx,mf,p,d,apc)

if best:
    e,xt,mn,mx,mf,p_tr,d_tr,apc_tr = best
    print(f"\n  Best TRAIN config: entry={e} exit={xt} min={mn}d max={mx}d mf_min={mf}")
    print(f"    TRAIN metrics:  prec={p_tr:.1%}  det={d_tr:.1%}  alarm_days={apc_tr:.1%}")
    # Apply to EVAL
    al_eval = hyst(gbm_full[eval_mask], elev_counts[eval_mask], e,xt,mn,mx,mf, feats.index[eval_mask])
    p_ev, d_ev, apc_ev = score(al_eval, evs_eval, feats.index[eval_mask])
    print(f"    EVAL metrics:   prec={p_ev:.1%}  det={d_ev:.1%}  alarm_days={apc_ev:.1%}  (n_alarms={len(al_eval)})")
    print(f"\n  This is the HONEST out-of-window performance.")

conn.close()
print("\nAudit complete.")
