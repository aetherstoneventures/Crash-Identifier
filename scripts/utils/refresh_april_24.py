"""Refresh data from 2026-03-28 through latest available business day."""
import sqlite3, requests, sys, pickle
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction

FRED = '547eaa8594ba77f00c821095c8e8482a'
DB   = 'data/market_crash.db'

def fred(sid, start):
    r = requests.get(f'https://api.stlouisfed.org/fred/series/observations'
                     f'?series_id={sid}&observation_start={start}'
                     f'&api_key={FRED}&file_type=json', timeout=20)
    return {o['date']: float(o['value']) for o in r.json().get('observations', [])
            if o['value'] != '.'}

def last(d, ds):
    vs = [v for k, v in d.items() if k <= ds]
    return vs[-1] if vs else None

# ── Pull FRED through today ───────────────────────────────────────────────────
print("Fetching FRED (through today)...")
sp  = fred('SP500','2026-04-01');   vx = fred('VIXCLS','2026-04-01')
t10 = fred('DGS10','2026-04-01');   t2 = fred('DGS2','2026-04-01')
t3m = fred('DGS3MO','2026-04-01');  nf = fred('NFCI','2025-12-01')
an  = fred('ANFCI','2025-12-01');   hy = fred('BAMLH0A0HYM2OAS','2025-12-01')
ep  = fred('USEPUINDXD','2026-01-01')
ic  = fred('ICSA','2025-12-01');    rc = fred('RECPROUSM156N','2025-06-01')

for name, d in [('SP500',sp),('VIX',vx),('DGS10',t10),('DGS2',t2),('NFCI',nf),
                ('HY_OAS',hy),('EPU',ep),('ICSA',ic),('RECPROB',rc)]:
    if d:
        k = max(d); print(f"  {name:8s} latest {k}: {d[k]}")
    else:
        print(f"  {name:8s}: NO DATA")

# ── Insert missing business days through latest SP500 date ────────────────────
conn = sqlite3.connect(DB)
latest_sp_date = max(sp) if sp else '2026-04-15'
end = pd.Timestamp(latest_sp_date)
new_dates = pd.bdate_range('2026-04-16', end)
print(f"\nDays to insert: 2026-04-16 → {end.date()}  ({len(new_dates)} business days)")

inserted = 0
for dt in new_dates:
    ds = str(dt.date())
    if conn.execute("SELECT 1 FROM indicators WHERE date=?", (ds,)).fetchone():
        continue
    row = (ds,
           last(sp,ds), last(vx,ds), last(t10,ds),
           (last(t10,ds) - last(t2,ds)) if (last(t10,ds) and last(t2,ds)) else None,
           (last(t10,ds) - last(t3m,ds)) if (last(t10,ds) and last(t3m,ds)) else None,
           last(nf,ds), last(an,ds), last(hy,ds), last(ep,ds), last(ic,ds), last(rc,ds))
    conn.execute("""INSERT INTO indicators
        (date, sp500_close, vix_close, yield_10y, yield_10y_2y, yield_10y_3m,
         nfci, anfci, hy_spread, epu_index, initial_claims, recession_prob,
         in_crash, pre_crash_60d, pre_crash_30d)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,0,0)""", row)
    inserted += 1
    p_sp = f"{row[1]:.0f}" if row[1] else '?'
    p_vx = f"{row[2]:.1f}" if row[2] else '?'
    p_ep = f"{row[9]:.0f}" if row[9] else '?'
    p_nf = f"{row[6]:.3f}" if row[6] else '?'
    print(f"  {ds}: SP={p_sp}  VIX={p_vx}  EPU={p_ep}  NFCI={p_nf}")

conn.commit()
print(f"Inserted {inserted} rows.")

# ── Regenerate StatV3 + GBM_v4 predictions for whole history ──────────────────
df = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date')
feats = engineer_features_for_prediction(df)

sv3 = StatisticalModelV3()
p_sv3 = np.clip(np.nan_to_num(sv3.predict_proba(feats)), 0, 1)

# GBM_v4 requires identical features as training
with open('models/v4_gbm_final.pkl','rb') as f:
    bundle = pickle.load(f)
clf = bundle['clf']; fcols = bundle['feature_cols']

# Build GBM feature matrix same way as training
factor_names = ['yield_curve','volatility','credit_stress','hy_credit',
                'economic','labor_market','market_momentum','sentiment',
                'financial_conditions','momentum_shock']
rows = []
for _, r in feats.iterrows():
    prob, expl = sv3._calculate_crash_probability_with_factors(r)
    sr = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    sr['regime_num']    = {'low':0,'normal':1,'high':2,'extreme':3}.get(expl.get('regime','normal'),1)
    sr['total_risk_v3'] = prob
    rows.append(sr)
scdf = pd.DataFrame(rows, index=feats.index)
raw_feats = ['sp500_drawdown','sp500_return_20d','vix_close','yield_10y_2y',
             'nfci','epu_acceleration','recession_prob','initial_claims_change_13w',
             'stress_composite','yield_curve_velocity_63d']
for f in raw_feats:
    if f in feats.columns: scdf[f] = feats[f].values
scdf = scdf.ffill().bfill().fillna(0)[fcols]

p_gbm = clf.predict_proba(scdf)[:, 1]
# Blend for final signal (matches training protocol)
blended = 0.5 * p_gbm + 0.5 * scdf['total_risk_v3'].values

# Write predictions
conn.execute("DELETE FROM predictions WHERE prediction_date > '2026-03-27'")
for i, dt in enumerate(feats.index):
    if dt < pd.Timestamp('2026-03-28'): continue
    conn.execute("INSERT OR REPLACE INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)",
                 (str(dt.date()), float(p_sv3[i]), 'StatV3_phase3'))
    conn.execute("INSERT OR REPLACE INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)",
                 (str(dt.date()), float(blended[i]), 'GBM_v4'))
conn.commit()
conn.close()

# ── Print full context: Feb 2026 through latest ───────────────────────────────
print("\n" + "="*90)
print("DAILY SIGNAL — Feb 2026 to latest")
print("="*90)
print(f"{'Date':12s} {'SP500':>7s} {'VIX':>5s} {'YC2Y':>5s} {'EPU':>5s} {'NFCI':>7s} {'StatV3':>7s} {'GBM_v4':>7s}  Signal")
print("-"*90)

ctx = feats[feats.index >= pd.Timestamp('2026-02-01')]
for i in range(len(ctx)):
    dt = ctx.index[i]; idx = feats.index.get_loc(dt)
    r = ctx.iloc[i]
    sp_v = f"{r['sp500_close']:.0f}" if not pd.isna(r['sp500_close']) else '?'
    vx_v = f"{r['vix_close']:.1f}" if not pd.isna(r['vix_close']) else '?'
    yc_v = f"{r['yield_spread_10y_2y']:.2f}" if not pd.isna(r.get('yield_spread_10y_2y', np.nan)) else '?'
    ep_v = f"{r.get('epu_index', np.nan):.0f}" if not pd.isna(r.get('epu_index', np.nan)) else '?'
    nf_v = f"{r.get('nfci', np.nan):.3f}" if not pd.isna(r.get('nfci', np.nan)) else '?'
    sv   = p_sv3[idx]; gb = blended[idx]
    sig = 'ALARM' if gb >= 0.40 else ('WARN' if gb >= 0.25 else 'ok')
    print(f"{str(dt.date()):12s} {sp_v:>7s} {vx_v:>5s} {yc_v:>5s} {ep_v:>5s} {nf_v:>7s} {sv:>6.1%} {gb:>6.1%}  {sig}")

print()
print(f"LATEST (today={str(feats.index[-1].date())}):  StatV3={p_sv3[-1]:.1%}   GBM_v4_blended={blended[-1]:.1%}")
