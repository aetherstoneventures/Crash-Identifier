"""Daily refresh pipeline for V5.

Steps:
  1. Pull fresh FRED data through today (core series + v5 add-ons)
  2. Purge bad rows (data sanity)
  3. Regenerate v5 predictions using the persisted model bundle

Idempotent — safe to run multiple times per day.
"""
import subprocess, sys, sqlite3, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from pathlib import Path

from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction

DB = 'data/market_crash.db'

def run(cmd):
    print(f'\n$ {cmd}')
    subprocess.run(cmd, shell=True, check=True)

# 1. Fresh FRED data
run('venv/bin/python3 -W ignore scripts/utils/refresh_april_24.py | tail -5')
# 2. Fetch v5 add-on features (idempotent upsert of nasdaq_close / baa / oil / dollar / epu_daily)
run('venv/bin/python3 -W ignore scripts/data/fetch_v5_features.py | tail -10')
# 3. Data sanity guard
run('venv/bin/python3 -W ignore scripts/data/purge_bad_rows.py | tail -10')

# 4. Regenerate v5 predictions using persisted model
print('\n--- Regenerating v5 predictions from persisted model ---')
bundle = pickle.load(open('models/v5/v5_final.pkl','rb'))
clf = bundle['clf']; feat_cols = bundle['feature_cols']; factor_names = bundle['factor_names']

conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
eq = ind['nasdaq_close'].copy()
ind = ind.loc[eq.dropna().index]
eq  = ind['nasdaq_close'].ffill()
feats = engineer_features_for_prediction(ind)
rm = eq.rolling(252, min_periods=50).max()
feats['equity_drawdown']   = eq / rm - 1
feats['equity_return_20d'] = eq.pct_change(20)
feats['equity_return_63d'] = eq.pct_change(63)
feats['equity_vol_20d']    = eq.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
feats['equity_vol_63d']    = eq.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
for col in ['baa_10y_spread','oil_wti','dollar_twi','epu_daily']:
    if col in ind.columns:
        feats[col] = ind[col].ffill()
feats['baa_chg_20d']   = feats.get('baa_10y_spread', pd.Series(0, index=feats.index)).diff(20)
feats['oil_return_20d']= feats.get('oil_wti', pd.Series(np.nan, index=feats.index)).pct_change(20)
feats['dollar_chg_20d']= feats.get('dollar_twi', pd.Series(np.nan, index=feats.index)).pct_change(20)
feats['epu_chg_20d']   = feats.get('epu_daily', pd.Series(np.nan, index=feats.index)).diff(20)

sv3 = StatisticalModelV3()
rows=[]
for _, row in feats.iterrows():
    p, expl = sv3._calculate_crash_probability_with_factors(row)
    r = {f: expl.get(f'{f}_score', 0.0) for f in factor_names}
    r['regime_num']    = {'low':0,'normal':1,'high':2,'extreme':3}.get(expl.get('regime','normal'), 1)
    r['total_risk_v3'] = p
    rows.append(r)
sc = pd.DataFrame(rows, index=feats.index)
for f in feat_cols:
    if f not in sc.columns and f in feats.columns:
        sc[f] = feats[f].values
sc = sc[feat_cols].ffill().bfill().fillna(0)

proba = pd.Series(clf.predict_proba(sc)[:, 1], index=sc.index)
blended = 0.5 * proba + 0.5 * sc['total_risk_v3']

conn.execute("DELETE FROM predictions WHERE model_version='v5'")
for dt, p in blended.items():
    conn.execute('INSERT INTO predictions (prediction_date, crash_probability, model_version) VALUES (?,?,?)',
                 (str(dt.date()), float(p), 'v5'))
conn.commit()

latest = blended.dropna().tail(1)
if len(latest):
    dt = latest.index[0]; p = float(latest.iloc[0])
    print(f'\nLatest v5 blended signal: {dt.date()} = {p:.1%}  {"ALARM" if p>=0.45 else "ok"}')
conn.close()
print('Refresh complete.')
