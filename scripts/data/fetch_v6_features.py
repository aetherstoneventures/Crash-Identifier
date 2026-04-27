"""
v6 data fetch — adds tail-risk + breadth + cross-asset features.

Sources:
  CBOE SKEW History.csv   (direct, 1990+ daily) — options-implied tail risk
  Yahoo Finance:
    ^VIX9D                 (2011+ daily)         — 9-day VIX (short tenor)
    SPY,QQQ,IWM,RSP        (breadth proxies)
    XLK,XLF,XLU,XLP,XLV,XLY,XLE,XLI  (sector ETFs)
    TLT,GLD,HYG,LQD        (cross-asset)

All data persists to indicators table under explicit column names with v6_ prefix
to avoid colliding with anything v5 reads.
"""
import sqlite3, time, requests
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

DB = 'data/market_crash.db'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

YAHOO_CHART = 'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=315532800&period2=2000000000&interval=1d'

def fetch_yahoo(sym):
    for attempt in range(3):
        try:
            r = requests.get(YAHOO_CHART.format(sym=sym), timeout=20, headers=HEADERS)
            if r.status_code != 200: time.sleep(2); continue
            j = r.json(); res = j['chart']['result'][0]
            ts = res['timestamp']
            close = res['indicators']['quote'][0]['close']
            df = pd.DataFrame({'date': pd.to_datetime(ts, unit='s').normalize(), 'close': close})
            df = df.dropna().drop_duplicates('date').set_index('date')
            return df['close'].astype(float)
        except Exception as e:
            print(f'    {sym} retry {attempt}: {e}'); time.sleep(2)
    raise RuntimeError(sym)

def fetch_cboe_skew():
    url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv'
    r = requests.get(url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE').sort_index()
    return df['SKEW'].astype(float)

# Define columns we'll add
SYMS = {
    'v6_spy':  'SPY',  'v6_qqq': 'QQQ',  'v6_iwm':  'IWM',  'v6_rsp':  'RSP',
    'v6_xlk':  'XLK',  'v6_xlf': 'XLF',  'v6_xlu':  'XLU',  'v6_xlp':  'XLP',
    'v6_xlv':  'XLV',  'v6_xly': 'XLY',  'v6_xle':  'XLE',  'v6_xli':  'XLI',
    'v6_tlt':  'TLT',  'v6_gld': 'GLD',  'v6_hyg':  'HYG',  'v6_lqd':  'LQD',
    'v6_vix9d':'^VIX9D',
}
SKEW_COL = 'v6_skew'

# 1) Add columns idempotently
conn = sqlite3.connect(DB)
existing = [r[1] for r in conn.execute('PRAGMA table_info(indicators)').fetchall()]
all_cols = list(SYMS.keys()) + [SKEW_COL]
for c in all_cols:
    if c not in existing:
        conn.execute(f'ALTER TABLE indicators ADD COLUMN {c} REAL')
        print(f'  added column: {c}')
conn.commit()

dates = pd.read_sql('SELECT date FROM indicators ORDER BY date', conn, parse_dates=['date'])['date']
all_dates = pd.DatetimeIndex(sorted(dates.unique()))
frame = pd.DataFrame(index=all_dates)

# 2) CBOE SKEW
print('Fetching CBOE SKEW ...')
skew = fetch_cboe_skew()
print(f'  {len(skew)} obs   {skew.index.min().date()} -> {skew.index.max().date()}')
frame[SKEW_COL] = skew.reindex(all_dates)

# 3) Yahoo symbols
for col, sym in SYMS.items():
    print(f'Fetching {sym} -> {col} ...')
    try:
        s = fetch_yahoo(sym)
        print(f'  {len(s)} obs   {s.index.min().date()} -> {s.index.max().date()}')
        frame[col] = s.reindex(all_dates)
    except Exception as e:
        print(f'  FAIL {sym}: {e}')
        frame[col] = np.nan

# 4) Upsert
print('\nUpserting...')
n = 0
for dt, row in frame.iterrows():
    vals = tuple(None if pd.isna(v) else float(v) for v in row[all_cols])
    conn.execute(
        f'UPDATE indicators SET {", ".join(c+"=?" for c in all_cols)} WHERE date=?',
        vals + (dt.strftime('%Y-%m-%d'),))
    n += 1
conn.commit()
print(f'Updated {n} rows')

print('\nNon-null counts:')
for c in all_cols:
    nn = conn.execute(f'SELECT COUNT({c}) FROM indicators').fetchone()[0]
    first = conn.execute(f'SELECT MIN(date) FROM indicators WHERE {c} IS NOT NULL').fetchone()[0]
    print(f'  {c:14s} {nn:>6d}  from {first}')
conn.close()
print('Done.')
