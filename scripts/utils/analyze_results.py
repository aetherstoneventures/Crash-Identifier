"""Analyze pipeline results from database for deep audit."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import sqlalchemy as sa
from src.utils.database import DatabaseManager, Indicator, CrashEvent, Prediction

db = DatabaseManager()

print("=" * 80)
print("DATABASE CONTENT ANALYSIS")
print("=" * 80)

with db.get_session() as session:
    row_count = session.query(Indicator).count()
    first = session.query(Indicator).order_by(Indicator.date.asc()).first()
    last  = session.query(Indicator).order_by(Indicator.date.desc()).first()
    crash_count = session.query(CrashEvent).count()
    pred_count  = session.query(Prediction).count()

    # Crash events
    crashes = session.query(CrashEvent).order_by(CrashEvent.start_date.asc()).all()
    crash_data = [(str(c.start_date), str(c.trough_date), str(c.end_date),
                   c.max_drawdown, c.crash_type) for c in crashes]

    # Prediction stats
    max_prob  = session.query(sa.func.max(Prediction.crash_probability)).scalar() or 0
    mean_prob = session.query(sa.func.avg(Prediction.crash_probability)).scalar() or 0

    # All predictions as DataFrame
    preds_raw = session.query(Prediction).order_by(Prediction.prediction_date.asc()).all()
    pred_rows = [(str(p.prediction_date), p.crash_probability) for p in preds_raw]

    # Key indicators — most recent
    recent_inds = session.query(Indicator).order_by(Indicator.date.desc()).limit(30).all()
    ind_rows = [(str(i.date), i.sp500_close, i.vix_close, i.yield_10y_3m,
                 i.yield_10y_2y, i.credit_spread_bbb, i.unemployment_rate,
                 i.fed_funds_rate, i.consumer_sentiment, i.lei) for i in recent_inds]

    session.expunge_all()

print(f"Indicator rows : {row_count}")
print(f"Date range     : {first.date} to {last.date}")
print(f"Crash events   : {crash_count}")
print(f"Predictions    : {pred_count}")
print(f"Max probability: {max_prob:.4f}")
print(f"Mean probability:{mean_prob:.4f}")

print("\n=== CRASH EVENTS DETECTED ===")
print(f"{'Start':12}  {'Trough':12}  {'End':12}  {'MaxDD':>7}  Type")
print("-" * 70)
for c in crash_data:
    print(f"{c[0]:12}  {c[1]:12}  {c[2]:12}  {c[3]:>6.1f}%  {c[4]}")

# Predictions DataFrame
pred_df = pd.DataFrame(pred_rows, columns=['date', 'prob'])
pred_df['date'] = pd.to_datetime(pred_df['date'])
pred_df = pred_df.set_index('date').sort_index()

print("\n=== HIGH-RISK PERIODS (prob > 40%) ===")
high = pred_df[pred_df['prob'] > 0.40].copy()
# Group into contiguous episodes
if len(high) > 0:
    high['gap'] = (high.index.to_series().diff() > pd.Timedelta('5D'))
    high['episode'] = high['gap'].cumsum()
    for ep, grp in high.groupby('episode'):
        print(f"  {grp.index[0].date()} -> {grp.index[-1].date()}  "
              f"peak={grp['prob'].max():.1%}  mean={grp['prob'].mean():.1%}  days={len(grp)}")
else:
    print("  None above 40%")

print("\n=== MODERATE RISK PERIODS (prob 25-40%) ===")
mod = pred_df[(pred_df['prob'] >= 0.25) & (pred_df['prob'] < 0.40)].copy()
if len(mod) > 0:
    mod['gap'] = (mod.index.to_series().diff() > pd.Timedelta('5D'))
    mod['episode'] = mod['gap'].cumsum()
    for ep, grp in mod.groupby('episode'):
        print(f"  {grp.index[0].date()} -> {grp.index[-1].date()}  "
              f"peak={grp['prob'].max():.1%}  mean={grp['prob'].mean():.1%}  days={len(grp)}")
else:
    print("  None in 25-40% range")

print("\n=== MOST RECENT 30 DAYS OF PREDICTIONS ===")
recent_preds = pred_df.tail(30)
for dt, row in recent_preds.iterrows():
    bar = '#' * int(row['prob'] * 40)
    print(f"  {dt.date()}  {row['prob']:6.1%}  {bar}")

print("\n=== MOST RECENT INDICATORS ===")
print(f"{'Date':12}  {'SP500':>8}  {'VIX':>6}  {'10Y-3M':>7}  {'10Y-2Y':>7}  {'CrSpr':>6}  {'Unemp':>6}  {'FF':>5}  {'Sent':>5}  {'LEI':>6}")
print("-" * 100)
for r in ind_rows[:20]:
    print(f"{r[0]:12}  {r[1] or 0:>8.1f}  {r[2] or 0:>6.1f}  "
          f"{r[3] or 0:>7.2f}  {r[4] or 0:>7.2f}  {r[5] or 0:>6.2f}  "
          f"{r[6] or 0:>6.2f}  {r[7] or 0:>5.2f}  {r[8] or 0:>5.1f}  {r[9] or 0:>6.2f}")

# Year-by-year max probability
print("\n=== YEAR-BY-YEAR MAX CRASH PROBABILITY ===")
pred_df['year'] = pred_df.index.year
yearly = pred_df.groupby('year')['prob'].agg(['max', 'mean', 'std']).round(4)
for yr, row in yearly.iterrows():
    bar = '#' * int(row['max'] * 50)
    print(f"  {yr}  max={row['max']:.1%}  mean={row['mean']:.1%}  {bar}")
