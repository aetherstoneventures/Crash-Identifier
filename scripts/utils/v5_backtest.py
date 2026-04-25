"""V5 regime-switching strategy backtest.

Strategy:
  Default = LONG NASDAQ
  ENTER CASH when v5_blended >= ENTRY (an alarm fires)
  RETURN to LONG when ALL of:
     - v5_blended < EXIT for at least N_CONFIRM consecutive days
     - NASDAQ closes above its 50-day moving average
     - NASDAQ has rallied >=5% off its post-alarm local low

Compares vs buy-and-hold NASDAQ. Honest accounting:
  - Daily marked-to-market
  - 5 bps slippage per switch (round-trip ~10 bps)
  - Cash earns 0% (conservative — could use risk-free rate)
  - Strictly applied to OOS predictions (model never saw those days during training)
"""
import sqlite3, json, sys
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from pathlib import Path

DB = 'data/market_crash.db'
cfg = json.loads(open('data/alarm_config_v5.json').read())
ENTRY   = cfg['entry']
EXIT    = cfg['exit']
N_CONFIRM = 10           # signal must stay below EXIT for 10 days
SLIPPAGE  = 0.0005       # 5 bps per switch
MA_PERIOD = 50
RECOVERY_PCT = 0.05      # 5% rally from post-alarm low

conn = sqlite3.connect(DB)
preds = pd.read_sql("SELECT prediction_date, crash_probability FROM predictions WHERE model_version='v5' ORDER BY prediction_date",
                    conn, parse_dates=['prediction_date']).set_index('prediction_date')['crash_probability']
ind = pd.read_sql('SELECT date, nasdaq_close FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date')
conn.close()

# Align
df = pd.DataFrame({'price': ind['nasdaq_close'], 'p': preds})
df = df.dropna()
df['ret'] = df['price'].pct_change().fillna(0)
df['ma50'] = df['price'].rolling(MA_PERIOD, min_periods=20).mean()

# Build position series
in_alarm = False
post_alarm_low = None
days_below_exit = 0
position = []
state = []
for dt, row in df.iterrows():
    p = row['p']; px = row['price']; ma = row['ma50']
    if not in_alarm:
        if p >= ENTRY:
            in_alarm = True
            post_alarm_low = px
            days_below_exit = 0
            position.append(0); state.append('CASH(enter)')
            continue
        position.append(1); state.append('LONG')
    else:
        post_alarm_low = min(post_alarm_low, px)
        if p < EXIT:
            days_below_exit += 1
        else:
            days_below_exit = 0
        recovered = (px / post_alarm_low - 1) >= RECOVERY_PCT
        above_ma = (not pd.isna(ma)) and (px > ma)
        if days_below_exit >= N_CONFIRM and recovered and above_ma:
            in_alarm = False
            position.append(1); state.append('LONG(reenter)')
        else:
            position.append(0); state.append('CASH')
df['pos'] = position
df['state'] = state

# Trade dates
switches = df['pos'].diff().fillna(0).abs()
n_switches = int(switches.sum())
df['cost'] = switches * SLIPPAGE

# Strategy returns
df['strat_ret'] = df['pos'].shift(1).fillna(1) * df['ret'] - df['cost']
df['strat_eq']  = (1 + df['strat_ret']).cumprod()
df['bh_eq']     = (1 + df['ret']).cumprod()

# Metrics
def metrics(returns, label):
    eq = (1 + returns).cumprod()
    n_yrs = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1/n_yrs) - 1
    vol  = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 1e-12)
    dd = eq / eq.cummax() - 1
    max_dd = dd.min()
    return dict(label=label, n_yrs=n_yrs, cagr=cagr, vol=vol, sharpe=sharpe,
                max_dd=max_dd, final_eq=eq.iloc[-1])

# Headline period: full history & blind window
def report(slc, title):
    sub = df.loc[slc]
    if len(sub) < 30: return
    bh = metrics(sub['ret'], 'Buy-and-hold NASDAQ')
    st = metrics(sub['strat_ret'], 'V5 regime-switch')
    print(f'\n--- {title} ({sub.index[0].date()} → {sub.index[-1].date()}, {len(sub)} days) ---')
    print(f'{"":24s} {"CAGR":>8s} {"Vol":>7s} {"Sharpe":>7s} {"MaxDD":>8s} {"x":>7s}')
    for m in (bh, st):
        print(f'{m["label"]:24s} {m["cagr"]:>7.1%} {m["vol"]:>6.1%} {m["sharpe"]:>7.2f} {m["max_dd"]:>7.1%} {m["final_eq"]:>6.2f}x')
    print(f'  Switches in window: {int(sub["pos"].diff().abs().sum())}')
    days_long = (sub['pos']==1).mean()
    print(f'  Time long: {days_long:.0%}   Time cash: {1-days_long:.0%}')

print('='*78)
print('V5 REGIME-SWITCH BACKTEST vs BUY-AND-HOLD NASDAQ')
print(f'  Rules: enter cash @ p≥{ENTRY:.0%}, re-enter when p<{EXIT:.0%} for {N_CONFIRM}d AND price>MA{MA_PERIOD} AND +{RECOVERY_PCT:.0%} off low')
print(f'  Costs: {SLIPPAGE*1e4:.0f} bps slippage per switch')
print('='*78)

# Total OOS history (since 1999 — fold 1 onward)
oos_start = pd.Timestamp('2000-01-01')
report(slice(oos_start, None), 'FULL OOS PERIOD')
report(slice(pd.Timestamp('2021-01-01'), None), 'BLIND TEST WINDOW (alarm params unseen)')
report(slice(pd.Timestamp('2000-01-01'), pd.Timestamp('2010-12-31')), 'Dotcom + GFC era')
report(slice(pd.Timestamp('2011-01-01'), pd.Timestamp('2019-12-31')), 'Mid-2010s bull')

# Show all switches in blind window
print('\n--- Switch log (2021+) ---')
sw = df.loc['2021-01-01':, ['price','p','pos','state']]
sw = sw[sw['state'].str.contains('enter|reenter')]
print(sw.to_string())

# Save full equity curve to csv for plotting
out = df[['price','p','pos','strat_eq','bh_eq']].copy()
out.to_csv('data/v5_backtest.csv')
print('\nSaved data/v5_backtest.csv')
