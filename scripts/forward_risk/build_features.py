"""Forward Risk — feature engineering layer.

Reads raw indicators from data/market_crash.db and produces a disciplined,
deterministic feature matrix for the multi-horizon forward-distribution model.

Design constraints (the honesty contract):
  * NASDAQ-only target asset (FRED NASDAQCOM, ~1971-now).
  * Every feature has a stated economic rationale (see FEATURE_SPEC).
  * No look-ahead — every feature at date t uses only data observable by t.
  * No fitting on data — z-scores use rolling windows, not full-sample stats.
  * Output: parquet at data/processed/forward_risk_features.parquet.

Run:
    venv/bin/python3 -W ignore scripts/forward_risk/build_features.py
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market_crash.db"
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "forward_risk_features.parquet"


# Feature catalog: (name, rationale).  Built deterministically from DB columns.
FEATURE_SPEC: list[tuple[str, str]] = [
    # --- equity (NASDAQ-derived) ---
    ("ret_1d",            "1-day NASDAQ log return"),
    ("ret_5d",            "5-day momentum (1 trading week)"),
    ("ret_21d",           "21-day momentum (1 trading month)"),
    ("ret_63d",           "63-day momentum (1 trading quarter)"),
    ("ret_252d",          "252-day momentum (1 trading year)"),
    ("vol_21d",           "21-day realized vol of daily returns (annualized)"),
    ("vol_63d",           "63-day realized vol (annualized)"),
    ("vol_ratio_21_63",   "21d / 63d vol ratio — vol-of-vol proxy"),
    ("drawdown_252d",     "Drawdown from 252-day rolling max (≤ 0)"),
    ("days_since_max",    "Days since the 252-day rolling max was set"),
    # --- VIX ---
    ("vix_level",         "VIX index level"),
    ("vix_z_60d",         "VIX z-score over 60 trading days"),
    ("vix_change_21d",    "21-day level change in VIX"),
    # --- yield curve ---
    ("y10_3m",            "10y minus 3m Treasury spread (recession proxy)"),
    ("y10_2y",            "10y minus 2y Treasury spread"),
    ("y10_3m_chg_63d",    "63d change in 10y-3m spread"),
    ("fed_funds",         "Federal Funds Rate"),
    ("ff_chg_252d",       "252d change in Fed Funds (policy direction)"),
    # --- credit ---
    ("baa_10y_spread",    "Moody's Baa - 10y Treasury (credit stress)"),
    ("baa_chg_63d",       "63d change in Baa-10y spread"),
    ("baa_z_252d",        "Baa-10y z-score over 252 days"),
    # --- macro (low frequency, ffill'd) ---
    ("unemp_chg_252d",    "252d change in unemployment rate"),
    ("ip_chg_252d",       "252d % change in industrial production"),
    ("lei_chg_252d",      "252d % change in Leading Economic Index"),
    ("cpi_chg_252d",      "252d % change in CPI (inflation rate)"),
    ("m2_chg_252d",       "252d % change in M2 money supply"),
    # --- sentiment ---
    ("sentiment_z_252d",  "Consumer sentiment z-score (252d)"),
    ("epu_level",         "Daily Economic Policy Uncertainty (Baker et al.)"),
    ("epu_ma_30d",        "30-day MA of EPU (smoothed uncertainty)"),
    # --- cross-asset ---
    ("dollar_chg_63d",    "63d % change in trade-weighted dollar"),
    ("oil_chg_63d",       "63d % change in WTI"),
    ("oil_vol_63d",       "63d realized vol of daily oil returns"),
]


def safe_log_ret(s: pd.Series, n: int) -> pd.Series:
    """log(P_t / P_{t-n}). Returns NaN for the first n-1 rows."""
    return np.log(s / s.shift(n))


def safe_pct_chg(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(periods=n, fill_method=None)


def rolling_z(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=max(20, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(20, win // 4)).std()
    return (s - mu) / sd.replace(0, np.nan)


def realized_vol(rets: pd.Series, win: int) -> pd.Series:
    """Annualized realized vol from daily log returns."""
    return rets.rolling(win, min_periods=max(10, win // 4)).std() * np.sqrt(252)


def drawdown_from_max(p: pd.Series, win: int) -> tuple[pd.Series, pd.Series]:
    rmax = p.rolling(win, min_periods=20).max()
    dd = (p / rmax) - 1.0  # ≤ 0
    # days_since_max: index distance to the argmax in the trailing window
    def _dsm(arr):
        if np.isnan(arr).all():
            return np.nan
        # argmax over last len(arr) days, returning #days back
        return float(len(arr) - 1 - int(np.nanargmax(arr)))
    dsm = p.rolling(win, min_periods=20).apply(_dsm, raw=True)
    return dd, dsm


def main() -> None:
    print("=" * 78)
    print("Forward Risk — feature engineering")
    print("=" * 78)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(DB_PATH)) as conn:
        cols = pd.read_sql("PRAGMA table_info(indicators)", conn)["name"].tolist()
        print(f"Loaded indicators schema: {len(cols)} columns")

        # Pull only what we need; coerce numeric.
        wanted = [
            "date", "nasdaq_close", "vix_close",
            "yield_10y_3m", "yield_10y_2y", "yield_10y", "fed_funds_rate",
            "credit_spread_bbb", "baa_10y_spread",
            "unemployment_rate", "industrial_production", "lei", "cpi", "m2_money_supply",
            "consumer_sentiment", "epu_daily",
            "dollar_twi", "oil_wti",
        ]
        present = [c for c in wanted if c in cols]
        missing = [c for c in wanted if c not in cols]
        if missing:
            print(f"  WARNING: missing DB columns (will be NaN): {missing}")

        df = pd.read_sql(
            f"SELECT {', '.join(present)} FROM indicators ORDER BY date",
            conn, parse_dates=["date"],
        ).set_index("date").sort_index()

    # Coerce all to numeric (legacy text/object columns).
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only rows where NASDAQ exists — that's our spine.
    if "nasdaq_close" not in df.columns or df["nasdaq_close"].isna().all():
        raise RuntimeError(
            "nasdaq_close is empty. Run scripts/data/fetch_v5_features.py first."
        )
    df = df[df["nasdaq_close"].notna()].copy()
    print(f"NASDAQ spine: {len(df):,} rows · {df.index.min().date()} → {df.index.max().date()}")

    # Forward-fill macro (monthly/quarterly) into daily grid.
    macro_cols = [c for c in [
        "unemployment_rate", "industrial_production", "lei", "cpi", "m2_money_supply",
        "consumer_sentiment", "fed_funds_rate", "credit_spread_bbb", "baa_10y_spread",
        "yield_10y_3m", "yield_10y_2y", "yield_10y", "vix_close",
        "epu_daily", "dollar_twi", "oil_wti",
    ] if c in df.columns]
    df[macro_cols] = df[macro_cols].ffill()

    # ------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------
    f = pd.DataFrame(index=df.index)
    p = df["nasdaq_close"]
    rets = np.log(p / p.shift(1))

    # Equity
    f["ret_1d"]   = rets
    f["ret_5d"]   = safe_log_ret(p, 5)
    f["ret_21d"]  = safe_log_ret(p, 21)
    f["ret_63d"]  = safe_log_ret(p, 63)
    f["ret_252d"] = safe_log_ret(p, 252)
    f["vol_21d"]  = realized_vol(rets, 21)
    f["vol_63d"]  = realized_vol(rets, 63)
    f["vol_ratio_21_63"] = f["vol_21d"] / f["vol_63d"].replace(0, np.nan)
    dd, dsm = drawdown_from_max(p, 252)
    f["drawdown_252d"]  = dd
    f["days_since_max"] = dsm

    # VIX
    if "vix_close" in df.columns:
        v = df["vix_close"]
        f["vix_level"]      = v
        f["vix_z_60d"]      = rolling_z(v, 60)
        f["vix_change_21d"] = v.diff(21)

    # Yield curve
    if "yield_10y_3m" in df.columns:
        y = df["yield_10y_3m"]
        f["y10_3m"]         = y
        f["y10_3m_chg_63d"] = y.diff(63)
    if "yield_10y_2y" in df.columns:
        f["y10_2y"] = df["yield_10y_2y"]
    if "fed_funds_rate" in df.columns:
        ff = df["fed_funds_rate"]
        f["fed_funds"]    = ff
        f["ff_chg_252d"]  = ff.diff(252)

    # Credit
    cs_col = "baa_10y_spread" if "baa_10y_spread" in df.columns else (
        "credit_spread_bbb" if "credit_spread_bbb" in df.columns else None
    )
    if cs_col is not None:
        cs = df[cs_col]
        f["baa_10y_spread"] = cs
        f["baa_chg_63d"]    = cs.diff(63)
        f["baa_z_252d"]     = rolling_z(cs, 252)

    # Macro
    if "unemployment_rate" in df.columns:
        f["unemp_chg_252d"] = df["unemployment_rate"].diff(252)
    if "industrial_production" in df.columns:
        f["ip_chg_252d"] = safe_pct_chg(df["industrial_production"], 252)
    if "lei" in df.columns:
        f["lei_chg_252d"] = safe_pct_chg(df["lei"], 252)
    if "cpi" in df.columns:
        f["cpi_chg_252d"] = safe_pct_chg(df["cpi"], 252)
    if "m2_money_supply" in df.columns:
        f["m2_chg_252d"] = safe_pct_chg(df["m2_money_supply"], 252)

    # Sentiment
    if "consumer_sentiment" in df.columns:
        f["sentiment_z_252d"] = rolling_z(df["consumer_sentiment"], 252)
    if "epu_daily" in df.columns:
        f["epu_level"]   = df["epu_daily"]
        f["epu_ma_30d"]  = df["epu_daily"].rolling(30, min_periods=10).mean()

    # Cross-asset
    if "dollar_twi" in df.columns:
        f["dollar_chg_63d"] = safe_pct_chg(df["dollar_twi"], 63)
    if "oil_wti" in df.columns:
        oil = df["oil_wti"]
        f["oil_chg_63d"]  = safe_pct_chg(oil, 63)
        f["oil_vol_63d"]  = realized_vol(np.log(oil / oil.shift(1)), 63)

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    print("\nFeature coverage (non-null pct):")
    print("-" * 78)
    for name, _why in FEATURE_SPEC:
        if name in f.columns:
            n = f[name].notna().sum()
            pct = n / len(f) * 100
            flag = "✅" if pct >= 50 else ("⚠️" if pct > 5 else "❌")
            print(f"  {flag} {name:22s} {n:6,} / {len(f):,}  ({pct:5.1f}%)")
        else:
            print(f"  ❌ {name:22s}  (not built — DB column missing)")

    # Final clean-up: drop rows where all features are NaN (early dates).
    f = f.dropna(how="all")
    print(f"\nFinal feature matrix: {f.shape[0]:,} rows × {f.shape[1]} features")
    print(f"  span: {f.index.min().date()} → {f.index.max().date()}")

    # Save
    f.to_parquet(OUT_PATH)
    print(f"\n✅ Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
