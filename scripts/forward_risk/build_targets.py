"""Forward Risk — target generator.

For each date t and horizon h in {21, 63, 126, 252} trading days, computes:
  ret_fwd_h    = log(P_{t+h} / P_t)        forward log-return
  maxdd_fwd_h  = min over k in (0, h] of  log(P_{t+k} / max_{j<=k} P_{t+j})
                 i.e. the *worst* peak-to-trough log drawdown within (t, t+h]
                 (always <= 0)

The last h rows in the dataset will have NaN for horizon h (no future).

Output: data/processed/forward_risk_targets.parquet
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market_crash.db"
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "forward_risk_targets.parquet"

HORIZONS = [21, 63, 126, 252]  # 1m / 3m / 6m / 12m trading days


def forward_max_drawdown(p: np.ndarray, h: int) -> np.ndarray:
    """For each i, compute min_{k in (0, h]} log(P_{i+k} / max_{j in [i, i+k]} P_{i+j}).

    Vectorized via a sliding running max within the forward window.
    Returns array of length len(p); positions where i+h >= len(p) are NaN.
    """
    n = len(p)
    out = np.full(n, np.nan, dtype=float)
    if n <= h:
        return out
    log_p = np.log(p)
    for i in range(n - h):
        window = log_p[i : i + h + 1]                # P_t .. P_{t+h}
        running_max = np.maximum.accumulate(window)  # max from t to t+k
        # Drawdown at each k>=1 relative to running max up to k
        dd = window[1:] - running_max[1:]            # log returns from peak
        out[i] = float(dd.min())                     # worst drawdown (<=0)
    return out


def main() -> None:
    print("=" * 78)
    print("Forward Risk — target generator")
    print("=" * 78)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql(
            "SELECT date, nasdaq_close FROM indicators "
            "WHERE nasdaq_close IS NOT NULL ORDER BY date",
            conn, parse_dates=["date"],
        ).set_index("date").sort_index()
    df["nasdaq_close"] = pd.to_numeric(df["nasdaq_close"], errors="coerce")
    df = df.dropna()
    print(f"NASDAQ spine: {len(df):,} rows  {df.index.min().date()} -> {df.index.max().date()}")

    p = df["nasdaq_close"].to_numpy(dtype=float)
    log_p = np.log(p)

    out = pd.DataFrame(index=df.index)
    for h in HORIZONS:
        # Forward log-return
        ret_fwd = np.full(len(p), np.nan, dtype=float)
        ret_fwd[: len(p) - h] = log_p[h:] - log_p[: len(p) - h]
        out[f"ret_fwd_{h}"] = ret_fwd

        # Forward max drawdown
        out[f"maxdd_fwd_{h}"] = forward_max_drawdown(p, h)

    print("\nTarget coverage (non-null pct) and summary stats:")
    print("-" * 78)
    for h in HORIZONS:
        for tname in (f"ret_fwd_{h}", f"maxdd_fwd_{h}"):
            s = out[tname].dropna()
            n = len(s)
            pct = n / len(out) * 100
            print(f"  {tname:18s}  n={n:6,} ({pct:5.1f}%)  "
                  f"mean={s.mean():+.4f}  std={s.std():.4f}  "
                  f"p05={s.quantile(0.05):+.4f}  p50={s.median():+.4f}  p95={s.quantile(0.95):+.4f}")

    out.to_parquet(OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
