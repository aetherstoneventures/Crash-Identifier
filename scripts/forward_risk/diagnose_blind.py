"""Forward Risk — BLIND failure diagnostics.

Reads predictions written by train_walkforward.py and analyzes WHY coverage
broke on BLIND (>= 2021-01-01).

Outputs:
  - per-quantile hit rate (compare to nominal q for each q)
  - PIT histogram (uniform if calibrated)
  - rolling coverage of nominal 90% interval
  - residual = y - q50 over time, by regime
  - bias decomposition: where do misses concentrate (left tail vs right tail)?
  - sharpness: median CI90 width

Saves PNG plots to data/processed/forward_risk_diagnostics/.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
PRED_PATH = PROC / "forward_risk_predictions.parquet"
OUT_DIR = PROC / "forward_risk_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QCOLS = ["q05", "q25", "q50", "q75", "q95"]


def per_q_hitrate(df: pd.DataFrame) -> pd.DataFrame:
    """For each (horizon, target), report fraction of obs <= q for each q."""
    out = []
    for (h, t), g in df.groupby(["horizon", "target"]):
        g = g.dropna(subset=["y_true"])
        if g.empty:
            continue
        row = {"horizon": h, "target": t, "n": len(g)}
        for qval, qcol in zip(QUANTILES, QCOLS):
            row[f"hit_{qcol}"] = float((g["y_true"] <= g[qcol]).mean())
            row[f"nom_{qcol}"] = qval
        out.append(row)
    return pd.DataFrame(out)


def pit_values(df: pd.DataFrame) -> pd.Series:
    """Approx PIT via piecewise-linear interp of CDF through the 5 quantiles.

    For each row, find F(y) where F is interpolated through (q05, 0.05), ...,
    (q95, 0.95), with linear extrapolation clamped to [0, 1].
    """
    g = df.dropna(subset=["y_true"]).copy()
    qmat = g[QCOLS].to_numpy()
    y = g["y_true"].to_numpy()
    pit = np.empty(len(g))
    qs = np.array(QUANTILES)
    for i in range(len(g)):
        xs = qmat[i]
        # Ensure monotone for interp (already enforced at training, but defensive)
        order = np.argsort(xs)
        xs = xs[order]
        ys = qs[order]
        if y[i] <= xs[0]:
            pit[i] = max(0.0, ys[0] - (xs[0] - y[i]) * (ys[1] - ys[0]) / max(1e-9, xs[1] - xs[0]))
        elif y[i] >= xs[-1]:
            pit[i] = min(1.0, ys[-1] + (y[i] - xs[-1]) * (ys[-1] - ys[-2]) / max(1e-9, xs[-1] - xs[-2]))
        else:
            pit[i] = float(np.interp(y[i], xs, ys))
    return pd.Series(pit, index=g.index)


def main() -> None:
    print("=" * 78)
    print("Forward Risk — BLIND failure diagnostics")
    print("=" * 78)

    df = pd.read_parquet(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_blind"]].copy()
    print(f"BLIND rows: {len(df):,}  ({df['date'].min().date()} -> {df['date'].max().date()})")

    # 1) Per-quantile hit rates
    hr = per_q_hitrate(df)
    print("\nPer-quantile hit rates on BLIND (nominal vs actual):")
    print("-" * 78)
    for _, r in hr.iterrows():
        print(f"  h={int(r['horizon']):3d} {r['target']:5s}  n={int(r['n']):4d}  "
              f"q05 {r['hit_q05']:.3f}/{r['nom_q05']}  "
              f"q25 {r['hit_q25']:.3f}/{r['nom_q25']}  "
              f"q50 {r['hit_q50']:.3f}/{r['nom_q50']}  "
              f"q75 {r['hit_q75']:.3f}/{r['nom_q75']}  "
              f"q95 {r['hit_q95']:.3f}/{r['nom_q95']}")
    hr.to_csv(OUT_DIR / "per_quantile_hitrate.csv", index=False)

    # 2) Residuals (y - q50) over time, stratified by horizon for one target each
    for t in ["ret", "maxdd"]:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for ax, h in zip(axes, [21, 63, 126, 252]):
            g = df[(df["horizon"] == h) & (df["target"] == t)].dropna(subset=["y_true"]).sort_values("date")
            if g.empty:
                continue
            inside = (g["y_true"] >= g["q05"]) & (g["y_true"] <= g["q95"])
            cov = inside.rolling(60, min_periods=20).mean() if False else inside.astype(float).rolling(60, min_periods=20).mean()
            ax.plot(g["date"], g["y_true"], color="black", lw=0.8, label="actual")
            ax.fill_between(g["date"], g["q05"], g["q95"], alpha=0.2, color="steelblue", label="90% CI")
            ax.plot(g["date"], g["q50"], color="steelblue", lw=1.0, label="median")
            ax.set_title(f"{t} h={h}d  cov90={inside.mean():.3f}  width(med)={(g['q95']-g['q05']).median():.3f}")
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"BLIND fan vs actual — target={t}", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fan_blind_{t}.png", dpi=110)
        plt.close(fig)
        print(f"  wrote {OUT_DIR / f'fan_blind_{t}.png'}")

    # 3) PIT histograms (1 panel per (horizon, target))
    fig, axes = plt.subplots(4, 2, figsize=(11, 10))
    for col, t in enumerate(["ret", "maxdd"]):
        for row, h in enumerate([21, 63, 126, 252]):
            g = df[(df["horizon"] == h) & (df["target"] == t)]
            pit = pit_values(g)
            ax = axes[row, col]
            ax.hist(pit, bins=20, range=(0, 1), edgecolor="black", color="steelblue", alpha=0.7)
            ax.axhline(len(pit) / 20, color="red", lw=1, ls="--", label="uniform")
            # Test for U-shape (over-confident: too many extremes)
            extremes = float(((pit < 0.1) | (pit > 0.9)).mean())
            ax.set_title(f"{t} h={h}d   extremes(<0.1 or >0.9)={extremes:.3f} (nom 0.20)")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)
    fig.suptitle("PIT histograms on BLIND — flat=well calibrated, U-shape=over-confident", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pit_histograms.png", dpi=110)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'pit_histograms.png'}")

    # 4) When did coverage break? Rolling 60-day coverage of CI90.
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for col, t in enumerate(["ret", "maxdd"]):
        for row, h in enumerate([21, 63, 126, 252]):
            g = df[(df["horizon"] == h) & (df["target"] == t)].dropna(subset=["y_true"]).sort_values("date")
            inside = ((g["y_true"] >= g["q05"]) & (g["y_true"] <= g["q95"])).astype(float)
            roll = inside.rolling(60, min_periods=20).mean()
            ax = axes[row, col]
            ax.plot(g["date"], roll, color="steelblue")
            ax.axhline(0.90, color="green", ls="--", lw=1, label="nominal 0.90")
            ax.axhline(0.85, color="orange", ls=":", lw=1, label="lower bound 0.85")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{t} h={h}d  rolling-60d CI90 coverage")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Where did coverage break on BLIND?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rolling_coverage.png", dpi=110)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'rolling_coverage.png'}")

    # 5) Bias decomposition: misses below q05 vs above q95
    rows = []
    for (h, t), g in df.groupby(["horizon", "target"]):
        g = g.dropna(subset=["y_true"])
        below = float((g["y_true"] < g["q05"]).mean())
        above = float((g["y_true"] > g["q95"]).mean())
        rows.append(dict(horizon=h, target=t, n=len(g),
                         below_q05=below, above_q95=above,
                         total_outside=below + above,
                         med_width_90=float((g["q95"] - g["q05"]).median())))
    bd = pd.DataFrame(rows).sort_values(["target", "horizon"])
    print("\nBias decomposition (where misses concentrate):")
    print("-" * 78)
    print(bd.to_string(index=False))
    bd.to_csv(OUT_DIR / "bias_decomposition.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
