"""Forward Risk — split-conformal CQR recalibration (single shot at BLIND).

Method (Romano, Patterson, Candes 2019 — Conformalized Quantile Regression):
  Conformity score per row in calibration set:
        s_i = max( q_lo - y_i ,  y_i - q_hi )
  i.e. how far OUTSIDE the predicted interval the truth fell (negative if inside).
  At desired miscoverage alpha, the (1-alpha)*(n+1)/n empirical quantile of {s_i}
  gives an inflation tau such that
        new interval = [q_lo - tau,  q_hi + tau]
  has marginal coverage >= 1 - alpha on exchangeable test points.

Calibration set: folds 1, 2, 3 OOF predictions (1999-2020). These are TRUE
out-of-fold predictions (the model that produced them never saw those targets).
They DO NOT overlap with BLIND (which is >= 2021-01-01 within fold 4).

Test set: BLIND (>= 2021-01-01) within fold 4. ONE evaluation.

Outputs:
  data/processed/forward_risk_predictions_conformal.parquet
  data/processed/forward_risk_conformal_summary.csv
  data/processed/forward_risk_diagnostics/conformal_*.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
PRED_PATH = PROC / "forward_risk_predictions.parquet"
OUT_PRED = PROC / "forward_risk_predictions_conformal.parquet"
SUMMARY_OUT = PROC / "forward_risk_conformal_summary.csv"
DIAG_DIR = PROC / "forward_risk_diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Coverage levels we want to certify — choose nominal 90% (q05 / q95)
# and 50% (q25 / q75) separately, each with its own conformal tau.
LEVELS = [
    ("q05", "q95", 0.10),  # 90% interval, alpha=0.10
    ("q25", "q75", 0.50),  # 50% interval, alpha=0.50
]


def conformal_tau(scores: np.ndarray, alpha: float) -> float:
    """Empirical (1-alpha)*(n+1)/n quantile of conformity scores."""
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def main() -> None:
    print("=" * 78)
    print("Forward Risk — split-conformal CQR recalibration")
    print("=" * 78)

    df = pd.read_parquet(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Calibration: folds 1-3 (any non-BLIND row in those folds with observed y)
    cal = df[(df["fold"].isin([1, 2, 3])) & df["y_true"].notna()].copy()
    test = df[df["is_blind"] & df["y_true"].notna()].copy()
    print(f"Calibration rows (folds 1-3 OOF, 1999-2020): {len(cal):,}")
    print(f"BLIND test rows (>= 2021-01-01):              {len(test):,}")

    # Note: each row has all 4 horizons x 2 targets stacked via groupby below.
    rows_summary: list[dict] = []
    out_rows: list[dict] = []

    for (h, t), g_test in test.groupby(["horizon", "target"]):
        g_cal = cal[(cal["horizon"] == h) & (cal["target"] == t)]
        if g_cal.empty:
            continue

        rec = {"horizon": int(h), "target": t, "n_cal": len(g_cal), "n_blind": len(g_test)}

        # Compute new intervals for each level
        adjusted = g_test.copy()
        for q_lo, q_hi, alpha in LEVELS:
            s = np.maximum(g_cal[q_lo] - g_cal["y_true"], g_cal["y_true"] - g_cal[q_hi]).to_numpy()
            tau = conformal_tau(s, alpha)
            new_lo = g_test[q_lo] - tau
            new_hi = g_test[q_hi] + tau

            # Pre-conformal coverage
            pre_cov = float(((g_test["y_true"] >= g_test[q_lo]) & (g_test["y_true"] <= g_test[q_hi])).mean())
            # Post-conformal coverage
            post_cov = float(((g_test["y_true"] >= new_lo) & (g_test["y_true"] <= new_hi)).mean())
            pre_width = float((g_test[q_hi] - g_test[q_lo]).median())
            post_width = float((new_hi - new_lo).median())

            tag = f"{int((1-alpha)*100)}"
            rec[f"tau_{tag}"]      = tau
            rec[f"cov_pre_{tag}"]  = pre_cov
            rec[f"cov_post_{tag}"] = post_cov
            rec[f"width_pre_{tag}"]  = pre_width
            rec[f"width_post_{tag}"] = post_width

            adjusted[f"{q_lo}_conf"] = new_lo.values
            adjusted[f"{q_hi}_conf"] = new_hi.values

        rows_summary.append(rec)

        # Persist per-row adjusted intervals
        for _, r in adjusted.iterrows():
            out_rows.append({
                "date": r["date"], "horizon": int(h), "target": t,
                "q05_orig": r["q05"], "q25_orig": r["q25"], "q50": r["q50"],
                "q75_orig": r["q75"], "q95_orig": r["q95"],
                "q05_conf": r.get("q05_conf"), "q25_conf": r.get("q25_conf"),
                "q75_conf": r.get("q75_conf"), "q95_conf": r.get("q95_conf"),
                "y_true": r["y_true"],
            })

    summ = pd.DataFrame(rows_summary).sort_values(["target", "horizon"])
    print("\nConformal recalibration results on BLIND:")
    print("-" * 78)
    cols_show = ["horizon", "target", "n_cal", "n_blind",
                 "tau_90", "cov_pre_90", "cov_post_90", "width_pre_90", "width_post_90",
                 "tau_50", "cov_pre_50", "cov_post_50"]
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summ[cols_show].to_string(index=False))
    summ.to_csv(SUMMARY_OUT, index=False)
    pd.DataFrame(out_rows).to_parquet(OUT_PRED)
    print(f"\nWrote {SUMMARY_OUT}")
    print(f"Wrote {OUT_PRED}")

    # ---------------------------------------------------------------------
    # Kill criterion: BLIND post-conformal CI90 coverage in [0.85, 0.95]
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Kill criterion check (target CI90 coverage in [0.85, 0.95]):")
    print("=" * 78)
    pass_n = 0
    fail_n = 0
    for _, r in summ.iterrows():
        c = r["cov_post_90"]
        ok = 0.85 <= c <= 0.95
        flag = "PASS" if ok else "FAIL"
        if ok:
            pass_n += 1
        else:
            fail_n += 1
        print(f"  h={int(r['horizon']):3d} {r['target']:5s}  cov90 pre={r['cov_pre_90']:.3f}"
              f"  post={r['cov_post_90']:.3f}  width pre={r['width_pre_90']:.3f}"
              f"  post={r['width_post_90']:.3f}   {flag}")
    print(f"\nOverall: {pass_n}/{pass_n + fail_n} cells PASS the [0.85, 0.95] band.")

    # Plot before/after fans
    long = pd.DataFrame(out_rows)
    for t in ["ret", "maxdd"]:
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for ax, h in zip(axes, [21, 63, 126, 252]):
            g = long[(long["horizon"] == h) & (long["target"] == t)].sort_values("date")
            if g.empty:
                continue
            ax.fill_between(g["date"], g["q05_conf"], g["q95_conf"], alpha=0.15, color="orange",
                            label="post-conformal 90%")
            ax.fill_between(g["date"], g["q05_orig"], g["q95_orig"], alpha=0.30, color="steelblue",
                            label="pre-conformal 90%")
            ax.plot(g["date"], g["q50"], color="steelblue", lw=1.0, label="median")
            ax.plot(g["date"], g["y_true"], color="black", lw=0.8, label="actual")
            inside_pre  = ((g["y_true"] >= g["q05_orig"]) & (g["y_true"] <= g["q95_orig"])).mean()
            inside_post = ((g["y_true"] >= g["q05_conf"]) & (g["y_true"] <= g["q95_conf"])).mean()
            ax.set_title(f"{t} h={h}d   pre cov90={inside_pre:.3f}   post cov90={inside_post:.3f}")
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Conformal recalibration on BLIND — {t}", fontsize=12)
        fig.tight_layout()
        out = DIAG_DIR / f"conformal_blind_{t}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
