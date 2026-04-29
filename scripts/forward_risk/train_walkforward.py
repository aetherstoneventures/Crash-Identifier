"""Forward Risk — walk-forward LightGBM quantile trainer.

Trains 40 models = 5 quantiles x 4 horizons x 2 targets, in expanding-window
walk-forward, on NASDAQ 1982-2026.

Folds (same boundaries as v5):
  Fold 1: train < 1999-01-01,  eval [1999-01-01, 2005-01-01)
  Fold 2: train < 2005-01-01,  eval [2005-01-01, 2012-01-01)
  Fold 3: train < 2012-01-01,  eval [2012-01-01, 2020-01-01)
  Fold 4: train < 2020-01-01,  eval [2020-01-01, 2026-12-31]   ** contains BLIND >= 2021 **

Target leakage guard:
  When training with cutoff e_train, the latest usable target row is e_train - h
  (since y_{h,t} uses prices up to t+h). We drop rows from training where t > e_train - h.

Discipline:
  * Fixed hyperparameters (see HPARAMS) — NO search touching BLIND.
  * Pinball loss reported per fold; one BLIND evaluation only.

Output: data/processed/forward_risk_predictions.parquet (long format).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FEATS_PATH   = PROC / "forward_risk_features.parquet"
TARGETS_PATH = PROC / "forward_risk_targets.parquet"
OUT_PATH     = PROC / "forward_risk_predictions.parquet"

HORIZONS  = [21, 63, 126, 252]
TARGETS   = ["ret", "maxdd"]                 # underlying target families
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

FOLD_BOUNDS = [
    ("1999-01-01", "2005-01-01"),
    ("2005-01-01", "2012-01-01"),
    ("2012-01-01", "2020-01-01"),
    ("2020-01-01", "2026-12-31"),
]
BLIND_START = pd.Timestamp("2021-01-01")

# Disciplined small-model config — chosen from priors, not tuned on BLIND.
HPARAMS = dict(
    objective="quantile",
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1,
)


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    diff = y - q_pred
    return float(np.maximum(q * diff, (q - 1) * diff).mean())


def main() -> None:
    print("=" * 78)
    print("Forward Risk — walk-forward quantile trainer")
    print("=" * 78)

    feats = pd.read_parquet(FEATS_PATH)
    targs = pd.read_parquet(TARGETS_PATH)
    print(f"Features:  {feats.shape}  span {feats.index.min().date()} -> {feats.index.max().date()}")
    print(f"Targets:   {targs.shape}  span {targs.index.min().date()} -> {targs.index.max().date()}")

    feature_cols = list(feats.columns)
    full = feats.join(targs, how="inner").sort_index()
    print(f"Joined panel: {len(full):,} rows x {len(feature_cols)} features")

    rows: list[dict] = []
    fold_summary: list[dict] = []

    for fi, (a, b) in enumerate(FOLD_BOUNDS, start=1):
        a_ts, b_ts = pd.Timestamp(a), pd.Timestamp(b)
        eval_mask = (full.index >= a_ts) & (full.index < b_ts)
        if not eval_mask.any():
            print(f"\nFold {fi}: empty eval slice {a} -> {b} (skipping)")
            continue
        eval_df = full[eval_mask]
        print(f"\nFold {fi}: train < {a}    eval [{a}, {b})    eval rows={len(eval_df):,}")

        for h in HORIZONS:
            # Training cutoff with leakage guard
            train_max_date = a_ts - pd.Timedelta(days=1) - pd.Timedelta(days=int(h * 1.6))
            #  ^ ~h trading days back ≈ h * 1.6 calendar days, loose buffer
            for tname in TARGETS:
                ycol = f"{tname}_fwd_{h}"
                # TRAIN: rows strictly before fold start AND with non-null y
                tr_mask = (full.index <= train_max_date) & full[ycol].notna()
                tr = full[tr_mask]
                if len(tr) < 500:
                    print(f"  h={h:3d} {tname:5s} skip — only {len(tr)} train rows")
                    continue

                X_tr = tr[feature_cols].astype(float).to_numpy()
                y_tr = tr[ycol].astype(float).to_numpy()

                # EVAL: rows in fold slice with feature row available
                ev = eval_df[eval_df[ycol].notna() | True].copy()
                X_ev = ev[feature_cols].astype(float).to_numpy()

                # NaN handling: LightGBM handles NaN natively, that's fine.
                preds_per_q: dict[float, np.ndarray] = {}
                for q in QUANTILES:
                    model = lgb.LGBMRegressor(alpha=q, **HPARAMS)
                    model.fit(X_tr, y_tr)
                    preds_per_q[q] = model.predict(X_ev)

                # Stack & enforce monotone non-crossing via sort.
                Q = np.stack([preds_per_q[q] for q in QUANTILES], axis=1)
                Q = np.sort(Q, axis=1)

                # Pinball summary on rows where truth is observed.
                y_ev = ev[ycol].to_numpy()
                obs = ~np.isnan(y_ev)
                pin_total = 0.0
                if obs.any():
                    for j, q in enumerate(QUANTILES):
                        pin_total += pinball(y_ev[obs], Q[obs, j], q)
                    pin_total /= len(QUANTILES)

                blind_obs = obs & np.asarray(ev.index >= BLIND_START)
                pin_blind = np.nan
                cov90_blind = np.nan
                if blind_obs.any():
                    pin_blind_acc = 0.0
                    for j, q in enumerate(QUANTILES):
                        pin_blind_acc += pinball(y_ev[blind_obs], Q[blind_obs, j], q)
                    pin_blind = pin_blind_acc / len(QUANTILES)
                    inside = (y_ev[blind_obs] >= Q[blind_obs, 0]) & (y_ev[blind_obs] <= Q[blind_obs, 4])
                    cov90_blind = float(inside.mean())

                fold_summary.append(dict(
                    fold=fi, horizon=h, target=tname,
                    n_train=int(len(tr)), n_eval_obs=int(obs.sum()),
                    pinball_eval=pin_total,
                    n_blind=int(blind_obs.sum()),
                    pinball_blind=pin_blind,
                    cov90_blind=cov90_blind,
                ))
                print(f"  h={h:3d} {tname:5s}  n_tr={len(tr):5d}  pinball={pin_total:.4f}"
                      f"  blind n={int(blind_obs.sum()):4d} pin={pin_blind:.4f} cov90={cov90_blind:.3f}"
                      if blind_obs.any() else
                      f"  h={h:3d} {tname:5s}  n_tr={len(tr):5d}  pinball={pin_total:.4f}")

                # Persist long-format predictions
                for k, dt in enumerate(ev.index):
                    rows.append(dict(
                        date=dt, fold=fi, horizon=h, target=tname,
                        q05=Q[k, 0], q25=Q[k, 1], q50=Q[k, 2], q75=Q[k, 3], q95=Q[k, 4],
                        y_true=float(y_ev[k]) if not np.isnan(y_ev[k]) else np.nan,
                        is_blind=bool(dt >= BLIND_START),
                    ))

    preds = pd.DataFrame(rows)
    preds.to_parquet(OUT_PATH)
    summ = pd.DataFrame(fold_summary)
    summ.to_csv(PROC / "forward_risk_fold_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("Fold summary:")
    print("=" * 78)
    print(summ.to_string(index=False))
    print(f"\nWrote {OUT_PATH}  ({len(preds):,} rows)")
    print(f"Wrote {PROC / 'forward_risk_fold_summary.csv'}")


if __name__ == "__main__":
    main()
