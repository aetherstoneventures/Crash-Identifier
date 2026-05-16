"""v6 Feature Vector Builder.

Loads daily indicators from the project SQLite DB and assembles the
~40-feature point-in-time vector specified in
`docs/CRASH_KPI_ENGINE_DESIGN.md` (sections 4.1 – 4.6).

DESIGN RULES (binding):

1. NO LOOK-AHEAD. All rolling stats use `min_periods` and shift where
   appropriate. Z-scores are computed with EXPANDING-window means/stds
   (rolling does not constitute leakage but is less appropriate for
   regime drift).
2. NO SOCIAL-MEDIA / NEWS-NLP SENTIMENT for index crashes (per design).
3. Options-implied sentiment (VIX, VIX9D, SKEW, put/call, EPU, GPR
   proxy) IS included.
4. If an underlying indicator column is missing from the DB on a given
   date, the corresponding feature is NaN — downstream engines must
   handle NaN with imputation or row-drop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.v6.config import DB_PATH


# ---------------------------------------------------------------------------
# Feature catalogue — single source of truth for downstream engines.
# Names must be unique across groups. Each name corresponds to a column
# in the DataFrame returned by `FeatureBuilder.build()`.
# ---------------------------------------------------------------------------
FEATURE_GROUPS: Dict[str, List[str]] = {
    # 4.1 Mathematical / statistical
    "math_stat": [
        "rv_21_z", "rv_63_z", "rv_252_z",
        "skew_63", "kurt_63",
        "dd_from_252h", "days_since_252h",
        "acf_10", "acf_20",
        "range_atr_z",
        "iv_rv_gap",
        "return_dispersion_63",
    ],
    # 4.2 Macro / economic
    "macro": [
        "yc_10y_3m", "yc_10y_2y", "yc_10y_3m_chg",
        "hy_spread_z", "hy_spread_chg",
        "ig_spread_z",
        "nfci", "nfci_chg",
        "epu_z",
        "sahm_indicator",
        "cape_proxy_z",
        "m2_yoy_z",
        "margin_debt_z",
        "real_rate_10y",
    ],
    # 4.3 Options-implied sentiment (KEPT — see design §4.3)
    "options_sentiment": [
        "vix_z", "vix_term_structure",
        "skew_z",
        "put_call_z",
    ],
    # 4.4 Breadth / technical
    "breadth": [
        "ma50_dist", "ma200_dist", "ma50_above_ma200",
        "cross_asset_corr_z",
        "dxy_z",
    ],
    # 4.5 Geopolitical / event
    "geo_event": [
        "oil_shock_z",
    ],
}

ALL_FEATURES: List[str] = [f for group in FEATURE_GROUPS.values() for f in group]


# ---------------------------------------------------------------------------
# Helper utilities (leakage-safe)
# ---------------------------------------------------------------------------
def _expanding_z(series: pd.Series, min_periods: int = 252) -> pd.Series:
    """Expanding-window z-score (point-in-time, no leakage)."""
    s = series.astype(float)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    z = (s - mu) / sd.replace(0, np.nan)
    return z


def _safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
    return s.astype(float).pct_change(periods=periods, fill_method=None)


def _safe_diff(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.astype(float).diff(periods)


def _rolling_acf(returns: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """Rolling autocorrelation at given lag."""
    def _acf(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        if len(x) < lag + 2:
            return np.nan
        return float(np.corrcoef(x[lag:], x[:-lag])[0, 1])
    return returns.rolling(window, min_periods=window).apply(_acf, raw=True)


def _first_available(raw: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return the first candidate column that exists AND has at least one
    non-null value. Returns None if none qualify (downstream uses NaN)."""
    for name in candidates:
        if name in raw.columns:
            s = raw[name]
            if s.notna().any():
                return s.astype(float)
    return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
@dataclass
class FeatureBuilder:
    """Builds the v6 feature DataFrame from the raw indicators table."""

    db_path: str = str(DB_PATH)
    price_col: str = "sp500_close"   # primary equity index for math/stat features
    secondary_price_col: str = "nasdaq_close"

    def load_raw(self) -> pd.DataFrame:
        """Load the indicators table as a date-indexed DataFrame."""
        import sqlite3
        with sqlite3.connect(self.db_path) as con:
            df = pd.read_sql_query(
                "SELECT * FROM indicators ORDER BY date ASC", con
            )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Drop bookkeeping columns.
        for col in ("id", "created_at", "updated_at"):
            if col in df.columns:
                df = df.drop(columns=col)
        return df

    # ---------------------------------------------------------------------
    # Feature subroutines
    # ---------------------------------------------------------------------
    def _math_stat(self, raw: pd.DataFrame) -> pd.DataFrame:
        px = raw[self.price_col].astype(float)
        ret = px.pct_change(fill_method=None)
        out = pd.DataFrame(index=raw.index)

        # Realised vol (annualised, rolling), then expanding z
        for w in (21, 63, 252):
            rv = ret.rolling(w, min_periods=w).std() * np.sqrt(252)
            out[f"rv_{w}_z"] = _expanding_z(rv, min_periods=max(252, w))

        out["skew_63"] = ret.rolling(63, min_periods=63).skew()
        out["kurt_63"] = ret.rolling(63, min_periods=63).kurt()

        # Drawdown from rolling 252d high
        roll_max = px.rolling(252, min_periods=252).max()
        out["dd_from_252h"] = (px / roll_max - 1.0) * 100.0

        # Days since the 252d high
        def _days_since_max(x: np.ndarray) -> float:
            if np.isnan(x).all():
                return np.nan
            return float(len(x) - 1 - int(np.nanargmax(x)))
        out["days_since_252h"] = px.rolling(252, min_periods=252).apply(_days_since_max, raw=True)

        out["acf_10"] = _rolling_acf(ret, window=63, lag=10)
        out["acf_20"] = _rolling_acf(ret, window=126, lag=20)

        # ATR-style range expansion: |return| rolling mean, then z
        atr_proxy = ret.abs().rolling(21, min_periods=21).mean()
        out["range_atr_z"] = _expanding_z(atr_proxy, min_periods=252)

        # IV - RV gap (annualised VIX / 100 - realised vol)
        if "vix_close" in raw.columns:
            rv_21 = ret.rolling(21, min_periods=21).std() * np.sqrt(252)
            out["iv_rv_gap"] = (raw["vix_close"].astype(float) / 100.0) - rv_21
        else:
            out["iv_rv_gap"] = np.nan

        # Cross-sectional dispersion proxy: std of daily returns over 63d
        out["return_dispersion_63"] = ret.rolling(63, min_periods=63).std()

        return out

    def _macro(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)

        yc_10y_3m = _first_available(raw, ["yield_10y_3m"])
        out["yc_10y_3m"] = yc_10y_3m if yc_10y_3m is not None else np.nan
        yc_10y_2y = _first_available(raw, ["yield_10y_2y"])
        out["yc_10y_2y"] = yc_10y_2y if yc_10y_2y is not None else np.nan
        out["yc_10y_3m_chg"] = (
            _safe_diff(out["yc_10y_3m"], 20) if yc_10y_3m is not None else np.nan
        )

        # HY OAS: prefer 'hy_spread', fallback to BAA-10Y as a proxy
        hy = _first_available(raw, ["hy_spread", "baa_10y_spread"])
        if hy is not None:
            out["hy_spread_z"] = _expanding_z(hy, min_periods=252)
            out["hy_spread_chg"] = _safe_diff(hy, 20)
        else:
            out["hy_spread_z"] = np.nan
            out["hy_spread_chg"] = np.nan

        ig = _first_available(raw, ["aaa_10y_spread", "baa_10y_spread", "credit_spread_bbb"])
        out["ig_spread_z"] = _expanding_z(ig, min_periods=252) if ig is not None else np.nan

        nfci = _first_available(raw, ["nfci", "anfci", "kcfsi"])
        if nfci is not None:
            out["nfci"] = nfci
            out["nfci_chg"] = _safe_diff(nfci, 20)
        else:
            out["nfci"] = np.nan
            out["nfci_chg"] = np.nan

        epu = _first_available(raw, ["epu_index", "epu_daily"])
        out["epu_z"] = _expanding_z(epu, min_periods=252) if epu is not None else np.nan

        # Sahm rule: 3m MA UE − min(12m trailing 3m MA UE)
        ue = _first_available(raw, ["unemployment_rate"])
        if ue is not None:
            ue_3m = ue.rolling(63, min_periods=63).mean()
            ue_12m_min = ue_3m.rolling(252, min_periods=252).min()
            out["sahm_indicator"] = ue_3m - ue_12m_min
        else:
            out["sahm_indicator"] = np.nan

        # CAPE proxy: price / 10y rolling mean (lacking real earnings)
        px = raw[self.price_col].astype(float)
        cape_proxy = px / px.rolling(252 * 10, min_periods=252 * 5).mean()
        out["cape_proxy_z"] = _expanding_z(cape_proxy, min_periods=252)

        m2 = _first_available(raw, ["m2_money_supply"])
        if m2 is not None:
            out["m2_yoy_z"] = _expanding_z(_safe_pct_change(m2, 252), min_periods=252)
        else:
            out["m2_yoy_z"] = np.nan

        md = _first_available(raw, ["margin_debt"])
        out["margin_debt_z"] = _expanding_z(md, min_periods=252) if md is not None else np.nan

        rr = _first_available(raw, ["dfii10", "t10yie"])
        if rr is not None:
            out["real_rate_10y"] = rr
        else:
            y10 = _first_available(raw, ["yield_10y"])
            cpi = _first_available(raw, ["cpi"])
            if y10 is not None and cpi is not None:
                cpi_yoy = _safe_pct_change(cpi, 252) * 100.0
                out["real_rate_10y"] = y10 - cpi_yoy
            else:
                out["real_rate_10y"] = np.nan

        return out

    def _options_sentiment(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        vix = _first_available(raw, ["vix_close"])
        out["vix_z"] = _expanding_z(vix, min_periods=252) if vix is not None else np.nan

        # VIX term structure: VIX9D / VIX (>1.0 = stress inversion).
        # Fall back to VIX/VXV (>1.0 also = inversion) if VIX9D missing.
        vix9d = _first_available(raw, ["v6_vix9d"])
        vxv = _first_available(raw, ["vxv_close"])
        if vix9d is not None and vix is not None:
            out["vix_term_structure"] = vix9d / vix.replace(0, np.nan)
        elif vxv is not None and vix is not None:
            out["vix_term_structure"] = vix / vxv.replace(0, np.nan)
        else:
            out["vix_term_structure"] = np.nan

        skew = _first_available(raw, ["v6_skew"])
        out["skew_z"] = _expanding_z(skew, min_periods=252) if skew is not None else np.nan

        pc = _first_available(raw, ["put_call_ratio"])
        out["put_call_z"] = _expanding_z(pc, min_periods=252) if pc is not None else np.nan

        return out

    def _breadth(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        px = raw[self.price_col].astype(float)
        ma50 = px.rolling(50, min_periods=50).mean()
        ma200 = px.rolling(200, min_periods=200).mean()
        out["ma50_dist"] = (px / ma50 - 1.0) * 100.0
        out["ma200_dist"] = (px / ma200 - 1.0) * 100.0
        out["ma50_above_ma200"] = (ma50 > ma200).astype(float)

        # Cross-asset corr: prefer TLT, fallback to 10y yield (inverse proxy)
        bond = _first_available(raw, ["v6_tlt"])
        if bond is None:
            y10 = _first_available(raw, ["yield_10y"])
            if y10 is not None:
                # Convert yield to a price proxy by negating change.
                bond = -y10
        if bond is not None:
            bond_ret = bond.pct_change(fill_method=None) if (bond > 0).all() else bond.diff()
            eq_ret = px.pct_change(fill_method=None)
            corr = eq_ret.rolling(63, min_periods=63).corr(bond_ret)
            out["cross_asset_corr_z"] = _expanding_z(corr, min_periods=252)
        else:
            out["cross_asset_corr_z"] = np.nan

        dxy = _first_available(raw, ["dollar_twi"])
        out["dxy_z"] = _expanding_z(dxy, min_periods=252) if dxy is not None else np.nan

        return out

    def _geo_event(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        oil = _first_available(raw, ["oil_wti"])
        if oil is not None:
            out["oil_shock_z"] = _expanding_z(_safe_pct_change(oil, 21), min_periods=252)
        else:
            out["oil_shock_z"] = np.nan
        return out

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def build(self, raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Assemble the full ~40-column point-in-time feature DataFrame."""
        if raw is None:
            raw = self.load_raw()
        parts = [
            self._math_stat(raw),
            self._macro(raw),
            self._options_sentiment(raw),
            self._breadth(raw),
            self._geo_event(raw),
        ]
        out = pd.concat(parts, axis=1)
        # Ensure column order matches the catalogue.
        for col in ALL_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
        out = out[ALL_FEATURES]
        # Attach the primary price column for downstream use.
        out["_price"] = raw[self.price_col].astype(float)
        return out
