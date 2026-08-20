"""v6 Feature Vector Builder.

Loads daily indicators from the project SQLite DB and assembles the
point-in-time feature vector specified in
`docs/CRASH_KPI_ENGINE_DESIGN.md` (sections 4.1 – 4.6).

DESIGN RULES (binding):

1. NO LOOK-AHEAD. All rolling stats use `min_periods`. Z-scores use
   EXPANDING-window means/stds, so a value's z-score only ever reflects
   history available at that date. Macro series arrive pre-shifted to
   their publication dates by `scripts/data/backfill_fred.py`.
2. NO FABRICATED INPUTS. Every raw column passes through
   `src/v6/features/quality.py` before it can become a feature. A column
   that fails is excluded with a recorded reason, never silently modelled.
3. THE PRICE COLUMN IS RESOLVED, NOT ASSUMED. See `resolve_price_column`.
4. NO SOCIAL-MEDIA / NEWS-NLP SENTIMENT for index crashes (per design).

WHY THE PRICE COLUMN IS RESOLVED
--------------------------------
v6.0.0-alpha hard-coded `price_col = "sp500_close"`. That column is fed by
FRED's `SP500` series, which is licensed as a **rolling 10-year window**,
so it began in 2016. Every price-derived feature here — realised vol,
drawdown, moving averages, skew, kurtosis — and the crash labels
themselves therefore existed only for the last ~20% of the sample. The
engines were not wrong; they were blind. `resolve_price_column` now picks
the longest quality-passing series (Nasdaq Composite, 1971+), which is
also this project's stated target index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.v6.config import DB_PATH
from src.v6.features.quality import (
    apply_quality_screen, rejected_columns, screen_column,
)


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
        "downside_vol_ratio_63",
    ],
    # 4.2 Macro / economic
    "macro": [
        "yc_10y_3m", "yc_10y_2y", "yc_10y_3m_chg",
        "hy_spread_z", "hy_spread_chg",
        "ig_spread_z",
        "nfci", "nfci_chg",
        "nfci_leverage", "nfci_credit",
        "stlfsi_z",
        "epu_z",
        "sahm_indicator",
        "claims_13w_chg_z",
        "cape_proxy_z",
        "m2_yoy_z",
        "real_rate_10y",
    ],
    # 4.3 Options-implied sentiment (KEPT — see design §4.3)
    "options_sentiment": [
        "vix_z", "vix_term_structure", "vix_shock_5d",
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

# Features from the alpha catalogue that are NOT built, and why. Kept
# visible so the omissions are auditable rather than forgotten.
UNAVAILABLE_FEATURES: Dict[str, str] = {
    "put_call_z": (
        "Source column `put_call_ratio` is synthetic noise (see "
        "quality.QUARANTINE). CBOE equity put/call is not on FRED and has "
        "no free full-history source."
    ),
    "skew_z": (
        "CBOE SKEW index is not available on FRED and `v6_skew` was never "
        "populated. Tail-risk pricing is partly covered by "
        "`vix_term_structure`."
    ),
    "margin_debt_z": (
        "Source column `margin_debt` is a constant fill for 93% of its "
        "history (see quality screen). FINRA margin debt needs a separate "
        "scraper."
    ),
}

# Candidate price columns, in preference order. The resolver takes the
# longest quality-passing series, so preference only breaks ties.
PRICE_CANDIDATES: Tuple[str, ...] = ("nasdaq_close", "sp500_close")


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


def resolve_price_column(raw: pd.DataFrame,
                         candidates: Tuple[str, ...] = PRICE_CANDIDATES) -> str:
    """Pick the price column with the longest usable history.

    A price series drives realised vol, drawdown, moving averages and the
    crash labels themselves, so a short one silently truncates the entire
    feature matrix. We therefore choose by evidence — quality screen plus
    observation count — rather than by hard-coded name.

    Raises
    ------
    RuntimeError
        If no candidate column passes the quality screen.
    """
    scored: List[Tuple[int, int, str]] = []
    failures: Dict[str, str] = {}
    for rank, name in enumerate(candidates):
        if name not in raw.columns:
            failures[name] = "column not present in the indicators table"
            continue
        verdict = screen_column(raw[name], name)
        if not verdict.passed:
            failures[name] = verdict.reason
            continue
        # Longest history wins; preference order breaks ties.
        scored.append((verdict.n_obs, -rank, name))
    if not scored:
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items()) or "no candidates"
        raise RuntimeError(f"No usable price column. Checked — {detail}")
    scored.sort(reverse=True)
    return scored[0][2]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
@dataclass
class FeatureBuilder:
    """Builds the v6 feature DataFrame from the raw indicators table.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    price_col : str | None
        Force a specific price column. Leave as None (default) to resolve
        the longest quality-passing series automatically.
    screen_quality : bool
        Drop raw columns that fail the data-quality screen before building.
    """

    db_path: str = str(DB_PATH)
    price_col: Optional[str] = None
    screen_quality: bool = True
    resolved_price_col_: Optional[str] = field(default=None, init=False)
    rejected_: Dict[str, str] = field(default_factory=dict, init=False)

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
    def _price(self, raw: pd.DataFrame) -> pd.Series:
        """The resolved primary price series."""
        return raw[self.resolved_price_col_].astype(float)

    def _math_stat(self, raw: pd.DataFrame) -> pd.DataFrame:
        px = self._price(raw)
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

        # IV - RV gap (annualised VIX / 100 - realised vol). A collapsing or
        # negative gap means options are cheap relative to realised risk.
        vix = _first_available(raw, ["vix_close"])
        rv_21 = ret.rolling(21, min_periods=21).std() * np.sqrt(252)
        out["iv_rv_gap"] = (vix / 100.0) - rv_21 if vix is not None else np.nan

        # Dispersion of daily returns over 63d
        out["return_dispersion_63"] = ret.rolling(63, min_periods=63).std()

        # Downside/upside vol ratio: rising means losses are getting more
        # violent than gains — an asymmetry that precedes disorderly selling.
        # min_periods is set against the ~31 same-signed days a 63-day window
        # holds, not against 63, or nearly every window fails the threshold.
        down = ret.where(ret < 0).rolling(63, min_periods=15).std()
        up = ret.where(ret > 0).rolling(63, min_periods=15).std()
        out["downside_vol_ratio_63"] = down / up.replace(0, np.nan)

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

        # HY OAS: prefer 'hy_spread', fallback to BAA-10Y as a proxy.
        # (ICE BofA OAS series on FRED are now a rolling 3-year window, so
        # BAA10Y is the only long credit-spread history available free.)
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

        # NFCI sub-indices separate *leverage* build-up from *credit* stress —
        # the distinction between a 2008-style balance-sheet crash and a
        # liquidity event.
        lev = _first_available(raw, ["nfci_leverage"])
        out["nfci_leverage"] = lev if lev is not None else np.nan
        cred = _first_available(raw, ["nfci_credit"])
        out["nfci_credit"] = cred if cred is not None else np.nan

        stl = _first_available(raw, ["stlfsi"])
        out["stlfsi_z"] = _expanding_z(stl, min_periods=252) if stl is not None else np.nan

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

        # Initial claims 13-week change — the highest-frequency labour signal,
        # and the one that turns first in a genuine downturn.
        claims = _first_available(raw, ["initial_claims"])
        if claims is not None:
            out["claims_13w_chg_z"] = _expanding_z(
                _safe_pct_change(claims, 65), min_periods=252
            )
        else:
            out["claims_13w_chg_z"] = np.nan

        # CAPE proxy: price / 10y rolling mean (lacking real earnings)
        px = self._price(raw)
        cape_proxy = px / px.rolling(252 * 10, min_periods=252 * 5).mean()
        out["cape_proxy_z"] = _expanding_z(cape_proxy, min_periods=252)

        m2 = _first_available(raw, ["m2_money_supply"])
        if m2 is not None:
            out["m2_yoy_z"] = _expanding_z(_safe_pct_change(m2, 252), min_periods=252)
        else:
            out["m2_yoy_z"] = np.nan

        rr = _first_available(raw, ["dfii10"])
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

        # VIX term structure: VIX / VIX3M. Above 1.0 = inverted = the market
        # is paying up for immediate protection, the classic stress tell.
        vxv = _first_available(raw, ["vxv_close"])
        if vix is not None and vxv is not None:
            out["vix_term_structure"] = vix / vxv.replace(0, np.nan)
        else:
            out["vix_term_structure"] = np.nan

        # 5-day VIX shock: the speed of the repricing, not its level.
        out["vix_shock_5d"] = (
            _expanding_z(_safe_pct_change(vix, 5), min_periods=252)
            if vix is not None else np.nan
        )
        return out

    def _breadth(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        px = self._price(raw)
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
        """Assemble the full point-in-time feature DataFrame.

        Returns a DataFrame with one column per entry in `ALL_FEATURES`,
        plus `_price` (the resolved primary price series).
        """
        if raw is None:
            raw = self.load_raw()

        # 1. Quality screen — record what was rejected, then drop it.
        self.rejected_ = rejected_columns(raw)
        if self.screen_quality:
            raw = apply_quality_screen(raw)

        # 2. Resolve the price column from what survived.
        self.resolved_price_col_ = self.price_col or resolve_price_column(raw)
        if self.resolved_price_col_ not in raw.columns:
            raise RuntimeError(
                f"Requested price column {self.resolved_price_col_!r} is not "
                f"available (it may have failed the quality screen: "
                f"{self.rejected_.get(self.resolved_price_col_, 'not present')})."
            )

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
        # Attach the resolved price column for downstream use.
        out["_price"] = self._price(raw)
        return out

    def coverage(self, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Per-feature coverage report: n non-null, first valid date, %."""
        if features is None:
            features = self.build()
        feats = features.drop(columns="_price", errors="ignore")
        notna = feats.notna()
        return pd.DataFrame({
            "n_obs": notna.sum(),
            "first_valid": notna.apply(lambda c: c[c].index.min() if c.any() else pd.NaT),
            "pct": (100 * notna.mean()).round(1),
        }).sort_values("pct")
