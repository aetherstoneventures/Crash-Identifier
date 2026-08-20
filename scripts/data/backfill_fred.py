"""Repair and extend the `indicators` table from FRED — point-in-time correct.

WHY THIS SCRIPT EXISTS
======================
The v6.0.0-alpha post-mortem (`docs/V6_POSTMORTEM.md`) traced the engine's
0-recall failure to the *data layer*, not the models. Three defects:

1. **The price column had a 10-year history.** `sp500_close` was sourced
   from FRED's `SP500` series, which is licensed as a **rolling 10-year
   window** (today: 2016-08-22 → now). Every price-derived feature —
   realised vol, drawdown, moving averages, skew, the crash labels
   themselves — therefore existed only from 2016. FRED's `NASDAQCOM`
   carries 1971→now, and the Nasdaq Composite is this project's stated
   target index. We use it as the primary price series.

2. **Monthly macro was stamped at the observation date, not the release
   date.** The April-2020 unemployment rate (14.7%) sat on 2020-04-01 in
   the DB; it was actually published 2020-05-08. Any model reading that
   row "knew" the pandemic unemployment shock five weeks early. Worse,
   the stored value was the *revised* 14.8, which was not knowable at any
   time in 2020. We fix both by pulling ALFRED **vintages** and stamping
   each observation on its **first release date**.

3. **Fabricated columns.** `vix_close` was the constant 17.24 for every
   pre-1990 row; `put_call_ratio` is noise centred on 1.0 (std 0.033,
   lag-1 autocorrelation 0.09, and no spike on Black Monday). Real data
   overwrites the first; the second is quarantined in
   `src/v6/features/quality.py`.

WHAT IT DOES
============
Fetches each series below, aligns it to the trading calendar with an
explicit point-in-time rule, and writes it into `data/market_crash.db`.
Idempotent: re-running overwrites the same columns and prints a
before/after coverage report.

POINT-IN-TIME RULES
===================
- ``pit="close"``    — daily market data, known at that day's close (lag 0).
- ``pit="vintage"``  — use ALFRED first-release dates. Before ALFRED's
  vintage epoch (1997-02-04) no vintages exist, so we fall back to the
  series' own median observed release lag.
- ``pit="lag"``      — fixed publication lag in calendar days, for weekly
  series whose release schedule is stable (NFCI: Wednesday for the prior
  Friday week-ending; ICSA: Thursday for the prior Saturday).

Every series is forward-filled *after* being shifted to its release date,
so a given trading day only ever sees numbers a human could have read
that morning.

Usage
-----
    python scripts/data/backfill_fred.py                # all series
    python scripts/data/backfill_fred.py --only NASDAQCOM,NFCI
    python scripts/data/backfill_fred.py --dry-run      # report only
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "market_crash.db"

# ALFRED only carries vintages from this date. Observations older than this
# all report the same realtime_start, so their "lag" is meaningless.
ALFRED_EPOCH = pd.Timestamp("1997-02-04")


@dataclass(frozen=True)
class Series:
    """One FRED series mapped onto one `indicators` column."""
    column: str          # destination column in the indicators table
    fred_id: str         # FRED series id
    pit: str             # 'close' | 'vintage' | 'lag'
    lag_days: int = 0    # used when pit == 'lag', and as vintage fallback
    note: str = ""

    @property
    def max_staleness_days(self) -> int:
        """How long one published value may stand in for later trading days.

        A value is carried forward only while it is still the latest thing a
        reader could know. Past this window the series is treated as absent
        rather than stale — otherwise a discontinued series (VXO ends 2021)
        would forward-fill its final print across every later row and look
        like a long constant, which is exactly the fabricated-data pattern
        ``quality.py`` rejects.
        """
        return {"close": 7, "lag": 14, "vintage": 45}.get(self.pit, 7)


# ---------------------------------------------------------------------------
# Series catalogue
# ---------------------------------------------------------------------------
SERIES: List[Series] = [
    # --- Equity indices -----------------------------------------------------
    Series("nasdaq_close", "NASDAQCOM", "close", note="PRIMARY price series, 1971+"),
    Series("sp500_close", "SP500", "close", note="rolling 10y window — secondary only"),

    # --- Volatility ---------------------------------------------------------
    Series("vix_close", "VIXCLS", "close", note="1990+; overwrites constant-17.24 fill"),
    Series("vxo_close", "VXOCLS", "close", note="1986-2021, splices VIX back to 1986"),
    Series("vxv_close", "VXVCLS", "close", note="2007+, enables real VIX term structure"),

    # --- Rates and curve ----------------------------------------------------
    Series("yield_10y", "DGS10", "close"),
    Series("yield_10y_3m", "T10Y3M", "close"),
    Series("yield_10y_2y", "T10Y2Y", "close"),
    Series("t10yie", "T10YIE", "close", note="2003+ breakeven inflation"),
    Series("dfii10", "DFII10", "close", note="2003+ 10y TIPS real rate"),

    # --- Credit -------------------------------------------------------------
    Series("baa_10y_spread", "BAA10Y", "close", note="1986+ long credit proxy"),
    Series("aaa_10y_spread", "AAA10Y", "close", note="1983+ IG proxy"),

    # --- Financial conditions (weekly, Wednesday release for Friday obs) ----
    Series("nfci", "NFCI", "lag", 5, "Chicago Fed NFCI, 1971+"),
    Series("anfci", "ANFCI", "lag", 5, "adjusted NFCI"),
    Series("nfci_leverage", "NFCILEVERAGE", "lag", 5),
    Series("nfci_risk", "NFCIRISK", "lag", 5),
    Series("nfci_credit", "NFCICREDIT", "lag", 5),
    Series("stlfsi", "STLFSI4", "lag", 5, "St. Louis Fed financial stress"),

    # --- Labour (weekly, Thursday release for prior Saturday obs) -----------
    Series("initial_claims", "ICSA", "lag", 5, "1967+ weekly jobless claims"),

    # --- Commodities / dollar ----------------------------------------------
    Series("oil_wti", "DCOILWTICO", "close"),
    Series("dollar_twi", "DTWEXBGS", "close", note="2006+; spliced with DTWEXM below"),

    # --- Monthly macro — TRUE POINT-IN-TIME via ALFRED vintages -------------
    Series("unemployment_rate", "UNRATE", "vintage", 35),
    Series("cpi", "CPIAUCSL", "vintage", 45),
    Series("m2_money_supply", "M2SL", "vintage", 45),
    Series("industrial_production", "INDPRO", "vintage", 45),
    Series("umcsent", "UMCSENT", "vintage", 15),
    Series("kcfsi", "KCFSI", "vintage", 30),
]

# Spliced series: destination column <- (primary, historical extension).
# The historical leg is level-shifted to match the primary at the overlap so
# the joined series has no artificial jump.
SPLICES: Dict[str, tuple] = {
    "dollar_twi": ("DTWEXBGS", "DTWEXM"),
}


# ---------------------------------------------------------------------------
# FRED access
# ---------------------------------------------------------------------------
def _get_fred():
    from dotenv import load_dotenv
    load_dotenv(str(PROJECT_ROOT / ".env"))
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not set. Add it to .env (see .env.example)."
        )
    from fredapi import Fred
    return Fred(api_key=key)


def fetch_plain(fred, fred_id: str) -> pd.Series:
    """Latest-vintage observations, indexed by observation date."""
    s = fred.get_series(fred_id).dropna()
    s.index = pd.to_datetime(s.index)
    return s.astype(float).sort_index()


def fetch_first_release(fred, fred_id: str, fallback_lag_days: int) -> pd.Series:
    """First-release values indexed by the date they became public.

    This is the honest point-in-time series: value as originally published,
    stamped on its publication date.

    Two eras need different handling:

    - **Vintage era.** ALFRED carries real release dates, so each value is
      stamped on the day it was actually published.
    - **Pre-vintage era.** ALFRED's vintage coverage starts later than the
      observation history for some series (UMCSENT vintages begin 1998
      though the series runs from 1952). For those older observations we
      fall back to the latest-vintage values shifted by the series' own
      median observed release lag. Those figures are revised rather than
      first-print, which we accept in exchange for keeping 40+ extra years
      of history; the alternative is discarding them entirely.
    """
    raw = fred.get_series_all_releases(fred_id)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["realtime_start"] = pd.to_datetime(raw["realtime_start"])
    first = (
        raw.sort_values("realtime_start")
        .groupby("date", as_index=False)
        .first()
        .dropna(subset=["value"])
    )
    if first.empty:
        return pd.Series(dtype=float)

    observed_lag = (first["realtime_start"] - first["date"]).dt.days
    # Only observations released after the ALFRED epoch have a real vintage.
    genuine = first["realtime_start"] > ALFRED_EPOCH
    median_lag = (
        float(observed_lag[genuine].median())
        if genuine.any() and np.isfinite(observed_lag[genuine].median())
        else float(fallback_lag_days)
    )

    known_on = first["realtime_start"].where(
        genuine, first["date"] + pd.to_timedelta(median_lag, unit="D")
    )
    out = pd.Series(
        pd.to_numeric(first["value"], errors="coerce").values, index=known_on
    ).dropna()
    # Two observations can share a release date (e.g. a catch-up publication);
    # keep the most recent observation for that date.
    out = out.groupby(level=0).last().sort_index()

    # Extend backwards with lag-shifted latest-vintage data where ALFRED has
    # no vintages at all, so we don't throw away decades of observations.
    try:
        plain = fetch_plain(fred, fred_id)
    except Exception:  # noqa: BLE001 — extension is best-effort
        return out
    if out.empty:
        plain.index = plain.index + pd.to_timedelta(fallback_lag_days, unit="D")
        return plain.groupby(level=0).last().sort_index()

    lag = pd.to_timedelta(median_lag, unit="D")
    earlier = plain.loc[plain.index + lag < out.index.min()]
    if earlier.empty:
        return out
    earlier = earlier.copy()
    earlier.index = earlier.index + lag
    return pd.concat([earlier, out]).groupby(level=0).last().sort_index()


def _splice(primary: pd.Series, historical: pd.Series) -> pd.Series:
    """Extend `primary` backwards with `historical`, level-matched at overlap.

    The historical leg is multiplied by the ratio of the two series over
    their common dates, so the joined series is continuous in level and
    identical to `primary` wherever `primary` exists.
    """
    overlap = primary.index.intersection(historical.index)
    if len(overlap) >= 20:
        ratio = float((primary.loc[overlap] / historical.loc[overlap]).median())
    else:
        ratio = 1.0
    tail = historical.loc[historical.index < primary.index.min()] * ratio
    return pd.concat([tail, primary]).sort_index()


# ---------------------------------------------------------------------------
# Calendar + DB helpers
# ---------------------------------------------------------------------------
def build_calendar(con: sqlite3.Connection, price: pd.Series) -> pd.DatetimeIndex:
    """Trading calendar = existing DB dates ∪ price-series observation dates.

    The price series (Nasdaq Composite) only prints on trading days, so it
    defines the calendar going forward and extends the table to the latest
    available data.
    """
    existing = pd.read_sql_query("SELECT date FROM indicators", con)
    existing_idx = pd.DatetimeIndex(pd.to_datetime(existing["date"]))
    return existing_idx.union(pd.DatetimeIndex(price.index)).sort_values()


def align(series: pd.Series, calendar: pd.DatetimeIndex, pit: str,
          lag_days: int, max_staleness_days: int = 7) -> pd.Series:
    """Place a series on the trading calendar under its point-in-time rule.

    Each trading day inherits the latest value published on or before it and
    nothing later. A value goes stale after `max_staleness_days`, so gaps and
    discontinued series become NaN instead of a forward-filled constant.
    """
    s = series.sort_index()
    if pit == "lag":
        s.index = s.index + pd.to_timedelta(lag_days, unit="D")
    # 'vintage' is already stamped on its release date; 'close' needs no shift.

    # Collapse duplicates that a shift can create.
    s = s.groupby(level=0).last()

    combined_index = s.index.union(calendar)
    values = s.reindex(combined_index).ffill()
    # Age of the value standing on each date; NaN out anything too old.
    stamped = pd.Series(s.index, index=s.index).reindex(combined_index).ffill()
    age_days = (combined_index.to_series() - stamped).dt.days
    values = values.where(age_days <= max_staleness_days)
    return values.reindex(calendar)


def ensure_columns(con: sqlite3.Connection, columns: List[str]) -> List[str]:
    """ALTER TABLE ADD COLUMN for any destination column that doesn't exist."""
    existing = {r[1] for r in con.execute("PRAGMA table_info(indicators)")}
    added = []
    for col in columns:
        if col not in existing:
            con.execute(f'ALTER TABLE indicators ADD COLUMN "{col}" REAL')
            added.append(col)
    return added


def ensure_rows(con: sqlite3.Connection, calendar: pd.DatetimeIndex) -> int:
    """Insert rows for calendar dates the table doesn't have yet."""
    existing = pd.read_sql_query("SELECT date FROM indicators", con)
    have = set(pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d"))
    want = [d.strftime("%Y-%m-%d") for d in calendar]
    missing = [d for d in want if d not in have]
    if missing:
        con.executemany(
            "INSERT INTO indicators (date) VALUES (?)", [(d,) for d in missing]
        )
    return len(missing)


def write_column(con: sqlite3.Connection, column: str, values: pd.Series) -> int:
    """Overwrite one column for every date in `values` (NaN -> NULL)."""
    payload = [
        (None if pd.isna(v) else float(v), d.strftime("%Y-%m-%d"))
        for d, v in values.items()
    ]
    con.executemany(
        f'UPDATE indicators SET "{column}" = ? WHERE date = ?', payload
    )
    return int(values.notna().sum())


def coverage_report(con: sqlite3.Connection, columns: List[str]) -> pd.DataFrame:
    """Non-null count and first/last valid date per column."""
    rows = []
    for col in columns:
        q = (
            f'SELECT COUNT("{col}") n, MIN(date) lo, MAX(date) hi '
            f'FROM indicators WHERE "{col}" IS NOT NULL'
        )
        try:
            n, lo, hi = con.execute(q).fetchone()
        except sqlite3.OperationalError:
            n, lo, hi = 0, None, None
        rows.append({"column": col, "n": n or 0, "first": lo, "last": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated FRED ids to refresh (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report, but do not write to the DB")
    ap.add_argument("--db", type=str, default=str(DB_PATH))
    args = ap.parse_args()

    wanted = set(s.strip().upper() for s in args.only.split(",")) if args.only else None
    todo = [s for s in SERIES if wanted is None or s.fred_id.upper() in wanted]
    if not todo:
        print("Nothing to do — no series matched --only.")
        return 1

    print(f"FRED backfill — {len(todo)} series -> {args.db}")
    fred = _get_fred()

    # 1. Fetch everything first, so a mid-run network failure leaves the DB
    #    untouched rather than half-updated.
    fetched: Dict[str, pd.Series] = {}
    for spec in todo:
        try:
            if spec.pit == "vintage":
                s = fetch_first_release(fred, spec.fred_id, spec.lag_days)
                kind = "vintage(first-release)"
            else:
                s = fetch_plain(fred, spec.fred_id)
                kind = f"{spec.pit}(lag={spec.lag_days}d)" if spec.pit == "lag" else "close"
            if s.empty:
                print(f"  {spec.fred_id:14s} EMPTY — skipped")
                continue
            fetched[spec.column] = s
            print(f"  {spec.fred_id:14s} -> {spec.column:22s} n={len(s):6d} "
                  f"{s.index.min().date()}..{s.index.max().date()}  [{kind}]")
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  {spec.fred_id:14s} FAILED: {type(exc).__name__}: {exc}")

    if not fetched:
        print("No series fetched. Aborting without touching the DB.")
        return 1

    # 2. Splices — extend a short modern series with a discontinued long one.
    for column, (primary_id, hist_id) in SPLICES.items():
        if column not in fetched:
            continue
        try:
            hist = fetch_plain(fred, hist_id)
            before = fetched[column].index.min().date()
            fetched[column] = _splice(fetched[column], hist)
            print(f"  spliced {column}: {primary_id} + {hist_id} "
                  f"-> history now starts {fetched[column].index.min().date()} "
                  f"(was {before})")
        except Exception as exc:  # noqa: BLE001
            print(f"  splice {column} FAILED: {type(exc).__name__}: {exc}")

    # 3. VIX back-extension: VIXCLS starts 1990, VXOCLS starts 1986.
    if "vix_close" in fetched and "vxo_close" in fetched:
        fetched["vix_close"] = _splice(fetched["vix_close"], fetched["vxo_close"])
        print(f"  spliced vix_close: VIXCLS + VXOCLS -> starts "
              f"{fetched['vix_close'].index.min().date()}")

    if args.dry_run:
        print("\n--dry-run: no database writes performed.")
        return 0

    # 4. Write.
    con = sqlite3.connect(args.db)
    try:
        price = fetched.get("nasdaq_close")
        if price is None:
            price = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        calendar = build_calendar(con, price)

        added = ensure_columns(con, list(fetched.keys()))
        if added:
            print(f"\nAdded columns: {', '.join(added)}")
        new_rows = ensure_rows(con, calendar)
        if new_rows:
            print(f"Inserted {new_rows} new calendar rows "
                  f"(table now spans to {calendar.max().date()})")

        before = coverage_report(con, list(fetched.keys())).set_index("column")

        print("\nWriting columns...")
        for column, series in fetched.items():
            spec = next(s for s in todo if s.column == column)
            aligned = align(
                series, calendar, pit=spec.pit, lag_days=spec.lag_days,
                max_staleness_days=spec.max_staleness_days,
            )
            n = write_column(con, column, aligned)
            print(f"  {column:22s} {n:6d} non-null rows on calendar")
        con.commit()

        after = coverage_report(con, list(fetched.keys())).set_index("column")
        delta = pd.DataFrame({
            "n_before": before["n"],
            "n_after": after["n"],
            "first_before": before["first"],
            "first_after": after["first"],
        })
        delta["gained"] = delta["n_after"] - delta["n_before"]
        print("\n=== Coverage change ===")
        print(delta.to_string())
    finally:
        con.close()

    print("\nDone. Re-run the v6 pipeline to pick up the repaired data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
