"""
Data validation & cleaning. Run this on any real dataset before modelling.

Checks / fixes:
- duplicated timestamps (DST fall-back or bad merges) -> averaged
- missing hours -> reindexed to a complete hourly grid, interpolated up to
  `max_gap_h` consecutive hours (longer gaps are left NaN and reported)
- physically impossible values -> flagged (negative load, solar at night,
  prices outside [-500, 4000] EUR/MWh EPEX limits)
- outlier report via robust z-score (median/MAD), NOT removed by default:
  price spikes are real information in power markets.

Returns (clean_df, report_dict). Nothing is silently discarded: everything
that was touched is counted in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPEX_MIN, EPEX_MAX = -500.0, 4000.0


def validate_hourly(
    df: pd.DataFrame,
    max_gap_h: int = 6,
    price_col: str = "price",
) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    x = df.copy().sort_index()

    # 1) duplicates (e.g. DST fall-back hour delivered twice)
    dup = x.index.duplicated(keep=False).sum()
    report["duplicate_timestamps"] = int(dup)
    if dup:
        x = x.groupby(level=0).mean()

    # 2) complete hourly grid + bounded interpolation
    full = pd.date_range(x.index[0], x.index[-1], freq="1h", tz=x.index.tz)
    missing = len(full) - len(x)
    report["missing_hours"] = int(missing)
    x = x.reindex(full)
    before = x.isna().sum().sum()
    x = x.interpolate(limit=max_gap_h, limit_direction="both")
    report["interpolated_values"] = int(before - x.isna().sum().sum())
    report["unresolved_nans"] = int(x.isna().sum().sum())

    # 3) physical sanity flags
    if price_col in x:
        oob = ((x[price_col] < EPEX_MIN) | (x[price_col] > EPEX_MAX)).sum()
        report["price_out_of_bounds"] = int(oob)
    for c in [c for c in x.columns if "load" in c or "wind" in c or "solar" in c]:
        report[f"negative_{c}"] = int((x[c] < 0).sum())
    for c in [c for c in x.columns if "solar" in c]:
        night = x.index.hour.isin([0, 1, 2, 3])
        report[f"solar_at_night_{c}"] = int((x.loc[night, c] > 50).sum())

    # 4) robust outlier count (informative only)
    if price_col in x:
        p = x[price_col]
        med, mad = p.median(), (p - p.median()).abs().median() / 0.6745
        report["price_robust_z>6"] = int(((p - med).abs() / max(mad, 1e-9) > 6).sum())

    return x, report


def assert_clean(report: dict, strict: bool = False):
    """Raise if the dataset is unusable; warn on soft issues."""
    if report.get("unresolved_nans", 0) > 0:
        msg = f"{report['unresolved_nans']} NaNs remain after interpolation"
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg} — affected days will be dropped by the panel builder.")
    if report.get("price_out_of_bounds", 0) > 0:
        print(f"WARNING: {report['price_out_of_bounds']} prices outside EPEX bounds.")
