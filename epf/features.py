"""
Feature engineering for day-ahead EPF.

Design follows the LEAR feature set of Lago, Marcjasz, De Schutter, Weron
(2021, Applied Energy) — the strongest publicly benchmarked linear setup —
extended with calendar features and residual load.

Day-ahead framing: at ~11:00 D-1 (before EPEX gate closure at 12:00) we forecast
the 24 hourly prices of day D. Available information:
  - prices up to day D-1 (lags 1d, 2d, 3d, 7d, each 24 hourly values)
  - day-ahead exogenous forecasts for day D (load_fc, wind_fc, solar_fc,
    res_load_fc) and their values for D-1 and D-7
  - calendar of day D

The dataset is reshaped to "one row per (day, hour)" with hour-specific models,
or "wide" 24-target format, depending on the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LAGS_DAYS = [1, 2, 3, 7]
EXOG_LAG_DAYS = [0, 1, 7]  # 0 = day-ahead forecast for target day itself


def build_daily_panel(
    df: pd.DataFrame, exog_cols: list[str], max_missing_h: int = 2, verbose: bool = True
) -> pd.DataFrame:
    """
    Pivot hourly series to daily rows: index = date, columns = f"{var}_h{H}".
    Days with up to `max_missing_h` missing hours (e.g. DST spring-forward in
    local-time data) are repaired by within-day linear interpolation; days
    missing more hours are dropped and reported — never silently.
    """
    x = df.copy()
    x["date"] = x.index.tz_convert("UTC").date
    x["hour"] = x.index.tz_convert("UTC").hour

    panels = []
    for col in ["price"] + exog_cols:
        if col not in x.columns:
            continue
        p = x.pivot_table(index="date", columns="hour", values=col)
        p = p.reindex(columns=range(24))
        p = p.interpolate(axis=1, limit=max_missing_h, limit_direction="both")
        p.columns = [f"{col}_h{h}" for h in p.columns]
        panels.append(p)
    panel = pd.concat(panels, axis=1)
    panel.index = pd.DatetimeIndex(panel.index)

    bad = panel.isna().any(axis=1)
    if bad.any() and verbose:
        print(f"build_daily_panel: dropped {int(bad.sum())} incomplete days "
              f"(>{max_missing_h} missing hours): {list(panel.index[bad].date)[:5]}...")
    return panel[~bad].sort_index()


def build_design_matrix(
    panel: pd.DataFrame, exog_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (X, Y):
      Y: (n_days, 24) target prices of day D
      X: features known at D-1 11:00
    """
    feats = {}
    # lagged prices (all 24 hours of each lagged day)
    for lag in PRICE_LAGS_DAYS:
        for h in range(24):
            feats[f"price_l{lag}_h{h}"] = panel[f"price_h{h}"].shift(lag)
    # exogenous day-ahead forecasts for D, D-1, D-7
    for col in exog_cols:
        for lag in EXOG_LAG_DAYS:
            for h in range(24):
                key = f"{col}_h{h}"
                if key in panel.columns:
                    feats[f"{col}_l{lag}_h{h}"] = panel[key].shift(lag)
    X = pd.DataFrame(feats, index=panel.index)

    # calendar
    dow = pd.get_dummies(X.index.dayofweek, prefix="dow").astype(float)
    dow.index = X.index
    X = pd.concat([X, dow], axis=1)
    X["doy_sin"] = np.sin(2 * np.pi * X.index.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * X.index.dayofyear / 365.25)

    Y = panel[[f"price_h{h}" for h in range(24)]]
    valid = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return X[valid], Y[valid]


# --- robust scaling (asinh + median/MAD), as in Uniejewski et al. -------------
class AsinhScaler:
    """Median/MAD standardization followed by asinh. Robust to spikes and
    handles negative prices natively (unlike log)."""

    def fit(self, a: np.ndarray):
        self.med_ = np.nanmedian(a, axis=0)
        mad = np.nanmedian(np.abs(a - self.med_), axis=0)
        self.mad_ = np.where(mad < 1e-8, 1.0, mad / 0.6745)
        return self

    def transform(self, a):
        return np.arcsinh((a - self.med_) / self.mad_)

    def inverse_transform(self, a):
        return np.sinh(a) * self.mad_ + self.med_

    def fit_transform(self, a):
        return self.fit(a).transform(a)
