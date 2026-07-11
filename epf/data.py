"""
Data loading for electricity price forecasting.

Three sources, in order of preference:
1. ENTSO-E Transparency API (via entsoe-py) -> richest data, needs free API token.
   Gives: DA prices, DA load forecast, DA wind/solar forecast, actual load,
   actual wind/solar generation. Enough for both day-ahead and intraday models.
2. epftoolbox open datasets (DE, NP, PJM, BE, FR) -> price + 2 exogenous series,
   fully reproducible benchmark data (Lago et al. 2021).
3. Synthetic generator -> for smoke tests / CI, mimics German market stylized
   facts (daily/weekly seasonality, merit-order link to residual load, spikes,
   negative prices).

All loaders return a canonical hourly DataFrame indexed by UTC timestamp with
columns (subset depending on source):
    price            day-ahead auction price [EUR/MWh]
    load_fc          day-ahead load forecast [MW]
    wind_fc          day-ahead wind forecast [MW]
    solar_fc         day-ahead solar forecast [MW]
    load_act         actual load [MW]
    wind_act         actual wind generation [MW]
    solar_act        actual solar generation [MW]
Derived:
    res_load_fc  = load_fc - wind_fc - solar_fc      (day-ahead residual load)
    res_load_act = load_act - wind_act - solar_act   (realized residual load)
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

EPFTOOLBOX_URLS = {
    # Open benchmark datasets from Lago et al. (2021), mirrored on Zenodo.
    m: f"https://zenodo.org/records/4624805/files/{m}.csv"
    for m in ["DE", "NP", "PJM", "BE", "FR"]
}


# ----------------------------------------------------------------------------- 
# 1) ENTSO-E
# -----------------------------------------------------------------------------
def load_entsoe(
    api_key: str,
    country_code: str = "DE_LU",
    start: str = "2019-01-01",
    end: str | None = None,
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Download full dataset from ENTSO-E Transparency (requires entsoe-py)."""
    from entsoe import EntsoePandasClient  # pip install entsoe-py

    client = EntsoePandasClient(api_key=api_key)
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz) if end else pd.Timestamp.now(tz=tz).normalize()

    price = client.query_day_ahead_prices(country_code, start=start_ts, end=end_ts)
    load_fc = client.query_load_forecast(country_code, start=start_ts, end=end_ts)
    load_act = client.query_load(country_code, start=start_ts, end=end_ts)
    ws_fc = client.query_wind_and_solar_forecast(country_code, start=start_ts, end=end_ts)
    gen = client.query_generation(country_code, start=start_ts, end=end_ts, psr_type=None)

    # Since Oct 2025 the DA market trading unit is 15 min: resample to hourly.
    df = pd.DataFrame({"price": _to_hourly(price)})
    df["load_fc"] = _to_hourly(load_fc)
    df["load_act"] = _to_hourly(load_act)
    df["wind_fc"] = _to_hourly(
        ws_fc.filter(like="Wind").sum(axis=1) if hasattr(ws_fc, "filter") else ws_fc
    )
    df["solar_fc"] = _to_hourly(ws_fc["Solar"]) if "Solar" in getattr(ws_fc, "columns", []) else np.nan
    wind_cols = [c for c in gen.columns if "Wind" in str(c)]
    solar_cols = [c for c in gen.columns if "Solar" in str(c)]
    df["wind_act"] = _to_hourly(gen[wind_cols].sum(axis=1)) if wind_cols else np.nan
    df["solar_act"] = _to_hourly(gen[solar_cols].sum(axis=1)) if solar_cols else np.nan

    df.index = df.index.tz_convert("UTC")
    return add_residual_load(df.sort_index())


def _to_hourly(s):
    s = s.squeeze()
    return s.resample("1h").mean()


# -----------------------------------------------------------------------------
# 2) epftoolbox open datasets
# -----------------------------------------------------------------------------
def load_epftoolbox(market: str = "DE", cache_dir: str = "data") -> pd.DataFrame:
    """
    Load one of the 5 open EPF benchmark datasets (6 years hourly each).
    DE: exog1 = load forecast proxy? -> in DE dataset: exog1 = Ampirion zonal load
    forecast, exog2 = wind+solar generation forecast. So residual load is
    directly exog1 - exog2.
    """
    import os

    path = os.path.join(cache_dir, f"{market}.csv")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        with urllib.request.urlopen(EPFTOOLBOX_URLS[market], timeout=60) as r:
            raw = r.read()
        with open(path, "wb") as f:
            f.write(raw)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = ["price", "exog1", "exog2"][: len(df.columns)]
    if market == "DE":
        df = df.rename(columns={"exog1": "load_fc", "exog2": "renewables_fc"})
        df["res_load_fc"] = df["load_fc"] - df["renewables_fc"]
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    return df


# -----------------------------------------------------------------------------
# 3) Synthetic generator (for tests and demos)
# -----------------------------------------------------------------------------
def make_synthetic(n_days: int = 730, seed: int = 7) -> pd.DataFrame:
    """
    Synthetic hourly German-like market. Price is generated from residual load
    through a convex merit-order curve + AR(1) noise + spikes, so a good model
    must exploit the exogenous variables — exactly like reality.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    h = idx.hour.values
    dow = idx.dayofweek.values
    doy = idx.dayofyear.values
    t = np.arange(len(idx))

    daily = 8000 * np.sin((h - 6) / 24 * 2 * np.pi) + 3000 * np.sin((h - 9) / 12 * 2 * np.pi)
    weekly = np.where(dow >= 5, -6000.0, 0.0)
    seasonal = 5000 * np.cos((doy - 15) / 365 * 2 * np.pi)
    load = 55000 + daily + weekly + seasonal + _ar1(rng, len(idx), 0.98, 500)

    solar_shape = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None)
    solar_season = 0.6 + 0.4 * np.cos((doy - 172) / 365 * 2 * np.pi) * -1
    solar = 30000 * solar_shape * solar_season * np.clip(_ar1(rng, len(idx), 0.9, 0.15) + 1, 0.1, None) / 2
    wind = np.clip(15000 + _ar1(rng, len(idx), 0.995, 800), 500, 45000)

    res_load = load - wind - solar
    # convex merit order + AR noise + spikes + regime drift (fuel prices)
    fuel = 1 + 0.4 * np.sin(t / (24 * 90) * 2 * np.pi) + _ar1(rng, len(idx), 0.999, 0.004)
    price = (
        -20
        + 0.0035 * res_load
        + 2.2e-9 * np.clip(res_load, 0, None) ** 2
    ) * fuel + _ar1(rng, len(idx), 0.85, 4)
    spikes = (rng.random(len(idx)) < 0.004) * rng.exponential(80, len(idx))
    price = price + spikes

    df = pd.DataFrame(
        {
            "price": price,
            "load_act": load,
            "wind_act": wind,
            "solar_act": solar,
        },
        index=idx,
    )
    # day-ahead forecasts = actuals + forecast error (persistent within day)
    for v, sd in [("load", 800), ("wind", 1800), ("solar", 900)]:
        err = np.repeat(rng.normal(0, sd, n_days), 24) + rng.normal(0, sd / 3, len(idx))
        df[f"{v}_fc"] = df[f"{v}_act"] + err
    return add_residual_load(df)


def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def add_residual_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"load_fc", "wind_fc", "solar_fc"}.issubset(df.columns):
        df["res_load_fc"] = df["load_fc"] - df["wind_fc"].fillna(0) - df["solar_fc"].fillna(0)
    if {"load_act", "wind_act", "solar_act"}.issubset(df.columns):
        df["res_load_act"] = df["load_act"] - df["wind_act"].fillna(0) - df["solar_act"].fillna(0)
    return df
