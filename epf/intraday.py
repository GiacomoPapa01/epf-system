"""
Intraday layer: forecast the DA -> ID spread from residual-load surprises.

Economic logic
--------------
The intraday market re-prices the day-ahead auction as new information arrives.
The single most important driver is the *residual load surprise*:

    surprise(h) = res_load_act(h) - res_load_fc(h)
                = (load error) - (wind error) - (solar error)

More residual load than forecast (wind/solar under-delivered or demand higher)
=> ID trades above DA; the opposite pushes ID below DA, down to negative prices.

Target
------
    spread(h) = ID_price(h) - DA_price(h)      e.g. ID3 or VWAP as ID reference

If you have EPEX intraday data (ID3/ID1 indices), pass it as `id_price`.
If not, the module can *train and validate the machinery* on a proxy spread
built from realized surprises (`make_proxy_spread`) — useful to develop and
backtest the full pipeline, NOT a substitute for real ID data in production.

Model
-----
Gradient boosting with quantile objectives (q10/q50/q90) -> point forecast +
distribution-aware bands, because the spread is heavy-tailed and asymmetric.
Evaluated walk-forward like the DA layer. Features only use information
available at trading time T-30min before delivery (configurable "knowledge
lag"): surprises are lagged by `known_lag_h` hours to mimic what a trader
actually observes (latest published actuals), never the concurrent hour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False

from .metrics import coverage, mae, rmse


def make_proxy_spread(df: pd.DataFrame, beta: float = 0.008, noise: float = 3.0,
                      seed: int = 11) -> pd.Series:
    """Synthetic ID-DA spread ~ beta * surprise + heavy-tailed noise (demo only)."""
    rng = np.random.default_rng(seed)
    surprise = (df["res_load_act"] - df["res_load_fc"]).fillna(0)
    noise_t = noise * rng.standard_t(df=4, size=len(df))
    convex = 1 + 0.6 * (df["res_load_fc"] > df["res_load_fc"].quantile(0.85))
    return beta * surprise * convex + noise_t


def build_intraday_features(df: pd.DataFrame, known_lag_h: int = 2) -> pd.DataFrame:
    """Features available shortly before delivery of hour h."""
    f = pd.DataFrame(index=df.index)
    surprise = df["res_load_act"] - df["res_load_fc"]
    # latest OBSERVED surprises (published with delay) - never hour h itself
    for lag in [known_lag_h, known_lag_h + 1, known_lag_h + 2, known_lag_h + 22]:
        f[f"surprise_l{lag}"] = surprise.shift(lag)
    f["surprise_ma6"] = surprise.shift(known_lag_h).rolling(6).mean()
    f["da_price"] = df["price"]
    f["da_price_ramp"] = df["price"].diff()
    f["res_load_fc"] = df["res_load_fc"]
    f["res_load_fc_ramp"] = df["res_load_fc"].diff()
    f["wind_fc"] = df.get("wind_fc")
    f["solar_fc"] = df.get("solar_fc")
    f["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    f["dow"] = df.index.dayofweek
    return f


def _qmodel(q: float):
    if _HAS_LGBM:
        return LGBMRegressor(objective="quantile", alpha=q, n_estimators=400,
                             learning_rate=0.05, num_leaves=63, verbosity=-1,
                             random_state=0)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                         max_iter=300, learning_rate=0.06,
                                         max_depth=6, random_state=0)


def backtest_intraday(
    features: pd.DataFrame,
    spread: pd.Series,
    cal_hours: int = 24 * 365,
    test_hours: int = 24 * 90,
    recal_every_h: int = 24 * 7,
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
    verbose: bool = True,
) -> pd.DataFrame:
    data = features.copy()
    data["y"] = spread
    data = data.dropna()
    if len(data) < cal_hours + test_hours:
        raise ValueError("Not enough hourly data for the requested windows.")
    test_idx = data.index[-test_hours:]

    out = pd.DataFrame(index=test_idx, columns=["q10", "q50", "q90", "y"], dtype=float)
    out["y"] = data.loc[test_idx, "y"]
    xcols = [c for c in data.columns if c != "y"]

    # One iteration per recalibration block: fit on data strictly before the
    # block, predict the whole block at once (same forecasts, far fewer calls).
    for i in range(0, len(test_idx), recal_every_h):
        block = test_idx[i : i + recal_every_h]
        pos = data.index.get_loc(block[0])
        tr = data.iloc[max(0, pos - cal_hours) : pos]
        if verbose:
            print(f"  intraday recalibration @ {block[0]} ({i}/{test_hours})")
        Xb = data.loc[block, xcols]
        for q, col in zip(quantiles, ["q10", "q50", "q90"]):
            out.loc[block, col] = _qmodel(q).fit(tr[xcols], tr["y"]).predict(Xb)

    # enforce quantile monotonicity
    out["q10"], out["q90"] = np.minimum(out["q10"], out["q90"]), np.maximum(out["q10"], out["q90"])

    # CQR correction (Romano et al. 2019): rolling conformal adjustment of the
    # bands using past out-of-sample conformity scores E = max(q10-y, y-q90).
    alpha = 1 - (quantiles[2] - quantiles[0])  # e.g. 0.2 for (0.1, 0.9)
    E = np.maximum(out["q10"] - out["y"], out["y"] - out["q90"])
    qE = E.rolling(24 * 30, min_periods=24 * 7).quantile(1 - alpha).shift(1)
    # no correction until enough past scores exist — backfilling would use
    # future conformity scores (look-ahead) on the early test hours
    qE = qE.fillna(0)
    out["q10"] -= qE
    out["q90"] += qE
    return out


def intraday_metrics(bt: pd.DataFrame) -> dict:
    y, q50 = bt["y"].values, bt["q50"].values
    naive = np.zeros_like(y)  # naive: spread = 0 (ID = DA)
    return {
        "MAE": round(mae(y, q50), 3),
        "MAE_naive(ID=DA)": round(mae(y, naive), 3),
        "rMAE": round(mae(y, q50) / mae(y, naive), 3),
        "RMSE": round(rmse(y, q50), 3),
        "cov80%": round(100 * coverage(y, bt["q10"].values, bt["q90"].values), 1),
        "directional_acc%": round(100 * np.mean(np.sign(q50) == np.sign(y)), 1),
    }
