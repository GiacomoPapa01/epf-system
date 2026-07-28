"""
Walk-forward backtest with rolling recalibration and split-conformal intervals.

Protocol (standard in the EPF literature, no look-ahead by construction):
- calibration window: last `cal_days` days before the target day (rolling)
- recalibration every `recal_every` days (1 = daily, the gold standard;
  7 keeps runtime low with ~equal accuracy for tree models)
- forecasts are truly out-of-sample: features for day D only use data
  available before D (enforced in features.build_design_matrix via shifts)

Uncertainty: split-conformal per hour — the empirical (1-alpha) quantile of
|residuals| over the last `conf_days` out-of-sample days is added/subtracted
to the point forecast. Distribution-free finite-sample coverage guarantee
under exchangeability; rolling window adapts to volatility regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import coverage, dm_test, mae, pinball_loss, rmae, rmse, smape
from .models import GBT, LEAR, Ensemble, LEARWindows, NaiveDaily


@dataclass
class BacktestResult:
    forecasts: dict[str, pd.DataFrame]  # model -> (days x 24) forecasts
    actuals: pd.DataFrame
    intervals: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)

    def metrics(self) -> pd.DataFrame:
        y = self.actuals.values
        naive = self.forecasts["naive"].values
        rows = {}
        for name, f in self.forecasts.items():
            fh = f.values
            rows[name] = {
                "MAE": mae(y, fh),
                "RMSE": rmse(y, fh),
                "rMAE": rmae(y, fh, naive),
                "sMAPE%": smape(y, fh),
            }
            if name in self.intervals:
                lo, hi = self.intervals[name]
                rows[name]["cov90%"] = 100 * coverage(y, lo.values, hi.values)
                rows[name]["pinball"] = pinball_loss(y, lo.values, hi.values)
        return pd.DataFrame(rows).T.round(3)

    def dm_matrix(self) -> pd.DataFrame:
        names = list(self.forecasts)
        y = self.actuals.values
        out = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            for b in names:
                if a != b:
                    _, p = dm_test(y, self.forecasts[a].values, self.forecasts[b].values)
                    out.loc[a, b] = round(p, 4)
        return out


def run_backtest(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    panel: pd.DataFrame,
    cal_days: int = 730,
    test_days: int = 180,
    recal_every: int = 1,
    alpha: float = 0.10,
    conf_days: int = 90,
    models: tuple[str, ...] = ("lear", "gbt", "ensemble"),
    verbose: bool = True,
) -> BacktestResult:
    dates = X.index
    if len(dates) < cal_days + test_days:
        raise ValueError(
            f"Need >= {cal_days + test_days} days, have {len(dates)}. "
            "Reduce cal_days/test_days or load more data."
        )
    test_dates = dates[-test_days:]

    need_lear = "lear" in models or "ensemble" in models
    need_gbt = "gbt" in models or "ensemble" in models
    preds: dict[str, list[np.ndarray]] = {m: [] for m in models}
    naive_model = NaiveDaily()
    fitted: dict[str, object] = {}

    # One iteration per recalibration block: models are fitted on data strictly
    # before the first day of the block, then predict the whole block at once
    # (identical to per-day prediction with the same fitted models, just faster).
    for i in range(0, len(test_dates), recal_every):
        block = test_dates[i : i + recal_every]
        pos = dates.get_loc(block[0])
        tr = dates[max(0, pos - cal_days) : pos]
        if need_lear:
            fitted["lear"] = LEAR().fit(X.loc[tr], Y.loc[tr])
        if "learw" in models:
            fitted["learw"] = LEARWindows().fit(X.loc[tr], Y.loc[tr])
        if need_gbt:
            fitted["gbt"] = GBT().fit(X.loc[tr], Y.loc[tr])
        if verbose and (i // recal_every) % 10 == 0:
            print(f"  [{i + 1}/{test_days}] recalibrated @ {block[0].date()}")

        Xb = X.loc[block]
        p_lear = fitted["lear"].predict(Xb) if need_lear else None
        p_gbt = fitted["gbt"].predict(Xb) if need_gbt else None
        if "lear" in models:
            preds["lear"].append(p_lear)
        if "learw" in models:
            preds["learw"].append(fitted["learw"].predict(Xb))
        if "gbt" in models:
            preds["gbt"].append(p_gbt)
        if "ensemble" in models:
            preds["ensemble"].append(Ensemble.combine(p_lear, p_gbt))

    cols = [f"h{h}" for h in range(24)]
    actuals = pd.DataFrame(Y.loc[test_dates].values, index=test_dates, columns=cols)
    forecasts = {
        "naive": pd.DataFrame(
            naive_model.predict_from_panel(panel, test_dates), index=test_dates, columns=cols
        )
    }
    for m in models:
        forecasts[m] = pd.DataFrame(np.vstack(preds[m]), index=test_dates, columns=cols)

    intervals = {
        m: _conformal(forecasts[m], actuals, alpha=alpha, window=conf_days)
        for m in models
    }
    return BacktestResult(forecasts=forecasts, actuals=actuals, intervals=intervals)


def _conformal(fcst: pd.DataFrame, act: pd.DataFrame, alpha: float, window: int):
    """
    Rolling split-conformal per hour, ASYMMETRIC: lower/upper bounds from the
    alpha/2 and 1-alpha/2 rolling quantiles of the signed out-of-sample
    residuals. Power prices are right-skewed (spikes), so symmetric |resid|
    bands over-cover on the left and under-cover on the right; asymmetric
    bands fix both.
    """
    resid = act - fcst  # signed
    mp = max(20, window // 3)
    # shift(1): the band for day D only uses residuals up to D-1. Days without
    # enough history get NaN bands (excluded from coverage/pinball) rather than
    # backfilled ones, which would leak future residuals into the early test.
    q_lo = resid.rolling(window, min_periods=mp).quantile(alpha / 2).shift(1)
    q_hi = resid.rolling(window, min_periods=mp).quantile(1 - alpha / 2).shift(1)
    return fcst + q_lo, fcst + q_hi
