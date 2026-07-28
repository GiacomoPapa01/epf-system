"""
Forecasting models. All expose fit(X, Y) / predict(X) with Y of shape (n, 24).

Models
------
NaiveDaily      similar-day benchmark: price(D) = price(D-1) [Mon/Sat/Sun -> D-7].
                Mandatory baseline: rMAE is computed against it.
LEAR            Lasso Estimated AutoRegressive (Lago et al. 2021): one Lasso per
                hour on asinh-scaled data, lambda chosen by AIC (LassoLarsIC).
LEARWindows     Average of LEAR fits over multiple calibration-window lengths
                (Lago et al. / Marcjasz et al.: window averaging is one of the
                most reliable accuracy boosts in EPF — short windows adapt to
                regime shifts, long windows stabilize).
GBT             LightGBM if installed, else sklearn HistGradientBoosting.
                Optional target winsorization for spike robustness.
Ensemble        Simple average of member forecasts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC

from .features import AsinhScaler

try:
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_LGBM = False


class NaiveDaily:
    """price(D) = price(D-1); for Mon, Sat, Sun -> price(D-7)."""

    name = "naive"

    def fit(self, X, Y):
        return self

    def predict_from_panel(self, panel: pd.DataFrame, dates) -> np.ndarray:
        cols = [f"price_h{h}" for h in range(24)]
        out = []
        for d in dates:
            lag = 7 if pd.Timestamp(d).dayofweek in (0, 5, 6) else 1
            ref = d - pd.Timedelta(days=lag)
            if ref in panel.index:
                out.append(panel.loc[ref, cols].values)
            else:  # gap (dropped incomplete day): use last available day before ref
                out.append(panel.loc[:ref, cols].iloc[-1].values)
        return np.asarray(out, dtype=float)


class LEAR:
    name = "lear"

    def __init__(self, criterion: str = "aic", max_iter: int = 2500):
        self.criterion = criterion
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.xscaler_ = AsinhScaler().fit(X.values)
        self.yscaler_ = AsinhScaler().fit(Y.values)
        Xs = self.xscaler_.transform(X.values)
        Ys = self.yscaler_.transform(Y.values)
        self.models_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for h in range(24):
                try:
                    m = LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter)
                    m.fit(Xs, Ys[:, h])
                except Exception:  # numerical issues -> fixed-alpha fallback
                    m = Lasso(alpha=1e-3, max_iter=self.max_iter)
                    m.fit(Xs, Ys[:, h])
                self.models_.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.xscaler_.transform(X.values)
        Ys = np.column_stack([m.predict(Xs) for m in self.models_])
        return self.yscaler_.inverse_transform(Ys)

    def nonzero_share(self) -> float:
        """Diagnostic: average share of features selected by the Lasso."""
        return float(np.mean([np.mean(m.coef_ != 0) for m in self.models_]))


class LEARWindows:
    """LEAR averaged over several calibration-window lengths (days)."""

    name = "learw"

    def __init__(self, windows: tuple[int, ...] = (56, 84, 365, 730)):
        self.windows = windows

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        n = len(X)
        self.members_ = []
        used = set()
        for w in self.windows:
            w_eff = min(w, n)
            if w_eff < 40 or w_eff in used:  # too short / duplicate after clipping
                continue
            used.add(w_eff)
            self.members_.append(LEAR().fit(X.iloc[-w_eff:], Y.iloc[-w_eff:]))
        if not self.members_:
            self.members_ = [LEAR().fit(X, Y)]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.members_], axis=0)


class GBT:
    name = "gbt"

    def __init__(self, winsorize: tuple[float, float] | None = (0.001, 0.999), **kw):
        self.winsorize = winsorize
        self.kw = dict(max_depth=6, learning_rate=0.05, random_state=0)
        if _HAS_LGBM:
            self.kw.update(n_estimators=500, num_leaves=63, subsample=0.8,
                           colsample_bytree=0.8, verbosity=-1)
        else:
            self.kw.update(max_iter=200, max_leaf_nodes=31,
                           l2_regularization=1.0, early_stopping=False)
            self.kw["learning_rate"] = 0.08
        self.kw.update(kw)

    def _new(self):
        return LGBMRegressor(**self.kw) if _HAS_LGBM else HistGradientBoostingRegressor(**self.kw)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        Yv = Y.values
        if self.winsorize:
            lo = np.quantile(Yv, self.winsorize[0])
            hi = np.quantile(Yv, self.winsorize[1])
            Yv = np.clip(Yv, lo, hi)
        # fit/predict both take the DataFrame so feature names stay consistent
        self.models_ = []
        for h in range(24):
            m = self._new()
            m.fit(X, Yv[:, h])
            self.models_.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])


class Ensemble:
    name = "ensemble"

    @staticmethod
    def combine(*preds: np.ndarray) -> np.ndarray:
        return np.mean(np.stack([p for p in preds if p is not None]), axis=0)
