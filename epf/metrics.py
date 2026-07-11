"""
Evaluation metrics and the Diebold-Mariano test.

Follows the recommendations of Lago et al. (2021):
- MAE, RMSE for level errors
- rMAE (relative MAE vs the naive similar-day benchmark) instead of MAPE,
  because MAPE explodes with prices near zero / negative
- sMAPE reported for comparability with older papers
- multivariate DM test on daily loss differentials (one observation per day,
  loss = mean over the 24 hours) — avoids the intraday autocorrelation problem
  of the "univariate" hourly DM test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def smape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(200 * np.mean(np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-9)))


def rmae(y, yhat, y_naive):
    return mae(y, yhat) / mae(y, y_naive)


def pinball_loss(y, q_lo, q_hi, alpha=0.1):
    """Average pinball loss of the two conformal bounds (alpha/2, 1-alpha/2)."""
    y = np.asarray(y)
    lo, hi = np.asarray(q_lo), np.asarray(q_hi)
    a = alpha / 2
    pl_lo = np.mean(np.maximum(a * (y - lo), (a - 1) * (y - lo)))
    pl_hi = np.mean(np.maximum((1 - a) * (y - hi), -a * (y - hi)))
    return float((pl_lo + pl_hi) / 2)


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def dm_test(y: np.ndarray, f1: np.ndarray, f2: np.ndarray, power: int = 1):
    """
    Multivariate Diebold-Mariano test.
    y, f1, f2: arrays of shape (n_days, 24).
    H0: equal accuracy. Returns (stat, p_value); stat < 0 -> model 1 better.
    """
    l1 = np.mean(np.abs(y - f1) ** power, axis=1)
    l2 = np.mean(np.abs(y - f2) ** power, axis=1)
    d = l1 - l2
    n = len(d)
    dbar = d.mean()
    var = d.var(ddof=1) / n
    if var <= 0:
        return 0.0, 1.0
    stat = dbar / np.sqrt(var)
    p = 2 * (1 - stats.t.cdf(np.abs(stat), df=n - 1))
    return float(stat), float(p)
