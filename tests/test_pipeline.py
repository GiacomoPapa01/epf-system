"""Minimal sanity tests: no look-ahead, shapes, conformal coverage logic."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epf import data, features, metrics


def test_no_lookahead_in_design_matrix():
    df = data.make_synthetic(n_days=60)
    exog = ["load_fc", "res_load_fc"]
    panel = features.build_daily_panel(df, exog)
    X, Y = features.build_design_matrix(panel, exog)
    d = X.index[30]
    # price features must come from strictly earlier days
    assert np.isclose(X.loc[d, "price_l1_h5"], panel.loc[d - pd.Timedelta(days=1), "price_h5"])
    # exog lag 0 is the day-ahead forecast for D itself (known at D-1): allowed
    assert np.isclose(X.loc[d, "load_fc_l0_h12"], panel.loc[d, "load_fc_h12"])
    # target never appears among features
    assert not any(c.startswith("price_l0") for c in X.columns)


def test_dm_test_symmetric():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(100, 24))
    f1 = y + rng.normal(0, 0.1, size=y.shape)
    f2 = y + rng.normal(0, 1.0, size=y.shape)
    stat, p = metrics.dm_test(y, f1, f2)
    assert stat < 0 and p < 0.01  # f1 clearly better


def test_asinh_scaler_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.normal(50, 30, size=(200, 5))
    sc = features.AsinhScaler().fit(a)
    back = sc.inverse_transform(sc.transform(a))
    assert np.allclose(a, back, atol=1e-8)


if __name__ == "__main__":
    test_no_lookahead_in_design_matrix()
    test_dm_test_symmetric()
    test_asinh_scaler_roundtrip()
    print("All tests passed.")


def test_validation_repairs_gaps_and_duplicates():
    from epf import validation
    df = data.make_synthetic(n_days=30)
    broken = pd.concat([df.iloc[:100], df.iloc[103:200], df.iloc[198:]])  # gap + dup
    clean, rep = validation.validate_hourly(broken)
    assert rep["duplicate_timestamps"] > 0
    assert rep["missing_hours"] >= 3
    assert len(clean) == len(df)
    assert rep["unresolved_nans"] == 0


def test_panel_repairs_dst_like_day():
    df = data.make_synthetic(n_days=40)
    df2 = df.drop(df.index[24 * 20 + 2])  # remove one hour of day 20 (like DST)
    panel = features.build_daily_panel(df2, ["res_load_fc"], verbose=False)
    assert len(panel) == 40  # day repaired, not dropped
    assert panel.notna().all().all()


def test_asymmetric_conformal_coverage():
    from epf.backtest import _conformal
    rng = np.random.default_rng(3)
    n = 400
    act = pd.DataFrame(rng.gamma(2, 10, size=(n, 24)))          # right-skewed
    fcst = act - rng.gamma(2, 10, size=(n, 24)) + 20             # biased, skewed errors
    lo, hi = _conformal(fcst, act, alpha=0.10, window=120)
    cov = ((act >= lo) & (act <= hi)).values[150:].mean()
    assert 0.85 < cov < 0.95, cov


def test_learwindows_runs():
    from epf.models import LEARWindows
    df = data.make_synthetic(n_days=200)
    exog = ["res_load_fc"]
    panel = features.build_daily_panel(df, exog, verbose=False)
    X, Y = features.build_design_matrix(panel, exog)
    m = LEARWindows(windows=(56, 84)).fit(X.iloc[:150], Y.iloc[:150])
    p = m.predict(X.iloc[150:155])
    assert p.shape == (5, 24) and np.isfinite(p).all()
