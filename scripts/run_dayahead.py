"""
Day-ahead backtest entry point.

Usage:
  python scripts/run_dayahead.py --source synthetic --days 500 --cal 300 --test 60
  python scripts/run_dayahead.py --source epftoolbox --market DE --cal 730 --test 365
  python scripts/run_dayahead.py --source entsoe --api-key $ENTSOE_KEY --start 2021-01-01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epf import backtest, data, features, validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "epftoolbox", "entsoe"])
    ap.add_argument("--market", default="DE")
    ap.add_argument("--api-key", default=None, help="default: ENTSOE_KEY env var or .env")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--days", type=int, default=730, help="synthetic only")
    ap.add_argument("--cal", type=int, default=365)
    ap.add_argument("--test", type=int, default=90)
    ap.add_argument("--recal", type=int, default=1)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", default="lear,gbt,ensemble", help="comma list: lear,learw,gbt,ensemble")
    a = ap.parse_args()

    if a.source == "synthetic":
        df = data.make_synthetic(n_days=a.days)
        exog = ["load_fc", "wind_fc", "solar_fc", "res_load_fc"]
    elif a.source == "epftoolbox":
        df = data.load_epftoolbox(a.market)
        exog = [c for c in ["load_fc", "renewables_fc", "res_load_fc", "exog1", "exog2"] if c in df.columns]
    else:
        df = data.load_entsoe_csv()  # cached by scripts/download_entsoe.py
        if df is None:
            df = data.load_entsoe(a.api_key or data.find_api_key(), start=a.start)
        exog = ["load_fc", "wind_fc", "solar_fc", "res_load_fc"]

    df, report = validation.validate_hourly(df)
    validation.assert_clean(report)
    print("Data quality report:", {k: v for k, v in report.items() if v})
    print(f"Data: {df.index[0]} -> {df.index[-1]}  ({len(df)} hours)")
    panel = features.build_daily_panel(df, exog)
    X, Y = features.build_design_matrix(panel, exog)
    print(f"Design matrix: {X.shape[0]} days x {X.shape[1]} features")

    res = backtest.run_backtest(X, Y, panel, cal_days=a.cal, test_days=a.test,
                                recal_every=a.recal,
                                models=tuple(a.models.split(",")))
    print("\n=== Metrics (out-of-sample) ===")
    print(res.metrics().to_string())
    print("\n=== Diebold-Mariano p-values (row vs col) ===")
    print(res.dm_matrix().to_string())

    os.makedirs(a.out, exist_ok=True)
    res.metrics().to_csv(os.path.join(a.out, "dayahead_metrics.csv"))
    res.dm_matrix().to_csv(os.path.join(a.out, "dayahead_dm_pvalues.csv"))
    for name, f in res.forecasts.items():
        f.to_csv(os.path.join(a.out, f"dayahead_forecast_{name}.csv"))
    for name, (lo, hi) in res.intervals.items():
        lo.to_csv(os.path.join(a.out, f"dayahead_interval_lo_{name}.csv"))
        hi.to_csv(os.path.join(a.out, f"dayahead_interval_hi_{name}.csv"))
    res.actuals.to_csv(os.path.join(a.out, "dayahead_actuals.csv"))
    print(f"\nSaved forecasts + metrics to {a.out}/")


if __name__ == "__main__":
    main()
