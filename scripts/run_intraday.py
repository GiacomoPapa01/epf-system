"""
Intraday (DA->ID spread) backtest.

Usage:
  python scripts/run_intraday.py --source synthetic --days 550
  # with real data: pass ENTSO-E key for actuals/forecasts and provide an
  # ID price CSV (EPEX ID3/ID1) via --id-csv with columns [timestamp, id_price]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epf import data, intraday


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "entsoe"])
    ap.add_argument("--days", type=int, default=550)
    ap.add_argument("--api-key", default=None, help="default: ENTSOE_KEY env var or .env")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--id-csv", default=None, help="CSV with real ID prices (timestamp,id_price)")
    ap.add_argument("--cal-hours", type=int, default=24 * 365)
    ap.add_argument("--test-hours", type=int, default=24 * 60)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    if a.source == "synthetic":
        df = data.make_synthetic(n_days=a.days)
    else:
        df = data.load_entsoe_csv()  # cached by scripts/download_entsoe.py
        if df is None:
            df = data.load_entsoe(a.api_key or data.find_api_key(), start=a.start)

    if a.id_csv:
        idp = pd.read_csv(a.id_csv, index_col=0, parse_dates=True).squeeze()
        idp.index = idp.index.tz_convert("UTC") if idp.index.tz else idp.index.tz_localize("UTC")
        spread = (idp - df["price"]).dropna()
        print("Using REAL intraday prices.")
    else:
        spread = intraday.make_proxy_spread(df)
        print("WARNING: using PROXY spread (no real ID data). Pipeline demo only.")

    feats = intraday.build_intraday_features(df)
    bt = intraday.backtest_intraday(feats, spread, cal_hours=a.cal_hours,
                                    test_hours=a.test_hours)
    m = intraday.intraday_metrics(bt)
    print("\n=== Intraday spread metrics (out-of-sample) ===")
    for k, v in m.items():
        print(f"  {k:>20}: {v}")

    os.makedirs(a.out, exist_ok=True)
    bt.to_csv(os.path.join(a.out, "intraday_backtest.csv"))
    pd.Series(m).to_csv(os.path.join(a.out, "intraday_metrics.csv"))
    print(f"\nSaved to {a.out}/")


if __name__ == "__main__":
    main()
