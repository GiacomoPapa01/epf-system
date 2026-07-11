"""
Download and cache the full ENTSO-E dataset (prices + load/wind/solar
forecasts and actuals) to data/entsoe_<zone>.csv.

The API key is read from --api-key, the ENTSOE_KEY environment variable,
or a .env file in the project root (ENTSOE_KEY=...). Get a free key at
https://transparency.entsoe.eu (account settings -> Web API Security Token).

Usage:
  python scripts/download_entsoe.py --zone DE_LU --start 2021-01-01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epf import data


def read_env_key(root: str) -> str | None:
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ENTSOE_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DE_LU")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--api-key", default=os.environ.get("ENTSOE_KEY") or read_env_key(root))
    ap.add_argument("--out", default=None, help="output CSV (default data/entsoe_<zone>.csv)")
    a = ap.parse_args()

    if not a.api_key:
        sys.exit("No API key: pass --api-key, set ENTSOE_KEY, or put ENTSOE_KEY=... in .env")

    out = a.out or os.path.join(root, "data", f"entsoe_{a.zone}.csv")
    print(f"Downloading {a.zone} from {a.start} to {a.end or 'today'} ...")
    df = data.load_entsoe(a.api_key, country_code=a.zone, start=a.start, end=a.end)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out)
    print(f"Saved {len(df)} hourly rows ({df.index[0]} -> {df.index[-1]}) to {out}")


if __name__ == "__main__":
    main()
