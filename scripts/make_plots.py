"""
Generate the README figures from a saved day-ahead backtest run.

Usage:
  python scripts/make_plots.py --run outputs/de_benchmark --out docs/img

Reads the CSVs written by run_dayahead.py (forecasts + actuals), rebuilds the
conformal bands with the exact backtest function (epf.backtest._conformal), and
writes three PNGs: a two-week forecast sample with the 90% band, MAE by model,
and MAE by hour of day.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epf.backtest import _conformal

# Reference dataviz palette (light mode), one fixed color per entity.
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {
    "actual": "#2a78d6",    # blue
    "ensemble": "#eb6834",  # orange
    "lear": "#1baf7a",      # aqua
    "gbt": "#eda100",       # yellow (sub-3:1 on light surface -> direct labels)
    "naive": MUT,           # benchmark stays neutral
}
LABELS = {"naive": "Naive", "lear": "LEAR", "gbt": "GBT", "ensemble": "Ensemble"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SEC,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUT,
    "ytick.color": MUT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def load_run(run_dir: str):
    read = lambda name: pd.read_csv(os.path.join(run_dir, name), index_col=0, parse_dates=True)
    actuals = read("dayahead_actuals.csv")
    forecasts = {}
    for name in ["naive", "lear", "gbt", "ensemble"]:
        path = os.path.join(run_dir, f"dayahead_forecast_{name}.csv")
        if os.path.exists(path):
            forecasts[name] = read(f"dayahead_forecast_{name}.csv")
    return actuals, forecasts


def to_hourly(wide: pd.DataFrame) -> pd.Series:
    """(days x 24) frame -> single hourly series."""
    s = wide.stack()
    idx = [d + pd.Timedelta(hours=int(h[1:])) for d, h in s.index]
    return pd.Series(s.values, index=pd.DatetimeIndex(idx))


def most_volatile_window(actuals: pd.DataFrame, days: int = 14) -> pd.DatetimeIndex:
    """The `days`-day OOS window with the highest price variance (hardest regime)."""
    days = min(days, len(actuals))
    daily_var = actuals.var(axis=1)
    score = daily_var.rolling(days).mean().dropna()
    end = score.idxmax()
    return actuals.loc[:end].index[-days:]


def fig_forecast_sample(actuals, forecasts, out):
    ens = forecasts["ensemble"]
    lo, hi = _conformal(ens, actuals, alpha=0.10, window=90)
    # restrict to days past the conformal warm-up so the band is drawable
    valid = lo.notna().all(axis=1)
    win = most_volatile_window(actuals.loc[valid] if valid.any() else actuals)
    a, f = to_hourly(actuals.loc[win]), to_hourly(ens.loc[win])
    l, h = to_hourly(lo.loc[win]), to_hourly(hi.loc[win])

    fig, ax = plt.subplots(figsize=(9.5, 3.8), dpi=150)
    ax.fill_between(l.index, l.values, h.values, color=COLORS["ensemble"],
                    alpha=0.15, linewidth=0, label="90% conformal band")
    ax.plot(a.index, a.values, color=COLORS["actual"], lw=1.4, label="Actual")
    ax.plot(f.index, f.values, color=COLORS["ensemble"], lw=1.4, label="Ensemble forecast")
    ax.set_ylabel("Day-ahead price (EUR/MWh)")
    ax.set_title(f"Most volatile two weeks of the test year "
                 f"({win[0].date()} - {win[-1].date()})", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_mae_by_model(actuals, forecasts, out):
    y = actuals.values
    order = ["naive", "lear", "gbt", "ensemble"]
    mae = {m: np.mean(np.abs(y - forecasts[m].values)) for m in order if m in forecasts}

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    names = list(mae)
    vals = [mae[m] for m in names]
    bars = ax.bar([LABELS[m] for m in names], vals,
                  color=[COLORS[m] for m in names], width=0.55,
                  edgecolor=SURFACE, linewidth=2)
    naive_mae = mae.get("naive")
    for b, m, v in zip(bars, names, vals):
        rmae = f"  (rMAE {v / naive_mae:.2f})" if naive_mae and m != "naive" else ""
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}{rmae}",
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Out-of-sample MAE by model — lower is better", color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_mae_by_hour(actuals, forecasts, out):
    y = actuals.values
    fig, ax = plt.subplots(figsize=(9.5, 3.5), dpi=150)
    for m in ["naive", "lear", "gbt", "ensemble"]:
        if m not in forecasts:
            continue
        mae_h = np.mean(np.abs(y - forecasts[m].values), axis=0)
        ax.plot(range(24), mae_h, color=COLORS[m], lw=1.6, label=LABELS[m])
        ax.annotate(LABELS[m], (23, mae_h[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=SEC)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("Out-of-sample MAE by delivery hour", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", ncols=4, fontsize=9)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/de_benchmark")
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args()

    actuals, forecasts = load_run(a.run)
    os.makedirs(a.out, exist_ok=True)
    fig_forecast_sample(actuals, forecasts, os.path.join(a.out, "forecast_sample.png"))
    fig_mae_by_model(actuals, forecasts, os.path.join(a.out, "mae_by_model.png"))
    fig_mae_by_hour(actuals, forecasts, os.path.join(a.out, "mae_by_hour.png"))
    print(f"Saved 3 figures to {a.out}/")


if __name__ == "__main__":
    main()
