# EPF System — Day-Ahead & Intraday Electricity Price Forecasting

End-to-end, walk-forward-backtested forecasting system for European power markets. Day-ahead layer follows the strongest published open benchmarks (Lago, Marcjasz, De Schutter, Weron, *Applied Energy* 2021); the intraday layer models the DA→ID spread as a function of residual-load surprises with quantile gradient boosting and conformalized bands.

## Layout

```
epf-system/
├── epf/
│   ├── data.py        # loaders: ENTSO-E API, epftoolbox open datasets, synthetic
│   ├── validation.py  # data quality: gap/duplicate repair + anomaly report
│   ├── features.py    # LEAR-style design matrix, residual load, AsinhScaler
│   ├── models.py      # NaiveDaily, LEAR, LEARWindows, GBT (winsorized), Ensemble
│   ├── backtest.py    # walk-forward engine, rolling recalibration, split-conformal
│   ├── intraday.py    # DA→ID spread model (quantile GBT + CQR correction)
│   └── metrics.py     # MAE, rMAE, sMAPE, pinball, coverage, Diebold-Mariano
├── scripts/
│   ├── run_dayahead.py
│   └── run_intraday.py
├── tests/test_pipeline.py   # 7 tests: no-look-ahead, DM, conformal coverage, repairs
├── ROBUSTNESS.md      # change log of every robustness improvement + rationale
├── data/      # cached downloads
└── outputs/   # forecasts, intervals, metrics, DM p-values (CSV)
```

## Quick start

```bash
pip install -r requirements.txt
python tests/test_pipeline.py

# 1) Reproducible open benchmark (6 years, German market, Lago et al. data)
python scripts/run_dayahead.py --source epftoolbox --market DE --cal 730 --test 365 --recal 1

# 2) Full real data via ENTSO-E Transparency (free token: transparency.entsoe.eu)
#    Put ENTSOE_KEY=... in a .env file (gitignored) or export it as env var.
python scripts/download_entsoe.py --zone DE_LU --start 2021-01-01   # caches to data/
python scripts/run_dayahead.py --source entsoe --start 2021-01-01 --cal 730 --test 365

# 3) Intraday spread (proxy demo; plug real EPEX ID3 with --id-csv)
python scripts/run_intraday.py --source entsoe --id-csv my_id3.csv
```

Smoke test without any data/API (synthetic German-like market):
```bash
python scripts/run_dayahead.py --source synthetic --days 480 --cal 300 --test 45 --recal 5
python scripts/run_intraday.py --source synthetic --days 300 --cal-hours 4800 --test-hours 720
```

## Methodology

### Day-ahead
- **Framing**: forecast the 24 hourly prices of day D using only information available before the 12:00 D−1 gate closure. No-look-ahead is enforced structurally (day-level shifts) and unit-tested.
- **Features** (LEAR set + extensions): full 24-hour price vectors at lags 1/2/3/7 days; day-ahead forecasts of load, wind, solar and **residual load** for D, D−1, D−7; weekday dummies; annual sin/cos.
- **Models**:
  - `NaiveDaily` — similar-day benchmark (mandatory: rMAE denominator);
  - `LEAR` — one Lasso per hour, λ by AIC, on asinh/median-MAD-scaled data (robust to spikes, handles negative prices);
  - `GBT` — LightGBM if available, sklearn HistGradientBoosting otherwise (auto-fallback, works on Python 3.14);
  - `Ensemble` — simple average (the hardest baseline to beat in EPF).
- **Backtest**: rolling calibration window (default 730 days), daily recalibration (`--recal 1`), long out-of-sample test (≥ 1 year recommended).
- **Uncertainty**: rolling **asymmetric** split-conformal per hour on signed out-of-sample residuals → distribution-free 90% intervals that adapt to volatility regimes and to spike skew.
- **Evaluation**: MAE, RMSE, rMAE, sMAPE, empirical coverage, pinball loss, and **multivariate Diebold–Mariano** p-value matrix on daily losses.

### Intraday
- **Target**: `spread(h) = ID_price(h) − DA_price(h)` (ID3/ID1/VWAP as reference).
- **Driver**: residual-load surprise `res_load_act − res_load_fc`, lagged by a configurable knowledge lag (default 2h) to mimic actual publication delays — the model never sees the concurrent hour.
- **Model**: quantile gradient boosting (q10/q50/q90) + **CQR** post-hoc conformal correction of the bands.
- **Metrics**: MAE vs the naive "ID = DA" benchmark, 80% band coverage, directional accuracy of the spread sign (the trading-relevant number).

### Reference results (synthetic smoke test, 45-day OOS)
| model | MAE | rMAE | cov90% |
|---|---|---|---|
| naive | 21.8 | 1.00 | — |
| LEAR | 11.5 | 0.53 | 87% |
| GBT | 16.8 | 0.77 | 85% |
| ensemble | 12.0 | 0.55 | 87.5% |

Intraday (proxy spread): rMAE 0.47 vs ID=DA, coverage 81.8% (nominal 80%), directional accuracy 86%.

## Data sources (maximize coverage)
1. **ENTSO-E Transparency** (free API): DA prices, load forecast/actual, wind & solar forecast/actual → both layers, any EU zone.
2. **epftoolbox open datasets**: DE, NP, PJM, BE, FR — 6 years each, exact reproducibility against published benchmarks.
3. **EPEX intraday indices** (ID1/ID3, licensed): required for the real intraday target; the pipeline accepts them via `--id-csv`.
4. Optional extensions: TTF gas & EUA carbon (fuel-switching regressors), cross-border flows/ATC, outage data (UMM).

## Honest limitations
- The intraday demo uses a **proxy spread**: the machinery is validated, the economics need real ID prices.
- DST days are dropped (standard in the literature); production code should handle 23/25-hour days explicitly.
- Daily recalibration of GBT over a 1-year test takes hours on a laptop; `--recal 7` is a good accuracy/runtime compromise for trees (keep `--recal 1` for LEAR).
