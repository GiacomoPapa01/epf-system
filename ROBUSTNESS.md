# ROBUSTNESS.md — Change Log & Rationale

Tracking of every robustness improvement over the baseline pipeline (v0.1.0 → v0.2.0).
Each entry: what changed, why, where, and how it is verified.

---

## v0.2.0 — 2026-07-10

### 1. Data validation module (`epf/validation.py`) — NEW
- **What**: `validate_hourly()` repairs duplicated timestamps (averaged), reindexes to a complete hourly grid, interpolates gaps up to 6h, and reports (never silently fixes): missing hours, unresolved NaNs, prices outside EPEX bounds [−500, 4000], negative load/wind/solar, solar generation at night, robust-z (median/MAD) spike count.
- **Why**: real ENTSO-E pulls routinely contain duplicated DST hours, publication gaps, and unit glitches; modelling on unvalidated data is the #1 silent failure mode.
- **Verified by**: `test_validation_repairs_gaps_and_duplicates` — injects a 3h gap + duplicated rows and checks full repair with correct reporting. Wired into `run_dayahead.py` (report printed at startup).

### 2. DST / incomplete-day repair (`features.build_daily_panel`)
- **What**: days missing ≤ 2 hours are repaired by within-day linear interpolation; days missing more are dropped **with an explicit printed report** (previously: any non-24h day silently dropped).
- **Why**: spring-forward days have 23 hours in local-time datasets; dropping ~2 days/year biases weekday coverage and breaks the lag structure of neighbouring days for the naive benchmark.
- **Verified by**: `test_panel_repairs_dst_like_day`.

### 3. Asymmetric conformal intervals for day-ahead (`backtest._conformal`)
- **What**: bands now use rolling quantiles of the **signed** residuals (α/2 and 1−α/2) instead of symmetric |residual| bands.
- **Why**: power price errors are right-skewed (spikes). Symmetric bands over-cover on the left, under-cover exactly where risk lives (upside spikes). Asymmetric split-conformal preserves the distribution-free guarantee while matching the skew.
- **Verified by**: `test_asymmetric_conformal_coverage` — gamma-skewed errors, empirical coverage required within [0.85, 0.95] at nominal 90%.

### 4. `LEARWindows` — calibration-window averaging (`epf/models.py`) — NEW
- **What**: LEAR averaged over multiple calibration windows (default 56, 84, 365, 730 days, auto-clipped to available history), selectable via `--models lear,learw,gbt,ensemble`.
- **Why**: one of the most consistent accuracy gains in the EPF literature (Lago et al. 2021; Marcjasz–Serafin–Weron): short windows adapt to regime shifts (e.g. 2021–22 gas crisis), long windows stabilize. On stationary synthetic data it will *not* beat full-window LEAR (expected — no regimes to adapt to); its value shows on real multi-regime data.
- **Verified by**: `test_learwindows_runs` + end-to-end backtest including DM comparison.

### 5. GBT target winsorization (`epf/models.py`)
- **What**: optional clipping of training targets at the (0.1%, 99.9%) quantiles of the calibration window (default on; `winsorize=None` to disable).
- **Why**: single extreme spikes dominate the squared-loss gradients of boosted trees and degrade all other hours. Winsorizing the *training target only* (never the evaluation data) is the standard fix; forecasts remain evaluated against true unclipped prices.

### 6. CQR correction for intraday bands (`epf/intraday.py`) — added late v0.1
- **What**: Conformalized Quantile Regression (Romano et al. 2019): rolling conformal adjustment of the q10/q90 GBT bands using past out-of-sample conformity scores.
- **Why / result**: raw quantile GBT under-covered (68.6% vs 80% nominal on the smoke test); with CQR coverage is 81.8% at unchanged point accuracy.

### 7. `NaiveDaily` gap handling fix (`epf/models.py`)
- **What**: rewrote the reference-day lookup: exact lagged day if present, otherwise last available day before it — previous one-liner was correct but fragile/unreadable; now explicit and covered by panel-repair behaviour.

### 8. Warning hygiene & diagnostics
- LassoLarsIC/Lasso convergence warnings suppressed *inside* `LEAR.fit` only (they are expected on collinear designs and were flooding logs, hiding real warnings).
- `LEAR.nonzero_share()` diagnostic: average fraction of features selected — sanity check for degenerate fits (≈0 → underfit; ≈1 → λ too small).

### 9. Test suite expanded: 3 → 7 tests
`no_lookahead` (structural), `dm_test` sanity, `AsinhScaler` roundtrip, validation repair, DST-day repair, asymmetric conformal coverage, LEARWindows shape/finiteness. All passing on v0.2.0.

---

## Known remaining gaps (deliberate, documented)
- Intraday layer still validated on a **proxy spread** until real EPEX ID3/ID1 data is plugged in (`--id-csv`).
- No hyperparameter search for GBT (fixed sensible defaults); a nested time-series CV would be the clean extension, at ~10× runtime.
- DM test uses daily multivariate losses with a t-approximation; for publication-grade claims add Giacomini–White conditional predictive ability.
- Market-day boundaries are UTC; for local-market day alignment (CET auctions) convert before `build_daily_panel`.

## Baseline (v0.1.0) — for reference
LEAR + GBT + ensemble, walk-forward with rolling recalibration, symmetric split-conformal, multivariate DM test, synthetic/epftoolbox/ENTSO-E loaders, intraday DA→ID spread with quantile GBT.
