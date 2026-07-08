# QCA — Coverage Validation of the Gap-Adjusted Davis-Kahan Perturbation Band

A Python research repository implementing and validating spectral monitoring
procedures for covariance matrix estimation based on the Davis-Kahan
perturbation theory.

## Overview

This repository validates:

| ID | Experiment | Hypothesis |
|----|-----------|-----------|
| exp1 | Coverage of the gap-adjusted Davis-Kahan band | H1: 100% coverage |
| exp2 | Alert-scale α calibration for null false-flag control | H1/H2 |
| exp3 | Functional-risk separation: op-norm vs spectral functionals | H2 |
| exp4 | Size & power of the two-window monitor | H1/H2/H3 |
| exp5a | Kurtosis immunity / variance inflation for scale-invariant functionals | H3 |
| exp5b | AR_K coverage under multiple κ estimation methods | H3/H4 |
| exp5c | Necessity of elliptical assumption | H3 |
| exp5d | Cleaning-debiasing wedge under spiked asymptotics | H4 |
| exp6 | Computational cost benchmarking | — |
| exp7 | Equity panel application (115 US large-cap stocks) | Applied |

## Repository Structure

```
├── data/               # data loading helpers (Massive API / yfinance)
├── src/                # core library
│   ├── estimators.py   # Sample, LW, OAS, QIS, MP-clip covariance estimators
│   ├── projector.py    # Davis-Kahan band & projector-distance (Lemma 1)
│   ├── simulation.py   # Factor-model DGP, spiked covariance, elliptical draws
│   ├── functionals.py  # AR_K, f_1, entropy H, trace; δ-method variances; κ estimators
│   └── monitoring.py   # Analytic null (Prop 2), bootstrap (Algo 2), rolling monitor
├── exp/                # experiment entry-points (call src/)
├── tests/              # pytest tests (98 tests, all passing)
└── results/            # metrics, plots, RESULTS.md
```

## Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install arch hmmlearn ruptures pyportfolioopt skfolio
pip install yfinance quantstats vectorbt
```

## Quick start

```bash
# Run all experiments (writes tables & figures to results/)
python exp/exp1_coverage.py
python exp/exp2_alpha_calibration.py
python exp/exp3_functional_risk.py
python exp/exp4_size_power.py
python exp/exp5a_kurtosis_immunity.py
python exp/exp5b_ar_coverage.py
python exp/exp5c_non_elliptical.py
python exp/exp5d_bias_wedge.py
python exp/exp6_benchmarking.py
python exp/exp7_equity_panel.py   # requires MASSIVE_TOKEN or internet for yfinance

# Run test suite
python -m pytest tests/ -v
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MASSIVE_TOKEN` | Massive REST API key for data download (exp7) |

## Key results

See `results/RESULTS.md` for full tables and figures.
