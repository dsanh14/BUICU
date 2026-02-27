# BUICU — Belief Updating for ICU Crowding Under Uncertainty

**CS 109 Challenge Project**

## Overview

BUICU models ICU crowding as a stochastic process using Bayesian inference.
The system performs sequential belief updating as new admission data arrives,
propagates uncertainty through Monte Carlo simulation, and computes calibrated
probabilistic forecasts of crowding events — all with explicit uncertainty
quantification, model comparison, and failure-mode analysis.

## Probabilistic Model

| Random Variable | Distribution | Description |
|---|---|---|
| λ | Gamma(α₀, β₀) | Prior on arrival rate |
| N_t \| λ | Poisson(λ · Δt) | Arrivals in window Δt |
| λ \| data | Gamma(α₀+Σk, β₀+T) | Posterior (conjugate update) |
| N_future | NegBin(α_post, β_post/(β_post+Δt)) | Posterior predictive |
| L | Empirical / LogNormal mixture | Length of stay |
| O_t | Monte Carlo simulation | Occupancy (random variable) |

## Project Structure

```
BUICU/
├── main.py                  # Full 10-step pipeline
├── requirements.txt
├── src/
│   ├── synthetic_data.py    # MIMIC-IV-calibrated synthetic data generator
│   ├── bayesian_model.py    # Gamma-Poisson model + windowed model + prior sensitivity
│   ├── failure_modes.py     # 5 structured failure-mode analyses
│   ├── nl_interface.py      # Natural-language explanation layer
│   └── visualizations.py    # 10 publication-quality visualizations
└── output/                  # Generated plots and writeup sections
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

All outputs (10 visualizations + writeup sections) are saved to `output/`.

## Key Outputs (10 Figures)

| # | Figure | What it demonstrates |
|---|---|---|
| 01 | Belief evolution | Posterior mean + CI + anomaly markers + KL divergence |
| 02 | Posterior predictive check | NegBin PMF vs empirical histogram + Q-Q plot |
| 03 | Calibration comparison | Stationary vs windowed model calibration + PIT |
| 04 | Occupancy forecast | 48h fan chart from near-capacity snapshot |
| 05 | Model comparison | Stationary vs windowed posterior trajectories |
| 06 | Prior sensitivity | 3 priors converging to same posterior |
| 07 | Information gain | KL divergence per observation + anomaly detection |
| 08 | LOS distribution | Heavy-tail analysis (linear + log scale) |
| 09 | Prior vs posterior | Bayesian update transformation |
| 10 | Full dashboard | All key results in single figure |

## Advanced Analyses

- **Windowed Bayesian Model**: 14-day sliding window that tracks regime shifts
  (surge mean: 15.8/day vs. stationary estimate: 11.7/day)
- **Prior Sensitivity**: Three priors (uninformative, weakly informative, strong wrong)
  converge to near-identical posteriors (KL divergence < 0.001 for first two)
- **KL Divergence**: Information gain per observation; spikes during surges reveal
  the stationary model being surprised
- **Anomaly Detection**: Posterior predictive p-values flag 33/180 days as anomalous,
  concentrated during surge windows

## Failure Modes Analyzed

- **FM1**: Non-stationarity — surge/normal ratio = 1.86 (detected, high severity)
- **FM2**: Independence violations — lag-1 autocorrelation = 0.47 (detected, high)
- **FM3**: Data quality — 2.2% missing discharges (detected, medium)
- **FM4**: Heavy tails — excess kurtosis = 110 (detected, high)
- **FM5**: Feedback loops / Goodhart's Law (structural, always flagged)

## Key Results

- **Posterior λ**: 11.72 admissions/day, 95% CI [11.23, 12.23]
- **LOS**: median 1.7 days, mean 3.7 days (MIMIC-IV calibrated)
- **Census**: 25.2% of hours over 50-bed capacity
- **48h Forecast** (from near-capacity snapshot): 41.5% probability of exceeding
  capacity (95% CI: [40.0%, 42.8%])
- **Calibration**: Windowed model significantly better calibrated than stationary
