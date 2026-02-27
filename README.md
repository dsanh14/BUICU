# BUICU — Belief Updating for ICU Crowding Under Uncertainty

**CS 109 Challenge Project**

## Overview

BUICU models ICU crowding as a stochastic process using Bayesian inference.
The system performs sequential belief updating as new admission data arrives,
propagates uncertainty through Monte Carlo simulation, and computes calibrated
probabilistic forecasts of crowding events — all with explicit uncertainty
quantification and failure-mode analysis.

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
├── main.py                  # Full pipeline entry point
├── requirements.txt
├── src/
│   ├── synthetic_data.py    # MIMIC-IV-calibrated data generator
│   ├── bayesian_model.py    # Gamma-Poisson model + MC occupancy simulator
│   ├── failure_modes.py     # 5 structured failure-mode analyses
│   ├── nl_interface.py      # Natural-language explanation layer
│   └── visualizations.py    # All required + supporting visualizations
└── output/                  # Generated plots and writeup sections
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

All outputs (7 visualizations + writeup sections) are saved to `output/`.

## Key Outputs

1. **Posterior belief evolution** — how λ estimates concentrate over 180 days
2. **Posterior predictive check** — NegBin vs. empirical daily counts
3. **Calibration plot** — coverage + PIT histogram
4. **Occupancy forecast** — 48h ahead with uncertainty fan chart
5. **LOS distribution** — heavy-tail analysis
6. **Prior → Posterior** — visualization of Bayesian updating
7. **Summary dashboard** — all results in one figure

## Failure Modes Analyzed

- FM1: Non-stationarity (surges detected, ratio = 1.86)
- FM2: Independence violations (lag-1 autocorrelation = 0.47)
- FM3: Data quality (2.2% missing discharges)
- FM4: Heavy tails (excess kurtosis = 110)
- FM5: Feedback loops / Goodhart's Law (structural)
