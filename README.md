# BUICU — Belief Updating for ICU Crowding Under Uncertainty

**CS 109 Challenge Project**

## Overview

BUICU models ICU crowding as a stochastic process using Bayesian inference.
The system performs sequential belief updating, propagates uncertainty via
Monte Carlo simulation, compares models with proper scoring rules, decomposes
uncertainty into epistemic vs. aleatoric components, and systematically
analyzes failure modes — demonstrating 16 CS109 concepts in a single project.

## Probabilistic Model

| Random Variable | Distribution | Description |
|---|---|---|
| λ | Gamma(α₀, β₀) | Prior on arrival rate |
| N_t \| λ | Poisson(λ · Δt) | Arrivals in window Δt |
| λ \| data | Gamma(α₀+Σk, β₀+T) | Posterior (conjugate update) |
| N_future | NegBin(α_post, β_post/(β_post+Δt)) | Posterior predictive |
| L | Empirical / LogNormal mixture | Length of stay |
| O_t | Monte Carlo simulation | Occupancy (random variable) |

## Usage

```bash
pip install -r requirements.txt
python main.py
```

All outputs (14 figures + writeup) are saved to `output/`.

## 14 Visualizations

| # | Figure | CS109 Concept |
|---|---|---|
| 01 | Belief evolution + anomalies + KL | Bayesian updating, KL divergence |
| 02 | Posterior predictive check + Q-Q | Posterior predictive, calibration |
| 03 | Calibration (stationary vs windowed) | Coverage, PIT, model comparison |
| 04 | 48h occupancy forecast fan chart | Monte Carlo, uncertainty intervals |
| 05 | Stationary vs windowed model | Non-stationarity, adaptive inference |
| 06 | Prior sensitivity convergence | Prior robustness, Bayesian consistency |
| 07 | Information gain + anomaly detection | KL divergence, hypothesis testing |
| 08 | LOS heavy-tail analysis | Distribution fitting, tail risk |
| 09 | Prior → Posterior transformation | Bayes' theorem visualization |
| 10 | Log predictive score comparison | Proper scoring rules |
| 11 | Sensitivity analysis tornado | Decision sensitivity, robustness |
| 12 | Variance decomposition | Law of total variance |
| 13 | MLE vs Bayesian comparison | Frequentist vs Bayesian, CLT |
| 14 | Full summary dashboard | All key results |

## CS109 Concepts Demonstrated (16)

1. Random Variables  2. Probability Distributions (Poisson, Gamma, NegBin, LogNormal)
3. Conditional Probability  4. Bayes' Theorem  5. Posterior Predictive
6. Conjugate Priors  7. Law of Total Variance  8. Monte Carlo Simulation
9. Maximum Likelihood Estimation  10. Central Limit Theorem
11. Information Theory (KL divergence)  12. Hypothesis Testing (p-values)
13. Model Comparison (proper scoring)  14. Calibration  15. Sensitivity Analysis
16. Prior Sensitivity

## Key Results

- **Posterior λ**: 11.72 adm/day, 95% CI [11.23, 12.23]
- **Model comparison**: Windowed wins by 39.4 log-score units
- **Variance decomposition**: 99.4% stochastic, 0.6% parameter after 180 days
- **Crowding forecast**: 41.5% within 48h (from near-capacity scenario)
- **Sensitivity**: Capacity -20% → P(crowded) jumps from 9% to 96%
- **5 failure modes detected**, combined CI widening ×2.27
