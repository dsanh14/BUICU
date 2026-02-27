"""
BUICU — Belief Updating for ICU Crowding Under Uncertainty
==========================================================

Main entry point. Executes the full pipeline:

  1. Generate synthetic MIMIC-IV-like dataset
  2. Define random variables and priors
  3. Perform sequential Bayesian updating
  4. Simulate occupancy trajectories (Monte Carlo)
  5. Compute crowding probabilities with uncertainty
  6. Analyze failure modes
  7. Generate visualizations
  8. Produce natural-language explanations and writeup sections

Usage:
    python main.py

All outputs are saved to the output/ directory.
"""

import os
import sys
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synthetic_data import SyntheticICUConfig, generate_dataset, summarize_dataset
from src.bayesian_model import (
    BayesianArrivalModel, LOSModel, OccupancySimulator, BeliefState
)
from src.failure_modes import FailureModeAnalyzer, format_failure_report
from src.nl_interface import (
    explain_current_belief, explain_belief_update,
    explain_crowding_forecast, explain_posterior_predictive,
    generate_writeup_sections
)
from src.visualizations import (
    plot_belief_evolution, plot_posterior_predictive_check,
    plot_calibration, plot_occupancy_forecast, plot_los_distribution,
    plot_prior_vs_posterior, create_summary_dashboard
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step1_generate_data():
    """Step 1: Generate and validate synthetic dataset."""
    print("=" * 70)
    print("STEP 1: Generating synthetic MIMIC-IV-like ICU dataset")
    print("=" * 70)

    config = SyntheticICUConfig(
        n_days=180,
        capacity=25,
        base_arrival_rate=10.0,
        surge_rate_multiplier=1.8,
        surge_windows=[(30, 50), (110, 130)],
        seed=42,
    )
    data = generate_dataset(config)
    summary = summarize_dataset(data)
    print(summary)
    print()
    return data, summary


def step2_compute_daily_counts(data):
    """Step 2: Aggregate admission times into daily counts for Bayesian updating."""
    print("=" * 70)
    print("STEP 2: Computing daily admission counts")
    print("=" * 70)

    n_days = data['n_days']
    admission_hours = data['admissions']
    daily_counts = np.zeros(n_days, dtype=int)
    for t in admission_hours:
        day_idx = int(t / 24.0)
        if 0 <= day_idx < n_days:
            daily_counts[day_idx] += 1

    print(f"  Total days: {n_days}")
    print(f"  Mean daily admissions: {np.mean(daily_counts):.2f}")
    print(f"  Std daily admissions:  {np.std(daily_counts):.2f}")
    print(f"  Min/Max daily:         {np.min(daily_counts)}/{np.max(daily_counts)}")
    print()
    return daily_counts


def step3_bayesian_updating(daily_counts, surge_windows):
    """
    Step 3: Sequential Bayesian updating.

    Prior: λ ~ Gamma(α₀=2, β₀=0.2)
      → prior mean = 10, prior std = 7.07
      → weakly informative: centered near typical ICU rate but very uncertain

    Each day's count k_i updates:
      α → α + k_i
      β → β + 1
    """
    print("=" * 70)
    print("STEP 3: Sequential Bayesian updating of arrival rate")
    print("=" * 70)

    ALPHA_0, BETA_0 = 2.0, 0.2
    model = BayesianArrivalModel(alpha_0=ALPHA_0, beta_0=BETA_0)

    print(f"  Prior: Gamma({ALPHA_0}, {BETA_0})")
    print(f"  Prior mean: {ALPHA_0 / BETA_0:.1f}, "
          f"Prior std: {np.sqrt(ALPHA_0) / BETA_0:.2f}")
    print()

    # Save belief at specific milestones for detailed explanation
    milestones = {1, 5, 10, 30, 50, 90, 130, 180}
    explanations = []

    for i, k in enumerate(daily_counts):
        old_belief = BeliefState(
            alpha=model.belief.alpha, beta=model.belief.beta,
            time=model.belief.time, total_arrivals=model.belief.total_arrivals
        )

        day_num = i + 1
        in_surge = any(s <= i < e for s, e in surge_windows)
        label = f"day {day_num}"
        if in_surge:
            label += " [SURGE]"

        model.update(int(k), window_duration=1.0, label=label)

        if day_num in milestones:
            expl = explain_belief_update(old_belief, model.belief, int(k), 1.0)
            explanations.append((day_num, expl))

    print(f"  Final posterior: Gamma({model.belief.alpha:.0f}, "
          f"{model.belief.beta:.1f})")
    ci = model.belief.credible_interval(0.95)
    print(f"  Posterior mean: {model.belief.mean:.3f}")
    print(f"  Posterior 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print()

    for day_num, expl in explanations:
        print(f"--- Day {day_num} belief update ---")
        print(expl)
        print()

    return model, ALPHA_0, BETA_0, explanations


def step4_posterior_predictive(model):
    """Step 4: Compute and explain posterior predictive distribution."""
    print("=" * 70)
    print("STEP 4: Posterior predictive distribution")
    print("=" * 70)

    k_vals, pmf = model.posterior_predictive_pmf(future_window=1.0)
    expl = explain_posterior_predictive(k_vals, pmf, window_days=1.0)
    print(expl)
    print()

    k_vals_2d, pmf_2d = model.posterior_predictive_pmf(future_window=2.0)
    expl_2d = explain_posterior_predictive(k_vals_2d, pmf_2d, window_days=2.0)
    print(expl_2d)
    print()

    return k_vals, pmf


def step5_occupancy_simulation(model, data):
    """
    Step 5: Monte Carlo occupancy simulation.

    We simulate forward from "now" (end of observation period) to
    forecast occupancy over the next 48 hours with full uncertainty.
    """
    print("=" * 70)
    print("STEP 5: Monte Carlo occupancy simulation (48h forecast)")
    print("=" * 70)

    los_model = LOSModel(data['los_hours'], mode="empirical")
    print("LOS model summary:", los_model.summary_stats())
    print()

    simulator = OccupancySimulator(model, los_model, data['capacity'])

    # Estimate current patients: those admitted in the last few days
    # whose LOS hasn't expired
    end_hour = data['n_days'] * 24
    recent_mask = (data['admission_times_true'] > end_hour - 72)
    remaining_los = []
    for i in np.where(recent_mask)[0]:
        discharge = data['discharge_times_true'][i]
        if discharge > end_hour:
            remaining_los.append(discharge - end_hour)
    current_patients = np.array(remaining_los) if remaining_los else np.array([0.0])

    print(f"  Current patients in ICU: {len(current_patients)}")
    print(f"  Mean remaining LOS: {np.mean(current_patients):.1f} hours")
    print()

    rng = np.random.default_rng(789)
    sim_result = simulator.simulate_trajectories(
        current_patients, forecast_hours=48,
        n_trajectories=2000, rng=rng
    )

    print(f"  Forecast horizon: 48 hours")
    print(f"  Trajectories simulated: 2000")
    print(f"  Mean occupancy at t=24h: {sim_result['mean'][24]:.1f}")
    print(f"  95% CI at t=24h: [{sim_result['ci_low'][24]:.0f}, "
          f"{sim_result['ci_high'][24]:.0f}]")
    print(f"  Peak P(overcrowded): {np.max(sim_result['p_overcrowded']):.3f}")
    print()

    crowd_result = simulator.crowding_probability(
        current_patients, horizon_hours=48, n_samples=5000, rng=rng
    )
    print(f"  P(overcrowded within 48h): {100 * crowd_result['probability']:.1f}%")
    print(f"  95% CI: [{100 * crowd_result['ci_low']:.1f}%, "
          f"{100 * crowd_result['ci_high']:.1f}%]")
    print()

    return sim_result, crowd_result, los_model


def step6_failure_analysis(daily_counts, data):
    """Step 6: Systematic failure-mode analysis."""
    print("=" * 70)
    print("STEP 6: Failure-mode analysis")
    print("=" * 70)

    analyzer = FailureModeAnalyzer(data['capacity'])
    missing_frac = np.mean(np.isnan(data['discharges']))

    reports = analyzer.analyze_all(
        daily_counts=daily_counts,
        los_hours=data['los_hours'],
        census_hourly=data['census_hourly'],
        surge_windows=data['surge_windows'],
        missing_fraction=missing_frac,
    )

    report_text = format_failure_report(reports)
    print(report_text)

    penalty = analyzer.combined_confidence_penalty(reports)
    print(f"\nCombined uncertainty widening factor: ×{penalty:.2f}")
    print()

    return reports, penalty


def step7_visualizations(model, history, daily_counts, sim_result,
                          data, alpha_0, beta_0):
    """Step 7: Generate all required visualizations."""
    print("=" * 70)
    print("STEP 7: Generating visualizations")
    print("=" * 70)

    history = model.history

    plot_belief_evolution(
        history, data['surge_windows'],
        save_path=os.path.join(OUTPUT_DIR, "1_belief_evolution.png")
    )
    print("  ✓ 1_belief_evolution.png")

    plot_posterior_predictive_check(
        model, daily_counts,
        save_path=os.path.join(OUTPUT_DIR, "2_posterior_predictive_check.png")
    )
    print("  ✓ 2_posterior_predictive_check.png")

    plot_calibration(
        model, daily_counts,
        save_path=os.path.join(OUTPUT_DIR, "3_calibration.png")
    )
    print("  ✓ 3_calibration.png")

    plot_occupancy_forecast(
        sim_result,
        save_path=os.path.join(OUTPUT_DIR, "4_occupancy_forecast.png")
    )
    print("  ✓ 4_occupancy_forecast.png")

    plot_los_distribution(
        data['los_hours'],
        save_path=os.path.join(OUTPUT_DIR, "5_los_distribution.png")
    )
    print("  ✓ 5_los_distribution.png")

    plot_prior_vs_posterior(
        model, alpha_0, beta_0,
        save_path=os.path.join(OUTPUT_DIR, "6_prior_vs_posterior.png")
    )
    print("  ✓ 6_prior_vs_posterior.png")

    create_summary_dashboard(
        history, model, daily_counts, sim_result, data['los_hours'],
        data['surge_windows'], alpha_0, beta_0,
        save_path=os.path.join(OUTPUT_DIR, "7_dashboard.png")
    )
    print("  ✓ 7_dashboard.png")
    print()


def step8_nl_output(model, crowd_result, reports, penalty, data_summary,
                     alpha_0, beta_0):
    """Step 8: Generate natural-language outputs and writeup."""
    print("=" * 70)
    print("STEP 8: Natural-language outputs")
    print("=" * 70)

    prior_mean = alpha_0 / beta_0
    belief_text = explain_current_belief(model.belief, prior_mean)
    print(belief_text)
    print()

    forecast_text = explain_crowding_forecast(crowd_result, reports, penalty)
    print(forecast_text)
    print()

    writeup = generate_writeup_sections(
        model.belief, prior_mean, crowd_result, reports, data_summary
    )

    writeup_path = os.path.join(OUTPUT_DIR, "writeup_sections.txt")
    with open(writeup_path, 'w') as f:
        f.write(writeup)
    print(f"  Writeup sections saved to {writeup_path}")
    print()


def main():
    """Execute the full BUICU pipeline."""
    ensure_output_dir()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  BUICU — Belief Updating for ICU Crowding Under Uncertainty        ║")
    print("║  CS109 Challenge Project                                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    data, data_summary = step1_generate_data()
    daily_counts = step2_compute_daily_counts(data)
    model, alpha_0, beta_0, _ = step3_bayesian_updating(
        daily_counts, data['surge_windows']
    )
    step4_posterior_predictive(model)
    sim_result, crowd_result, los_model = step5_occupancy_simulation(model, data)
    reports, penalty = step6_failure_analysis(daily_counts, data)
    step7_visualizations(model, model.history, daily_counts, sim_result,
                          data, alpha_0, beta_0)
    step8_nl_output(model, crowd_result, reports, penalty, data_summary,
                     alpha_0, beta_0)

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
