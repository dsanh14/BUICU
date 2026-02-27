"""
BUICU — Belief Updating for ICU Crowding Under Uncertainty
==========================================================

Main entry point. Executes the full pipeline:

  1. Generate synthetic MIMIC-IV-like dataset
  2. Define random variables and priors
  3. Perform sequential Bayesian updating (with KL tracking + anomaly detection)
  4. Run windowed (adaptive) Bayesian model for comparison
  5. Run prior sensitivity analysis
  6. Simulate occupancy trajectories (Monte Carlo)
  7. Compute crowding probabilities with uncertainty
  8. Analyze failure modes
  9. Generate all visualizations (10 figures)
 10. Produce natural-language explanations and writeup sections

Usage:
    python main.py

All outputs are saved to the output/ directory.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synthetic_data import SyntheticICUConfig, generate_dataset, summarize_dataset
from src.bayesian_model import (
    BayesianArrivalModel, LOSModel, OccupancySimulator, BeliefState,
    WindowedBayesianModel, PriorSensitivityAnalysis, kl_divergence_gamma
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
    plot_prior_vs_posterior, plot_model_comparison, plot_prior_sensitivity,
    plot_information_gain, create_summary_dashboard
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
        capacity=50,
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
    """Step 2: Aggregate admission times into daily counts."""
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
    Step 3: Sequential Bayesian updating with KL tracking and anomaly detection.

    Prior: λ ~ Gamma(α₀=2, β₀=0.2)
      → prior mean = 10, prior std = 7.07
      → weakly informative: centered near typical ICU rate but very uncertain
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

    ci = model.belief.credible_interval(0.95)
    print(f"  Final posterior: Gamma({model.belief.alpha:.0f}, "
          f"{model.belief.beta:.1f})")
    print(f"  Posterior mean: {model.belief.mean:.3f}")
    print(f"  Posterior 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Anomaly summary
    anomalies = model.history.anomaly_flags
    n_anomalies = sum(anomalies)
    print(f"\n  Anomalous days detected: {n_anomalies} / {len(daily_counts)}")

    # Total information gain
    total_kl = sum(model.history.kl_divergences)
    print(f"  Total KL divergence (information gain): {total_kl:.2f}")
    print()

    for day_num, expl in explanations:
        print(f"--- Day {day_num} belief update ---")
        print(expl)
        print()

    return model, ALPHA_0, BETA_0, explanations


def step4_windowed_model(daily_counts, surge_windows):
    """
    Step 4: Windowed Bayesian model for non-stationarity.

    Uses only the last 14 days of data at each time step, allowing the
    posterior to track regime changes rather than averaging over all history.
    """
    print("=" * 70)
    print("STEP 4: Windowed (adaptive) Bayesian model")
    print("=" * 70)

    windowed = WindowedBayesianModel(window_days=14, alpha_0=2.0, beta_0=0.2)
    w_history = windowed.fit(daily_counts)

    # Compare during surge vs. normal
    surge_days = set()
    for s, e in surge_windows:
        surge_days.update(range(int(s), int(e)))

    surge_means = [w_history.means[t] for t in surge_days if t < len(w_history.means)]
    normal_means = [w_history.means[t] for t in range(len(w_history.means))
                    if t not in surge_days]

    print(f"  Window size: 14 days")
    print(f"  Windowed mean during surges:  {np.mean(surge_means):.2f}")
    print(f"  Windowed mean during normal:  {np.mean(normal_means):.2f}")
    print(f"  Ratio: {np.mean(surge_means) / np.mean(normal_means):.2f}")
    print(f"  (Stationary model cannot distinguish these regimes)")
    print()

    return windowed, w_history


def step5_prior_sensitivity(daily_counts):
    """
    Step 5: Prior sensitivity analysis.

    Three priors: uninformative, weakly informative, strong (wrong center).
    All should converge to similar posteriors — demonstrating how evidence
    overwhelms prior beliefs.
    """
    print("=" * 70)
    print("STEP 5: Prior sensitivity analysis")
    print("=" * 70)

    psa = PriorSensitivityAnalysis()
    histories = psa.run(daily_counts)

    for name, hist in histories.items():
        final_mean = hist.means[-1]
        final_ci = (hist.ci_lows[-1], hist.ci_highs[-1])
        print(f"  {name:30s}: mean={final_mean:.3f}, "
              f"95% CI=[{final_ci[0]:.3f}, {final_ci[1]:.3f}]")

    # Compute pairwise KL divergences between final posteriors
    items = list(histories.items())
    print("\n  Pairwise KL divergences (final posteriors):")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            n1, h1 = items[i]
            n2, h2 = items[j]
            kl = kl_divergence_gamma(h1.alphas[-1], h1.betas[-1],
                                     h2.alphas[-1], h2.betas[-1])
            print(f"    KL({n1[:12]:12s} || {n2[:12]:12s}) = {kl:.6f}")
    print()

    return psa, histories


def step6_posterior_predictive(model):
    """Step 6: Compute and explain posterior predictive distribution."""
    print("=" * 70)
    print("STEP 6: Posterior predictive distribution")
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


def step7_occupancy_simulation(model, data):
    """
    Step 7: Monte Carlo occupancy simulation.

    We forecast from a mid-surge snapshot (day 40) where the ICU is under
    stress, making the crowding prediction non-trivial and demonstrating
    how the model behaves when it matters most.
    """
    print("=" * 70)
    print("STEP 7: Monte Carlo occupancy simulation (48h forecast)")
    print("=" * 70)

    los_model = LOSModel(data['los_hours'], mode="empirical")
    print("  LOS model summary:", los_model.summary_stats())
    print()

    simulator = OccupancySimulator(model, los_model, data['capacity'])

    # Snapshot at day 36 (early surge, near capacity tipping point)
    snapshot_day = 36
    snapshot_hour = snapshot_day * 24 + 12
    print(f"  Forecasting from day {snapshot_day} (near capacity, surge onset)")

    # Find patients present at the snapshot time
    remaining_los = []
    for i in range(data['n_patients']):
        admit = data['admission_times_true'][i]
        discharge = data['discharge_times_true'][i]
        if admit <= snapshot_hour < discharge:
            remaining_los.append(discharge - snapshot_hour)
    current_patients = np.array(remaining_los) if remaining_los else np.array([0.0])

    print(f"  Current patients in ICU at snapshot: {len(current_patients)}")
    print(f"  Current occupancy: {len(current_patients)} / {data['capacity']}")
    print(f"  Mean remaining LOS: {np.mean(current_patients):.1f} hours")
    print()

    rng = np.random.default_rng(789)
    sim_result = simulator.simulate_trajectories(
        current_patients, forecast_hours=48,
        n_trajectories=3000, rng=rng
    )

    print(f"  Forecast horizon: 48 hours")
    print(f"  Trajectories simulated: 3000")
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


def step8_failure_analysis(daily_counts, data):
    """Step 8: Systematic failure-mode analysis."""
    print("=" * 70)
    print("STEP 8: Failure-mode analysis")
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
    print(f"Combined uncertainty widening factor: x{penalty:.2f}")
    print()

    return reports, penalty


def step9_visualizations(model, daily_counts, sim_result, data,
                          alpha_0, beta_0, w_history, sensitivity_histories):
    """Step 9: Generate all visualizations."""
    print("=" * 70)
    print("STEP 9: Generating visualizations (10 figures)")
    print("=" * 70)

    history = model.history

    # 1. Belief evolution (enhanced with KL + anomalies)
    plot_belief_evolution(
        history, data['surge_windows'],
        save_path=os.path.join(OUTPUT_DIR, "01_belief_evolution.png")
    )
    print("  [1/10] 01_belief_evolution.png")

    # 2. Posterior predictive check
    plot_posterior_predictive_check(
        model, daily_counts,
        save_path=os.path.join(OUTPUT_DIR, "02_posterior_predictive_check.png")
    )
    print("  [2/10] 02_posterior_predictive_check.png")

    # 3. Calibration (with windowed comparison)
    plot_calibration(
        model, daily_counts, windowed_history=w_history,
        save_path=os.path.join(OUTPUT_DIR, "03_calibration.png")
    )
    print("  [3/10] 03_calibration.png")

    # 4. Occupancy forecast
    plot_occupancy_forecast(
        sim_result,
        save_path=os.path.join(OUTPUT_DIR, "04_occupancy_forecast.png")
    )
    print("  [4/10] 04_occupancy_forecast.png")

    # 5. Model comparison: stationary vs. windowed
    plot_model_comparison(
        history, w_history, data['surge_windows'],
        save_path=os.path.join(OUTPUT_DIR, "05_model_comparison.png")
    )
    print("  [5/10] 05_model_comparison.png")

    # 6. Prior sensitivity analysis
    plot_prior_sensitivity(
        sensitivity_histories,
        save_path=os.path.join(OUTPUT_DIR, "06_prior_sensitivity.png")
    )
    print("  [6/10] 06_prior_sensitivity.png")

    # 7. Information gain + anomaly detection
    plot_information_gain(
        history, data['surge_windows'],
        save_path=os.path.join(OUTPUT_DIR, "07_information_gain.png")
    )
    print("  [7/10] 07_information_gain.png")

    # 8. LOS distribution
    plot_los_distribution(
        data['los_hours'],
        save_path=os.path.join(OUTPUT_DIR, "08_los_distribution.png")
    )
    print("  [8/10] 08_los_distribution.png")

    # 9. Prior vs posterior
    plot_prior_vs_posterior(
        model, alpha_0, beta_0,
        save_path=os.path.join(OUTPUT_DIR, "09_prior_vs_posterior.png")
    )
    print("  [9/10] 09_prior_vs_posterior.png")

    # 10. Full dashboard
    create_summary_dashboard(
        history, model, daily_counts, sim_result, data['los_hours'],
        data['surge_windows'], alpha_0, beta_0, windowed_history=w_history,
        save_path=os.path.join(OUTPUT_DIR, "10_dashboard.png")
    )
    print("  [10/10] 10_dashboard.png")
    print()


def step10_nl_output(model, crowd_result, reports, penalty, data_summary,
                      alpha_0, beta_0):
    """Step 10: Generate natural-language outputs and writeup."""
    print("=" * 70)
    print("STEP 10: Natural-language outputs")
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
    print("+" + "=" * 68 + "+")
    print("|  BUICU — Belief Updating for ICU Crowding Under Uncertainty        |")
    print("|  CS109 Challenge Project                                           |")
    print("+" + "=" * 68 + "+")
    print()

    # Core pipeline
    data, data_summary = step1_generate_data()
    daily_counts = step2_compute_daily_counts(data)
    model, alpha_0, beta_0, _ = step3_bayesian_updating(
        daily_counts, data['surge_windows']
    )

    # Advanced analyses
    _, w_history = step4_windowed_model(daily_counts, data['surge_windows'])
    _, sensitivity_histories = step5_prior_sensitivity(daily_counts)

    # Predictions
    step6_posterior_predictive(model)
    sim_result, crowd_result, los_model = step7_occupancy_simulation(model, data)

    # Failure analysis
    reports, penalty = step8_failure_analysis(daily_counts, data)

    # Outputs
    step9_visualizations(model, daily_counts, sim_result, data,
                          alpha_0, beta_0, w_history, sensitivity_histories)
    step10_nl_output(model, crowd_result, reports, penalty, data_summary,
                      alpha_0, beta_0)

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
