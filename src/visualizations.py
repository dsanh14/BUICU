"""
Visualizations for BUICU
=========================

Produces required and advanced visualizations:

  Required:
    1. Posterior belief evolution over time (with anomaly flags + KL divergence)
    2. Posterior predictive vs. empirical outcomes
    3. Uncertainty / calibration visualization

  Advanced (model comparison & Bayesian insights):
    4. Occupancy forecast with uncertainty fan
    5. Stationary vs. windowed model comparison
    6. Prior sensitivity analysis (convergence demonstration)
    7. KL divergence / information gain over time
    8. Anomaly detection timeline
    9. Prior → Posterior transformation
   10. Summary dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import List, Optional, Dict

from src.bayesian_model import BayesianArrivalModel, BeliefHistory


plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})

SURGE_COLOR = '#FFE0B2'
ANOMALY_COLOR = '#D32F2F'


def _shade_surges(ax, surge_windows, first_only_label=True):
    """Add surge-window shading to an axis."""
    for i, (s, e) in enumerate(surge_windows):
        label = 'Surge window' if (i == 0 and first_only_label) else ''
        ax.axvspan(s, e, alpha=0.12, color='orange', label=label)


def plot_belief_evolution(history: BeliefHistory, surge_windows: List[tuple],
                          save_path: Optional[str] = None):
    """
    REQUIRED VISUALIZATION 1: Posterior belief evolution over time.

    Enhanced with anomaly detection markers and KL-divergence subplot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                              height_ratios=[3, 1, 1], sharex=True)
    ax1, ax2, ax3 = axes

    times = np.array(history.times)
    means = np.array(history.means)
    ci_lows = np.array(history.ci_lows)
    ci_highs = np.array(history.ci_highs)
    obs = np.array(history.observed_counts)
    anomalies = np.array(history.anomaly_flags)
    kl_divs = np.array(history.kl_divergences)

    # --- Top: belief trajectory ---
    ax1.plot(times, means, 'b-', linewidth=1.5, label='Posterior mean E[λ|data]')
    ax1.fill_between(times, ci_lows, ci_highs, alpha=0.2, color='blue',
                     label='95% credible interval')
    ax1.scatter(times[1:], obs[1:], c='gray', s=12, alpha=0.4, zorder=4,
                label='Observed daily count')
    _shade_surges(ax1, surge_windows)

    # Highlight anomalous days
    anom_mask = anomalies[1:]
    if np.any(anom_mask):
        ax1.scatter(times[1:][anom_mask], obs[1:][anom_mask],
                    c=ANOMALY_COLOR, s=40, marker='x', linewidths=1.5,
                    zorder=6, label='Anomaly (outside 95% predictive)')

    ax1.set_ylabel('Arrival rate λ (admissions/day)')
    ax1.set_title('Posterior Belief Evolution: ICU Arrival Rate', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # --- Middle: CI width ---
    ci_width = ci_highs - ci_lows
    ax2.plot(times, ci_width, 'g-', linewidth=1.0)
    ax2.fill_between(times, 0, ci_width, alpha=0.2, color='green')
    ax2.set_ylabel('CI width')
    ax2.set_title('Posterior Uncertainty (credible interval width)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    _shade_surges(ax2, surge_windows, first_only_label=False)

    # --- Bottom: KL divergence (information gain) ---
    ax3.bar(times[1:], kl_divs[1:], width=0.8, color='purple', alpha=0.5)
    ax3.set_ylabel('KL divergence')
    ax3.set_xlabel('Time (days)')
    ax3.set_title('Information Gain Per Observation (bits)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    _shade_surges(ax3, surge_windows, first_only_label=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_posterior_predictive_check(model: BayesianArrivalModel,
                                    daily_counts: np.ndarray,
                                    save_path: Optional[str] = None):
    """
    REQUIRED VISUALIZATION 2: Posterior predictive vs. empirical outcomes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    k_vals, pmf = model.posterior_predictive_pmf(future_window=1.0, max_k=40)

    max_obs = int(np.max(daily_counts))
    bins = np.arange(-0.5, max(max_obs, 40) + 1.5, 1)
    ax1.hist(daily_counts, bins=bins, density=True, alpha=0.5, color='steelblue',
             edgecolor='white', label='Empirical (observed)')
    ax1.plot(k_vals, pmf, 'r-o', markersize=3, linewidth=1.5,
             label='Posterior predictive (NegBin)')
    ax1.set_xlabel('Daily admissions')
    ax1.set_ylabel('Probability')
    ax1.set_title('Posterior Predictive Check', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    empirical_sorted = np.sort(daily_counts)
    n = len(empirical_sorted)
    alpha = model.belief.alpha
    beta = model.belief.beta
    p_nb = beta / (beta + 1.0)
    theoretical_quantiles = np.array([
        stats.nbinom.ppf((i + 0.5) / n, n=alpha, p=p_nb) for i in range(n)
    ])

    ax2.scatter(theoretical_quantiles, empirical_sorted, s=10, alpha=0.5,
                color='steelblue')
    lims = [min(theoretical_quantiles.min(), empirical_sorted.min()) - 1,
            max(theoretical_quantiles.max(), empirical_sorted.max()) + 1]
    ax2.plot(lims, lims, 'r--', linewidth=1, label='Perfect calibration')
    ax2.set_xlabel('Theoretical quantiles (NegBin)')
    ax2.set_ylabel('Empirical quantiles')
    ax2.set_title('Q-Q Plot: Posterior Predictive vs. Observed', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_calibration(model: BayesianArrivalModel,
                     daily_counts: np.ndarray,
                     windowed_history: Optional[BeliefHistory] = None,
                     save_path: Optional[str] = None):
    """
    REQUIRED VISUALIZATION 3: Uncertainty calibration plot.

    Enhanced: compares calibration of stationary model vs. windowed model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax_cal, ax_pit, ax_comp = axes

    alpha = model.belief.alpha
    beta = model.belief.beta
    p_nb = beta / (beta + 1.0)

    # --- Calibration curve (stationary) ---
    nominal_levels = np.arange(0.05, 1.0, 0.05)
    emp_cov_stat = []
    for level in nominal_levels:
        lo = stats.nbinom.ppf((1 - level) / 2, n=alpha, p=p_nb)
        hi = stats.nbinom.ppf(1 - (1 - level) / 2, n=alpha, p=p_nb)
        emp_cov_stat.append(np.mean((daily_counts >= lo) & (daily_counts <= hi)))
    emp_cov_stat = np.array(emp_cov_stat)

    ax_cal.plot(nominal_levels, emp_cov_stat, 'bo-', markersize=4,
                label='Stationary model')

    # Windowed model calibration (if available)
    if windowed_history is not None:
        emp_cov_win = _windowed_calibration(windowed_history, daily_counts, nominal_levels)
        ax_cal.plot(nominal_levels, emp_cov_win, 'gs-', markersize=4,
                    label='Windowed model (14d)')

    ax_cal.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect')
    se = np.sqrt(nominal_levels * (1 - nominal_levels) / len(daily_counts))
    ax_cal.fill_between(nominal_levels, nominal_levels - 2 * se,
                        nominal_levels + 2 * se, alpha=0.12, color='red',
                        label='±2σ band')
    ax_cal.set_xlabel('Nominal coverage')
    ax_cal.set_ylabel('Empirical coverage')
    ax_cal.set_title('Calibration Comparison', fontweight='bold')
    ax_cal.legend(fontsize=7)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_aspect('equal')

    # --- PIT histogram (stationary) ---
    pit_values = stats.nbinom.cdf(daily_counts, n=alpha, p=p_nb)
    ax_pit.hist(pit_values, bins=20, density=True, alpha=0.6, color='steelblue',
                edgecolor='white')
    ax_pit.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Uniform (ideal)')
    ax_pit.set_xlabel('PIT value')
    ax_pit.set_ylabel('Density')
    ax_pit.set_title('PIT Histogram (stationary model)', fontweight='bold')
    ax_pit.legend()
    ax_pit.grid(True, alpha=0.3)

    # --- Windowed PIT (if available) ---
    if windowed_history is not None:
        pit_win = _windowed_pit(windowed_history, daily_counts)
        ax_comp.hist(pit_win, bins=20, density=True, alpha=0.6, color='green',
                     edgecolor='white')
        ax_comp.axhline(1.0, color='red', linestyle='--', linewidth=1,
                        label='Uniform (ideal)')
        ax_comp.set_title('PIT Histogram (windowed model)', fontweight='bold')
    else:
        ax_comp.text(0.5, 0.5, 'No windowed model', ha='center', va='center',
                     transform=ax_comp.transAxes)
        ax_comp.set_title('PIT Histogram (windowed model)', fontweight='bold')
    ax_comp.set_xlabel('PIT value')
    ax_comp.set_ylabel('Density')
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def _windowed_calibration(history: BeliefHistory, daily_counts: np.ndarray,
                           nominal_levels: np.ndarray) -> np.ndarray:
    """Compute calibration for windowed model using one-step-ahead predictions."""
    n = len(daily_counts)
    coverages = []
    for level in nominal_levels:
        inside = 0
        total = 0
        for t in range(1, min(n, len(history.alphas))):
            a, b = history.alphas[t - 1], history.betas[t - 1]
            p = b / (b + 1.0)
            lo = stats.nbinom.ppf((1 - level) / 2, n=a, p=p)
            hi = stats.nbinom.ppf(1 - (1 - level) / 2, n=a, p=p)
            if lo <= daily_counts[t] <= hi:
                inside += 1
            total += 1
        coverages.append(inside / max(total, 1))
    return np.array(coverages)


def _windowed_pit(history: BeliefHistory, daily_counts: np.ndarray) -> np.ndarray:
    """Compute PIT values for windowed model (one-step-ahead)."""
    n = len(daily_counts)
    pit = []
    for t in range(1, min(n, len(history.alphas))):
        a, b = history.alphas[t - 1], history.betas[t - 1]
        p = b / (b + 1.0)
        pit.append(stats.nbinom.cdf(daily_counts[t], n=a, p=p))
    return np.array(pit)


def plot_model_comparison(stationary_history: BeliefHistory,
                          windowed_history: BeliefHistory,
                          surge_windows: List[tuple],
                          save_path: Optional[str] = None):
    """
    ADVANCED: Stationary vs. windowed model comparison.

    Demonstrates how the windowed model tracks regime shifts while
    the stationary model averages over them.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    t_s = np.array(stationary_history.times)
    t_w = np.array(windowed_history.times)

    # Posterior means
    ax1.plot(t_s, stationary_history.means, 'b-', linewidth=1.5,
             label='Stationary model (all data)', alpha=0.8)
    ax1.fill_between(t_s, stationary_history.ci_lows, stationary_history.ci_highs,
                     alpha=0.1, color='blue')

    ax1.plot(t_w, windowed_history.means, 'g-', linewidth=1.5,
             label='Windowed model (14-day window)', alpha=0.8)
    ax1.fill_between(t_w, windowed_history.ci_lows, windowed_history.ci_highs,
                     alpha=0.1, color='green')

    ax1.scatter(t_s[1:], stationary_history.observed_counts[1:],
                c='gray', s=10, alpha=0.3, zorder=3)

    _shade_surges(ax1, surge_windows)
    ax1.set_ylabel('λ (admissions/day)')
    ax1.set_title('Model Comparison: Stationary vs. Adaptive (Windowed)',
                  fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # CI widths comparison
    ci_s = np.array(stationary_history.ci_highs) - np.array(stationary_history.ci_lows)
    ci_w = np.array(windowed_history.ci_highs) - np.array(windowed_history.ci_lows)
    ax2.plot(t_s, ci_s, 'b-', label='Stationary CI width')
    ax2.plot(t_w, ci_w, 'g-', label='Windowed CI width')
    ax2.set_ylabel('CI width')
    ax2.set_xlabel('Time (days)')
    ax2.set_title('Uncertainty Comparison', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _shade_surges(ax2, surge_windows, first_only_label=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_prior_sensitivity(histories: Dict[str, BeliefHistory],
                            save_path: Optional[str] = None):
    """
    ADVANCED: Prior sensitivity analysis.

    Shows three different priors converging to similar posteriors,
    demonstrating that evidence overwhelms prior beliefs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'Uninformative': '#E53935', 'Weakly informative': '#1E88E5',
              'Strong (wrong center)': '#43A047'}

    for name, hist in histories.items():
        times = np.array(hist.times)
        means = np.array(hist.means)
        ax1.plot(times, means, '-', color=colors.get(name, 'gray'),
                 linewidth=1.5, label=name)
        ax1.fill_between(times, hist.ci_lows, hist.ci_highs,
                         alpha=0.08, color=colors.get(name, 'gray'))

    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Posterior mean E[λ|data]')
    ax1.set_title('Prior Sensitivity: Convergence of Posteriors', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final posterior comparison
    x = np.linspace(0, 25, 500)
    for name, hist in histories.items():
        a, b = hist.alphas[-1], hist.betas[-1]
        pdf = stats.gamma.pdf(x, a=a, scale=1.0 / b)
        ax2.plot(x, pdf, '-', color=colors.get(name, 'gray'), linewidth=2,
                 label=f'{name}\nα={a:.0f}, β={b:.1f}, μ={a / b:.2f}')

    ax2.set_xlabel('λ (admissions/day)')
    ax2.set_ylabel('Density')
    ax2.set_title('Final Posteriors (day 180)', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_information_gain(history: BeliefHistory, surge_windows: List[tuple],
                          save_path: Optional[str] = None):
    """
    ADVANCED: KL divergence (information gain) over time.

    Shows how much each observation changes our beliefs. High KL during
    surges reveals that the stationary model is being surprised.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    times = np.array(history.times[1:])
    kl = np.array(history.kl_divergences[1:])
    pvals = np.array(history.predictive_pvalues[1:])
    anomalies = np.array(history.anomaly_flags[1:])

    # KL divergence
    colors_kl = ['purple' if not a else ANOMALY_COLOR for a in anomalies]
    ax1.bar(times, kl, width=0.8, color=colors_kl, alpha=0.6)
    ax1.set_ylabel('KL(posterior || prior)')
    ax1.set_title('Information Gain Per Observation', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    _shade_surges(ax1, surge_windows)

    cum_kl = np.cumsum(kl)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, cum_kl, 'k--', linewidth=1.0, alpha=0.5,
                  label='Cumulative KL')
    ax1_twin.set_ylabel('Cumulative KL', color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')

    # Predictive p-values (anomaly detection)
    ax2.scatter(times[~anomalies], pvals[~anomalies], c='steelblue', s=12,
                alpha=0.4, label='Normal')
    ax2.scatter(times[anomalies], pvals[anomalies], c=ANOMALY_COLOR, s=35,
                marker='x', linewidths=1.5, zorder=5,
                label=f'Anomaly ({int(np.sum(anomalies))} days)')
    ax2.axhline(0.025, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(0.975, color='red', linestyle=':', alpha=0.5)
    ax2.fill_between([times[0], times[-1]], 0, 0.025, alpha=0.05, color='red')
    ax2.fill_between([times[0], times[-1]], 0.975, 1.0, alpha=0.05, color='red')
    ax2.set_ylabel('Predictive p-value')
    ax2.set_xlabel('Time (days)')
    ax2.set_title('Anomaly Detection via Posterior Predictive P-values',
                  fontweight='bold')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    _shade_surges(ax2, surge_windows, first_only_label=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_occupancy_forecast(sim_result: dict, save_path: Optional[str] = None):
    """Occupancy forecast with uncertainty fan chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                                     sharex=True)

    t = sim_result['time_grid']
    cap = sim_result['capacity']

    ax1.fill_between(t, sim_result['ci_low'], sim_result['ci_high'],
                     alpha=0.15, color='blue', label='95% CI')

    p25 = np.percentile(sim_result['trajectories'], 25, axis=0)
    p75 = np.percentile(sim_result['trajectories'], 75, axis=0)
    ax1.fill_between(t, p25, p75, alpha=0.3, color='blue', label='50% CI')

    ax1.plot(t, sim_result['mean'], 'b-', linewidth=1.5, label='Mean')
    ax1.plot(t, sim_result['median'], 'b--', linewidth=1.0, label='Median')
    ax1.axhline(cap, color='red', linestyle=':', linewidth=2,
                label=f'Capacity ({cap})')
    ax1.set_ylabel('ICU Occupancy')
    ax1.set_title('Occupancy Forecast with Uncertainty', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(t, 0, sim_result['p_overcrowded'], alpha=0.4, color='red')
    ax2.plot(t, sim_result['p_overcrowded'], 'r-', linewidth=1.0)
    ax2.set_ylabel('P(overcrowded)')
    ax2.set_xlabel('Hours from now')
    ax2.set_title('Probability of Exceeding Capacity Over Time')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_los_distribution(los_hours: np.ndarray, save_path: Optional[str] = None):
    """Length-of-stay distribution with tail analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    los_days = los_hours[~np.isnan(los_hours)] / 24.0

    ax1.hist(los_days, bins=80, density=True, alpha=0.6, color='steelblue',
             edgecolor='white', range=(0, np.percentile(los_days, 99)))
    ax1.axvline(np.median(los_days), color='red', linestyle='--',
                label=f'Median = {np.median(los_days):.1f}d')
    ax1.axvline(np.mean(los_days), color='orange', linestyle='--',
                label=f'Mean = {np.mean(los_days):.1f}d')
    ax1.set_xlabel('Length of Stay (days)')
    ax1.set_ylabel('Density')
    ax1.set_title('ICU Length-of-Stay Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(los_days, bins=100, density=True, alpha=0.6, color='steelblue',
             edgecolor='white')
    ax2.set_yscale('log')
    ax2.axvline(np.percentile(los_days, 99), color='red', linestyle=':',
                label=f'p99 = {np.percentile(los_days, 99):.1f}d')
    ax2.axvline(np.percentile(los_days, 99.9), color='darkred', linestyle=':',
                label=f'p99.9 = {np.percentile(los_days, 99.9):.1f}d')
    ax2.set_xlabel('Length of Stay (days)')
    ax2.set_ylabel('Density (log scale)')
    ax2.set_title('LOS Heavy-Tail Analysis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_prior_vs_posterior(model: BayesianArrivalModel,
                            alpha_0: float, beta_0: float,
                            save_path: Optional[str] = None):
    """Prior vs. posterior comparison for λ."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(0, 25, 500)
    prior_pdf = stats.gamma.pdf(x, a=alpha_0, scale=1.0 / beta_0)
    post_pdf = stats.gamma.pdf(x, a=model.belief.alpha,
                                scale=1.0 / model.belief.beta)

    ax.plot(x, prior_pdf, 'r--', linewidth=2,
            label=f'Prior: Gamma({alpha_0}, {beta_0})\n'
                  f'  mean={alpha_0 / beta_0:.1f}, '
                  f'std={np.sqrt(alpha_0) / beta_0:.1f}')
    ax.fill_between(x, prior_pdf, alpha=0.1, color='red')

    ax.plot(x, post_pdf, 'b-', linewidth=2,
            label=f'Posterior: Gamma({model.belief.alpha:.0f}, '
                  f'{model.belief.beta:.1f})\n'
                  f'  mean={model.belief.mean:.2f}, '
                  f'std={model.belief.std:.3f}')
    ax.fill_between(x, post_pdf, alpha=0.2, color='blue')

    ax.set_xlabel('λ (admissions/day)')
    ax.set_ylabel('Density')
    ax.set_title('Prior → Posterior: Bayesian Update of Arrival Rate',
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def create_summary_dashboard(history: BeliefHistory,
                              model: BayesianArrivalModel,
                              daily_counts: np.ndarray,
                              sim_result: dict,
                              los_hours: np.ndarray,
                              surge_windows: List[tuple],
                              alpha_0: float, beta_0: float,
                              windowed_history: Optional[BeliefHistory] = None,
                              save_path: Optional[str] = None):
    """Combined dashboard with all key results."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.3)

    # (0,0)-(0,1): Belief evolution
    ax1 = fig.add_subplot(gs[0, :2])
    times = np.array(history.times)
    ax1.plot(times, history.means, 'b-', linewidth=1.5, label='Stationary')
    ax1.fill_between(times, history.ci_lows, history.ci_highs, alpha=0.15, color='blue')
    if windowed_history:
        t_w = np.array(windowed_history.times)
        ax1.plot(t_w, windowed_history.means, 'g-', linewidth=1.2,
                 label='Windowed (14d)', alpha=0.8)
    ax1.scatter(times[1:], history.observed_counts[1:], c='gray', s=8, alpha=0.3)
    anomalies = np.array(history.anomaly_flags[1:])
    obs = np.array(history.observed_counts[1:])
    if np.any(anomalies):
        ax1.scatter(times[1:][anomalies], obs[anomalies],
                    c=ANOMALY_COLOR, s=25, marker='x', linewidths=1.2, zorder=5)
    _shade_surges(ax1, surge_windows)
    ax1.set_title('Posterior Belief Evolution (+ anomalies)', fontweight='bold',
                  fontsize=10)
    ax1.set_ylabel('λ (adm/day)')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # (0,2): Prior vs posterior
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.linspace(0, 25, 300)
    ax2.plot(x, stats.gamma.pdf(x, a=alpha_0, scale=1.0 / beta_0), 'r--',
             label='Prior')
    ax2.plot(x, stats.gamma.pdf(x, a=model.belief.alpha,
                                 scale=1.0 / model.belief.beta),
             'b-', label='Posterior')
    ax2.set_title('Prior → Posterior', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # (1,0)-(1,1): Occupancy forecast
    ax3 = fig.add_subplot(gs[1, :2])
    t = sim_result['time_grid']
    ax3.fill_between(t, sim_result['ci_low'], sim_result['ci_high'],
                     alpha=0.15, color='blue', label='95% CI')
    ax3.plot(t, sim_result['mean'], 'b-', linewidth=1.5, label='Mean')
    ax3.axhline(sim_result['capacity'], color='red', linestyle=':', linewidth=2,
                label=f"Capacity ({sim_result['capacity']})")
    ax3.set_title('48h Occupancy Forecast', fontweight='bold', fontsize=10)
    ax3.set_ylabel('Occupancy')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # (1,2): P(overcrowded)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.fill_between(t, 0, sim_result['p_overcrowded'], alpha=0.4, color='red')
    ax4.set_title('P(overcrowded)', fontweight='bold', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # (2,0): Posterior predictive check
    ax5 = fig.add_subplot(gs[2, 0])
    k_vals, pmf = model.posterior_predictive_pmf(1.0, max_k=40)
    bins = np.arange(-0.5, 41.5, 1)
    ax5.hist(daily_counts, bins=bins, density=True, alpha=0.5, color='steelblue')
    ax5.plot(k_vals, pmf, 'r-o', markersize=2, linewidth=1)
    ax5.set_title('Posterior Predictive Check', fontweight='bold', fontsize=10)
    ax5.set_xlabel('Daily admissions')
    ax5.grid(True, alpha=0.3)

    # (2,1): Calibration
    ax6 = fig.add_subplot(gs[2, 1])
    alpha_post = model.belief.alpha
    beta_post = model.belief.beta
    p_nb = beta_post / (beta_post + 1.0)
    nominal_levels = np.arange(0.05, 1.0, 0.05)
    emp_cov = [np.mean((daily_counts >= stats.nbinom.ppf((1 - l) / 2, n=alpha_post, p=p_nb)) &
               (daily_counts <= stats.nbinom.ppf(1 - (1 - l) / 2, n=alpha_post, p=p_nb)))
               for l in nominal_levels]
    ax6.plot(nominal_levels, emp_cov, 'bo-', markersize=3, label='Stationary')
    if windowed_history:
        emp_cov_w = _windowed_calibration(windowed_history, daily_counts, nominal_levels)
        ax6.plot(nominal_levels, emp_cov_w, 'gs-', markersize=3, label='Windowed')
    ax6.plot([0, 1], [0, 1], 'r--')
    ax6.set_title('Calibration', fontweight='bold', fontsize=10)
    ax6.set_xlabel('Nominal')
    ax6.set_ylabel('Empirical')
    ax6.legend(fontsize=7)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    # (2,2): LOS distribution
    ax7 = fig.add_subplot(gs[2, 2])
    los_clean = los_hours[~np.isnan(los_hours)] / 24.0
    ax7.hist(los_clean, bins=60, density=True, alpha=0.6, color='steelblue',
             range=(0, np.percentile(los_clean, 99)))
    ax7.axvline(np.median(los_clean), color='red', linestyle='--',
                label=f'med={np.median(los_clean):.1f}d')
    ax7.set_title('LOS Distribution', fontweight='bold', fontsize=10)
    ax7.set_xlabel('Days')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    # (3,0): KL divergence
    ax8 = fig.add_subplot(gs[3, 0])
    kl = np.array(history.kl_divergences[1:])
    anom = np.array(history.anomaly_flags[1:])
    c_kl = ['purple' if not a else ANOMALY_COLOR for a in anom]
    ax8.bar(times[1:], kl, width=0.8, color=c_kl, alpha=0.5)
    ax8.set_title('Information Gain (KL)', fontweight='bold', fontsize=10)
    ax8.set_xlabel('Day')
    ax8.grid(True, alpha=0.3)

    # (3,1): Anomaly p-values
    ax9 = fig.add_subplot(gs[3, 1])
    pvals = np.array(history.predictive_pvalues[1:])
    ax9.scatter(times[1:][~anom], pvals[~anom], c='steelblue', s=8, alpha=0.4)
    if np.any(anom):
        ax9.scatter(times[1:][anom], pvals[anom], c=ANOMALY_COLOR, s=25,
                    marker='x', linewidths=1.2, zorder=5)
    ax9.axhline(0.025, color='red', linestyle=':', alpha=0.5)
    ax9.axhline(0.975, color='red', linestyle=':', alpha=0.5)
    ax9.set_title('Anomaly P-values', fontweight='bold', fontsize=10)
    ax9.set_xlabel('Day')
    ax9.set_ylim(-0.02, 1.02)
    ax9.grid(True, alpha=0.3)

    # (3,2): Prior sensitivity (final posteriors)
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.text(0.5, 0.5, 'See prior\nsensitivity plot', ha='center', va='center',
              transform=ax10.transAxes, fontsize=11, color='gray')
    ax10.set_title('Prior Sensitivity', fontweight='bold', fontsize=10)
    ax10.grid(True, alpha=0.3)

    fig.suptitle('BUICU: Bayesian ICU Crowding Under Uncertainty — Full Dashboard',
                 fontsize=14, fontweight='bold', y=1.01)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig
