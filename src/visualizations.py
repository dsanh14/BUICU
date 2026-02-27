"""
Visualizations for BUICU
=========================

Produces three required visualizations:
  1. Posterior belief evolution over time
  2. Posterior predictive vs. empirical outcomes
  3. Uncertainty calibration

Plus additional supporting plots:
  4. Occupancy forecast with uncertainty fan
  5. LOS distribution with tail analysis
  6. Crowding probability over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import List, Optional

from src.bayesian_model import BayesianArrivalModel, BeliefHistory


plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})


def plot_belief_evolution(history: BeliefHistory, surge_windows: List[tuple],
                          save_path: Optional[str] = None):
    """
    REQUIRED VISUALIZATION 1: Posterior belief evolution over time.

    Shows:
      - Posterior mean of λ at each update step
      - 95% credible interval band
      - Observed daily counts (scatter)
      - Surge window annotations
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                                     sharex=True)

    times = np.array(history.times)
    means = np.array(history.means)
    ci_lows = np.array(history.ci_lows)
    ci_highs = np.array(history.ci_highs)
    obs = np.array(history.observed_counts)

    # Posterior mean and CI
    ax1.plot(times, means, 'b-', linewidth=1.5, label='Posterior mean E[λ|data]')
    ax1.fill_between(times, ci_lows, ci_highs, alpha=0.2, color='blue',
                     label='95% credible interval')

    # Observed counts
    ax1.scatter(times[1:], obs[1:], c='red', s=15, alpha=0.6, zorder=5,
                label='Observed daily count')

    # Surge window shading
    for s_start, s_end in surge_windows:
        ax1.axvspan(s_start, s_end, alpha=0.1, color='orange',
                    label='Surge window' if s_start == surge_windows[0][0] else '')

    ax1.set_ylabel('Arrival rate λ (admissions/day)')
    ax1.set_title('Posterior Belief Evolution: ICU Arrival Rate')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # CI width over time (shows how uncertainty decreases)
    ci_width = np.array(ci_highs) - np.array(ci_lows)
    ax2.plot(times, ci_width, 'g-', linewidth=1.0)
    ax2.fill_between(times, 0, ci_width, alpha=0.2, color='green')
    ax2.set_ylabel('CI width')
    ax2.set_xlabel('Time (days)')
    ax2.set_title('Posterior Uncertainty (credible interval width)')
    ax2.grid(True, alpha=0.3)

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

    Overlays the posterior predictive PMF against the empirical histogram
    of daily admission counts. Good calibration means these should agree.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    k_vals, pmf = model.posterior_predictive_pmf(future_window=1.0, max_k=40)

    # Histogram of observed counts
    max_obs = int(np.max(daily_counts))
    bins = np.arange(-0.5, max(max_obs, 40) + 1.5, 1)
    ax1.hist(daily_counts, bins=bins, density=True, alpha=0.5, color='steelblue',
             edgecolor='white', label='Empirical (observed)')
    ax1.plot(k_vals, pmf, 'r-o', markersize=3, linewidth=1.5,
             label='Posterior predictive (NegBin)')
    ax1.set_xlabel('Daily admissions')
    ax1.set_ylabel('Probability')
    ax1.set_title('Posterior Predictive Check')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q style: empirical quantiles vs predictive quantiles
    empirical_sorted = np.sort(daily_counts)
    n = len(empirical_sorted)
    theoretical_quantiles = []
    alpha = model.belief.alpha
    beta = model.belief.beta
    p_nb = beta / (beta + 1.0)
    for i in range(n):
        q = (i + 0.5) / n
        theoretical_quantiles.append(stats.nbinom.ppf(q, n=alpha, p=p_nb))
    theoretical_quantiles = np.array(theoretical_quantiles)

    ax2.scatter(theoretical_quantiles, empirical_sorted, s=10, alpha=0.5,
                color='steelblue')
    lims = [min(theoretical_quantiles.min(), empirical_sorted.min()) - 1,
            max(theoretical_quantiles.max(), empirical_sorted.max()) + 1]
    ax2.plot(lims, lims, 'r--', linewidth=1, label='Perfect calibration')
    ax2.set_xlabel('Theoretical quantiles (NegBin)')
    ax2.set_ylabel('Empirical quantiles')
    ax2.set_title('Q–Q Plot: Posterior Predictive vs. Observed')
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
                     save_path: Optional[str] = None):
    """
    REQUIRED VISUALIZATION 3: Uncertainty calibration plot.

    For a well-calibrated model, x% of observations should fall within
    the x% predictive interval. We check this across multiple levels.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    alpha = model.belief.alpha
    beta = model.belief.beta
    p_nb = beta / (beta + 1.0)

    # Calibration curve
    nominal_levels = np.arange(0.05, 1.0, 0.05)
    empirical_coverage = []
    for level in nominal_levels:
        lo = stats.nbinom.ppf((1 - level) / 2, n=alpha, p=p_nb)
        hi = stats.nbinom.ppf(1 - (1 - level) / 2, n=alpha, p=p_nb)
        coverage = np.mean((daily_counts >= lo) & (daily_counts <= hi))
        empirical_coverage.append(coverage)
    empirical_coverage = np.array(empirical_coverage)

    ax1.plot(nominal_levels, empirical_coverage, 'bo-', markersize=4,
             label='Model calibration')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect calibration')
    ax1.fill_between(nominal_levels,
                     nominal_levels - 2 * np.sqrt(nominal_levels * (1 - nominal_levels) / len(daily_counts)),
                     nominal_levels + 2 * np.sqrt(nominal_levels * (1 - nominal_levels) / len(daily_counts)),
                     alpha=0.15, color='red', label='±2σ (finite sample)')
    ax1.set_xlabel('Nominal coverage level')
    ax1.set_ylabel('Empirical coverage')
    ax1.set_title('Calibration Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')

    # PIT histogram (probability integral transform)
    pit_values = stats.nbinom.cdf(daily_counts, n=alpha, p=p_nb)
    ax2.hist(pit_values, bins=20, density=True, alpha=0.6, color='steelblue',
             edgecolor='white')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1,
                label='Uniform (ideal)')
    ax2.set_xlabel('PIT value')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Integral Transform (PIT) Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_occupancy_forecast(sim_result: dict, save_path: Optional[str] = None):
    """
    Occupancy forecast with uncertainty fan chart.

    Shows Monte Carlo mean, median, 50% CI, 95% CI, and capacity line.
    """
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
    ax1.axhline(cap, color='red', linestyle=':', linewidth=2, label=f'Capacity ({cap})')
    ax1.set_ylabel('ICU Occupancy')
    ax1.set_title('Occupancy Forecast with Uncertainty')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # P(overcrowded) over time
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
    """
    Length-of-stay distribution with tail analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    los_days = los_hours[~np.isnan(los_hours)] / 24.0

    # Main distribution (truncated for readability)
    ax1.hist(los_days, bins=80, density=True, alpha=0.6, color='steelblue',
             edgecolor='white', range=(0, np.percentile(los_days, 99)))
    ax1.axvline(np.median(los_days), color='red', linestyle='--',
                label=f'Median = {np.median(los_days):.1f}d')
    ax1.axvline(np.mean(los_days), color='orange', linestyle='--',
                label=f'Mean = {np.mean(los_days):.1f}d')
    ax1.set_xlabel('Length of Stay (days)')
    ax1.set_ylabel('Density')
    ax1.set_title('ICU Length-of-Stay Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-scale tail view
    ax2.hist(los_days, bins=100, density=True, alpha=0.6, color='steelblue',
             edgecolor='white')
    ax2.set_yscale('log')
    ax2.axvline(np.percentile(los_days, 99), color='red', linestyle=':',
                label=f'p99 = {np.percentile(los_days, 99):.1f}d')
    ax2.axvline(np.percentile(los_days, 99.9), color='darkred', linestyle=':',
                label=f'p99.9 = {np.percentile(los_days, 99.9):.1f}d')
    ax2.set_xlabel('Length of Stay (days)')
    ax2.set_ylabel('Density (log scale)')
    ax2.set_title('LOS Heavy-Tail Analysis')
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
    """
    Side-by-side comparison of prior and posterior distributions for λ.

    This directly illustrates the Bayesian update: how data transforms
    our beliefs from a vague prior into a concentrated posterior.
    """
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
    ax.set_title('Prior → Posterior: Bayesian Update of Arrival Rate')
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
                              save_path: Optional[str] = None):
    """
    Combined dashboard showing all key results in a single figure.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # (0,0)-(0,1): Belief evolution
    ax1 = fig.add_subplot(gs[0, :2])
    times = np.array(history.times)
    ax1.plot(times, history.means, 'b-', linewidth=1.5)
    ax1.fill_between(times, history.ci_lows, history.ci_highs, alpha=0.2, color='blue')
    ax1.scatter(times[1:], history.observed_counts[1:], c='red', s=10, alpha=0.5)
    for s_start, s_end in surge_windows:
        ax1.axvspan(s_start, s_end, alpha=0.08, color='orange')
    ax1.set_title('Posterior Belief Evolution')
    ax1.set_ylabel('λ (adm/day)')
    ax1.grid(True, alpha=0.3)

    # (0,2): Prior vs posterior
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.linspace(0, 25, 300)
    ax2.plot(x, stats.gamma.pdf(x, a=alpha_0, scale=1.0 / beta_0), 'r--', label='Prior')
    ax2.plot(x, stats.gamma.pdf(x, a=model.belief.alpha, scale=1.0 / model.belief.beta),
             'b-', label='Posterior')
    ax2.set_title('Prior → Posterior')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # (1,0)-(1,1): Occupancy forecast
    ax3 = fig.add_subplot(gs[1, :2])
    t = sim_result['time_grid']
    ax3.fill_between(t, sim_result['ci_low'], sim_result['ci_high'], alpha=0.15, color='blue')
    ax3.plot(t, sim_result['mean'], 'b-', linewidth=1.5)
    ax3.axhline(sim_result['capacity'], color='red', linestyle=':', linewidth=2)
    ax3.set_title('48h Occupancy Forecast')
    ax3.set_ylabel('Occupancy')
    ax3.grid(True, alpha=0.3)

    # (1,2): P(overcrowded)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.fill_between(t, 0, sim_result['p_overcrowded'], alpha=0.4, color='red')
    ax4.set_title('P(overcrowded)')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # (2,0): Posterior predictive check
    ax5 = fig.add_subplot(gs[2, 0])
    k_vals, pmf = model.posterior_predictive_pmf(1.0, max_k=40)
    bins = np.arange(-0.5, 41.5, 1)
    ax5.hist(daily_counts, bins=bins, density=True, alpha=0.5, color='steelblue')
    ax5.plot(k_vals, pmf, 'r-o', markersize=2, linewidth=1)
    ax5.set_title('Posterior Predictive Check')
    ax5.set_xlabel('Daily admissions')
    ax5.grid(True, alpha=0.3)

    # (2,1): Calibration
    ax6 = fig.add_subplot(gs[2, 1])
    alpha_post = model.belief.alpha
    beta_post = model.belief.beta
    p_nb = beta_post / (beta_post + 1.0)
    nominal_levels = np.arange(0.05, 1.0, 0.05)
    empirical_coverage = []
    for level in nominal_levels:
        lo = stats.nbinom.ppf((1 - level) / 2, n=alpha_post, p=p_nb)
        hi = stats.nbinom.ppf(1 - (1 - level) / 2, n=alpha_post, p=p_nb)
        coverage = np.mean((daily_counts >= lo) & (daily_counts <= hi))
        empirical_coverage.append(coverage)
    ax6.plot(nominal_levels, empirical_coverage, 'bo-', markersize=3)
    ax6.plot([0, 1], [0, 1], 'r--')
    ax6.set_title('Calibration')
    ax6.set_xlabel('Nominal')
    ax6.set_ylabel('Empirical')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    # (2,2): LOS distribution
    ax7 = fig.add_subplot(gs[2, 2])
    los_clean = los_hours[~np.isnan(los_hours)] / 24.0
    ax7.hist(los_clean, bins=60, density=True, alpha=0.6, color='steelblue',
             range=(0, np.percentile(los_clean, 99)))
    ax7.axvline(np.median(los_clean), color='red', linestyle='--',
                label=f'med={np.median(los_clean):.1f}d')
    ax7.set_title('LOS Distribution')
    ax7.set_xlabel('Days')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    fig.suptitle('BUICU: Bayesian ICU Crowding Under Uncertainty — Dashboard',
                 fontsize=14, fontweight='bold', y=1.01)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig
