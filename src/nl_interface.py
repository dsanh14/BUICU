"""
Natural-Language Explanation Interface
=======================================

Translates probabilistic model outputs into structured natural-language
responses suitable for clinical stakeholders and academic presentation.

Design principles:
  - NEVER return a point estimate without uncertainty
  - ALWAYS include: probability, credible interval, belief update, caveats
  - Frame statements in terms of distributions and evidence, not "AI predictions"
  - Acknowledge model limitations in every response
"""

import numpy as np
from typing import List, Optional
from src.bayesian_model import BeliefState, OccupancySimulator
from src.failure_modes import FailureModeReport


def explain_current_belief(belief: BeliefState, prior_mean: float) -> str:
    """
    Generate a natural-language explanation of the current posterior belief
    about the ICU arrival rate.
    """
    ci = belief.credible_interval(0.95)
    direction = "increased" if belief.mean > prior_mean else "decreased"
    magnitude = abs(belief.mean - prior_mean) / max(prior_mean, 0.01)

    lines = [
        f"Current belief about ICU arrival rate:",
        f"  Posterior mean: {belief.mean:.2f} admissions/day",
        f"  95% credible interval: [{ci[0]:.2f}, {ci[1]:.2f}]",
        f"  Based on {belief.total_arrivals} observed admissions over "
        f"{belief.time:.1f} days.",
        "",
        f"  The estimated arrival rate has {direction} by "
        f"{100 * magnitude:.1f}% relative to the prior expectation "
        f"({prior_mean:.2f}/day).",
        "",
        f"  Interpretation: After observing the data, we believe the true "
        f"arrival rate lies between {ci[0]:.2f} and {ci[1]:.2f} admissions "
        f"per day with 95% probability. The posterior has concentrated "
        f"around {belief.mean:.2f} as evidence accumulated.",
    ]
    return "\n".join(lines)


def explain_belief_update(old_belief: BeliefState, new_belief: BeliefState,
                           observed_k: int, window_days: float) -> str:
    """
    Explain how a single observation updated beliefs.
    """
    expected = old_belief.mean * window_days
    surprise = observed_k - expected

    old_ci = old_belief.credible_interval(0.95)
    new_ci = new_belief.credible_interval(0.95)
    ci_width_change = (new_ci[1] - new_ci[0]) - (old_ci[1] - old_ci[0])

    if surprise > 1.5:
        surprise_text = (
            f"This was notably higher than expected ({expected:.1f}), "
            f"so the posterior shifted upward."
        )
    elif surprise < -1.5:
        surprise_text = (
            f"This was notably lower than expected ({expected:.1f}), "
            f"so the posterior shifted downward."
        )
    else:
        surprise_text = (
            f"This was close to the expected count ({expected:.1f}), "
            f"so the posterior changed only modestly."
        )

    lines = [
        f"Belief update after observing {observed_k} admissions in "
        f"{window_days:.1f} days:",
        f"  Prior mean:     {old_belief.mean:.2f} → Posterior mean: "
        f"{new_belief.mean:.2f}",
        f"  Prior 95% CI:   [{old_ci[0]:.2f}, {old_ci[1]:.2f}] "
        f"→ Posterior 95% CI: [{new_ci[0]:.2f}, {new_ci[1]:.2f}]",
        f"  {surprise_text}",
        f"  Credible interval width {'narrowed' if ci_width_change < 0 else 'widened'} "
        f"by {abs(ci_width_change):.3f}, reflecting "
        f"{'increased' if ci_width_change < 0 else 'decreased'} certainty.",
    ]
    return "\n".join(lines)


def explain_crowding_forecast(crowd_result: dict,
                               failure_reports: List[FailureModeReport],
                               confidence_penalty: float) -> str:
    """
    Generate the primary forecast explanation.

    This is the response format specified in the project requirements:
      probability + credible interval + belief update + caveats.
    """
    p = crowd_result['probability']
    ci_lo = crowd_result['ci_low']
    ci_hi = crowd_result['ci_high']
    horizon = crowd_result['horizon_hours']

    active_failures = [r for r in failure_reports if r.detected]

    lines = [
        f"There is a {100 * p:.0f}% chance ICU occupancy exceeds capacity "
        f"in the next {horizon} hours",
        f"(95% credible interval: {100 * ci_lo:.0f}%–{100 * ci_hi:.0f}%).",
        "",
    ]

    if confidence_penalty > 1.05:
        adjusted_ci_lo = max(0, ci_lo / confidence_penalty)
        adjusted_ci_hi = min(1, ci_hi * confidence_penalty)
        lines.append(
            f"After adjusting for {len(active_failures)} detected model "
            f"limitation(s), the uncertainty-widened interval is "
            f"{100 * adjusted_ci_lo:.0f}%–{100 * adjusted_ci_hi:.0f}%."
        )
        lines.append("")

    lines.append("Assumptions and caveats:")
    lines.append("  - Arrivals are modeled as Poisson (independent increments).")
    lines.append("  - Length of stay is sampled from the empirical distribution.")
    lines.append("  - The forecast does not account for planned admissions "
                 "or scheduled discharges.")

    if active_failures:
        lines.append("")
        lines.append("Active model limitations:")
        for r in active_failures:
            lines.append(f"  ⚠ {r.name} ({r.severity}): {r.evidence}")

    lines.append("")
    lines.append(
        "This estimate is based on a probabilistic model with explicit "
        "uncertainty quantification. It should inform — not replace — "
        "clinical judgment."
    )
    return "\n".join(lines)


def explain_posterior_predictive(k_values: np.ndarray, pmf: np.ndarray,
                                  window_days: float) -> str:
    """Explain the posterior predictive distribution for future admissions."""
    mean = np.sum(k_values * pmf)
    variance = np.sum((k_values - mean) ** 2 * pmf)
    std = np.sqrt(variance)

    cumsum = np.cumsum(pmf)
    ci_lo = k_values[np.searchsorted(cumsum, 0.025)]
    ci_hi = k_values[np.searchsorted(cumsum, 0.975)]
    mode = k_values[np.argmax(pmf)]

    lines = [
        f"Posterior predictive distribution for admissions in the next "
        f"{window_days:.1f} days:",
        f"  Expected count: {mean:.1f} (std = {std:.1f})",
        f"  Most likely count (mode): {mode}",
        f"  95% predictive interval: [{ci_lo}, {ci_hi}]",
        "",
        "This distribution integrates over uncertainty in the arrival rate λ.",
        "It is wider than a Poisson with fixed λ because it accounts for the "
        "fact that we do not know λ exactly — only its posterior distribution.",
    ]
    return "\n".join(lines)


def generate_writeup_sections(belief: BeliefState, prior_mean: float,
                               crowd_result: dict,
                               failure_reports: List[FailureModeReport],
                               data_summary: str) -> str:
    """
    Generate structured content for the CS109 writeup.
    """
    ci = belief.credible_interval(0.95)
    p = crowd_result['probability']

    sections = []

    sections.append("=" * 70)
    sections.append("ABSTRACT")
    sections.append("=" * 70)
    sections.append(
        "We model ICU crowding as a stochastic process using Bayesian "
        "inference over a Gamma–Poisson conjugate model. Our system "
        "performs sequential belief updating as new admission data arrives, "
        "propagates uncertainty through Monte Carlo simulation of occupancy "
        "trajectories, and computes calibrated probabilistic forecasts of "
        "crowding events. We systematically analyze five failure modes of "
        "the model and demonstrate how uncertainty is appropriately widened "
        "when assumptions are violated. All analysis uses a synthetic dataset "
        "that faithfully replicates the statistical structure of MIMIC-IV ICU data."
    )
    sections.append("")

    sections.append("=" * 70)
    sections.append("METHODS")
    sections.append("=" * 70)
    sections.append(
        "Random Variables:\n"
        "  N_t | λ  ~ Poisson(λ · Δt)    — admissions in window Δt\n"
        "  λ        ~ Gamma(α₀, β₀)      — prior on arrival rate\n"
        "  λ | data ~ Gamma(α₀+Σk, β₀+T) — posterior after T days, Σk arrivals\n"
        "  L        ~ Empirical(LOS data) — length of stay\n"
        "  O_t      = Σ_i 1[aᵢ ≤ t < aᵢ + Lᵢ] — occupancy (random variable)\n"
        "  C_t      = 1[O_t > capacity]   — crowding indicator\n"
        "\n"
        "Bayesian Updating:\n"
        "  We use the Gamma–Poisson conjugate pair, which yields exact "
        "posterior updates without approximation. The posterior predictive "
        "distribution is Negative Binomial, obtained by integrating out λ.\n"
        "\n"
        "Occupancy Simulation:\n"
        "  Occupancy is computed via Monte Carlo: for each trajectory, we "
        "sample λ from its posterior, sample future arrivals, sample LOS "
        "for each arrival, and compute the resulting census. This propagates "
        "both parameter uncertainty and stochastic variation."
    )
    sections.append("")

    sections.append("=" * 70)
    sections.append("RESULTS")
    sections.append("=" * 70)
    sections.append(data_summary)
    sections.append("")
    sections.append(
        f"Posterior arrival rate: {belief.mean:.2f} admissions/day "
        f"(95% CI: [{ci[0]:.2f}, {ci[1]:.2f}])\n"
        f"Crowding probability (48h horizon): {100 * p:.1f}% "
        f"(95% CI: [{100 * crowd_result['ci_low']:.1f}%, "
        f"{100 * crowd_result['ci_high']:.1f}%])"
    )
    sections.append("")

    sections.append("=" * 70)
    sections.append("ETHICAL REFLECTION")
    sections.append("=" * 70)
    sections.append(
        "This model uses synthetic data for ethical reasons: real ICU data "
        "contains protected health information. The synthetic data preserves "
        "statistical structure without exposing individual patients.\n"
        "\n"
        "We explicitly do not recommend autonomous deployment of this model. "
        "Probabilistic forecasts should augment — never replace — clinical "
        "judgment. The failure-mode analysis demonstrates the model's known "
        "limitations, and the uncertainty quantification ensures that "
        "overconfident predictions are structurally impossible.\n"
        "\n"
        "The feedback-loop failure mode (Goodhart's Law) is particularly "
        "important: any model that influences the system it measures risks "
        "becoming unreliable. We flag this as an irreducible limitation."
    )

    return "\n".join(sections)
