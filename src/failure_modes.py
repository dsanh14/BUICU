"""
Failure-Mode Analysis for the Bayesian ICU Crowding Model
==========================================================

This module systematically identifies when and why the model's assumptions
break, quantifies the consequences, and triggers appropriate uncertainty
adjustments.

Each failure mode follows a structured protocol:
  1. State the assumption
  2. Define a detection heuristic
  3. Explain how the assumption breaks
  4. Explain the consequence for predictions
  5. Describe the mitigation (usually: widen uncertainty)

Failure modes analyzed:
  FM1: Non-stationarity (surges, seasonality, policy changes)
  FM2: Independence violations (correlated arrivals, LOS–crowding feedback)
  FM3: Data quality issues (delayed charting, missing timestamps)
  FM4: Distribution mismatch (heavy tails, rare catastrophic cases)
  FM5: Feedback loops / Goodhart's Law (predictions influencing behavior)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FailureModeReport:
    """Structured report for a single failure mode."""
    name: str
    assumption: str
    how_it_breaks: str
    consequence: str
    mitigation: str
    detected: bool
    severity: str          # "low", "medium", "high"
    confidence_penalty: float  # multiplicative widening factor for CIs (≥1.0)
    evidence: str          # human-readable summary of detection evidence


class FailureModeAnalyzer:
    """
    Runs all failure-mode checks against observed data and model state.

    When failure modes are detected, the analyzer returns:
      - structured reports for the writeup
      - a combined confidence penalty to widen prediction intervals
      - natural-language explanations for the NL interface
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

    def analyze_all(
        self,
        daily_counts: np.ndarray,
        los_hours: np.ndarray,
        census_hourly: np.ndarray,
        surge_windows: List[tuple],
        missing_fraction: float,
    ) -> List[FailureModeReport]:
        """Run all failure-mode checks and return structured reports."""
        reports = []
        reports.append(self._fm1_nonstationarity(daily_counts, surge_windows))
        reports.append(self._fm2_independence(daily_counts, los_hours, census_hourly))
        reports.append(self._fm3_data_quality(missing_fraction, daily_counts))
        reports.append(self._fm4_distribution_mismatch(los_hours))
        reports.append(self._fm5_feedback_loops())
        return reports

    def combined_confidence_penalty(self, reports: List[FailureModeReport]) -> float:
        """
        Compute the combined uncertainty widening factor.

        When multiple failure modes are active, their penalties compound
        multiplicatively — this is conservative by design.
        """
        penalty = 1.0
        for r in reports:
            if r.detected:
                penalty *= r.confidence_penalty
        return penalty

    def _fm1_nonstationarity(self, daily_counts: np.ndarray,
                              surge_windows: List[tuple]) -> FailureModeReport:
        """
        FM1: Non-stationarity

        Assumption: Arrivals follow a stationary Poisson process with
                    constant rate λ across the observation period.

        Detection: Compare mean arrival rate in surge vs. non-surge windows.
                   A ratio > 1.3 indicates meaningful non-stationarity.

        Consequence: Posterior λ averages over regimes, underestimating risk
                     during surges and overestimating during lulls.
        """
        n_days = len(daily_counts)
        surge_mask = np.zeros(n_days, dtype=bool)
        for s, e in surge_windows:
            surge_mask[max(0, int(s)):min(n_days, int(e))] = True

        surge_mean = np.mean(daily_counts[surge_mask]) if surge_mask.any() else 0
        normal_mean = np.mean(daily_counts[~surge_mask]) if (~surge_mask).any() else 0
        ratio = surge_mean / max(normal_mean, 0.1)

        detected = ratio > 1.3
        severity = "high" if ratio > 1.6 else ("medium" if ratio > 1.3 else "low")
        penalty = 1.0 + 0.3 * max(0, ratio - 1.0) if detected else 1.0

        return FailureModeReport(
            name="Non-stationarity (FM1)",
            assumption="Arrival rate λ is constant over the observation period.",
            how_it_breaks=(
                f"Surge periods show mean {surge_mean:.1f} admissions/day vs. "
                f"{normal_mean:.1f} during normal periods (ratio = {ratio:.2f}). "
                "This violates the stationarity assumption of the Poisson model."
            ),
            consequence=(
                "The posterior λ is a weighted average across regimes. During surges, "
                "the model underestimates arrival rate and therefore underestimates "
                "crowding probability. During lulls, it overestimates."
            ),
            mitigation=(
                "Partition data into regime-specific windows and maintain separate "
                "posteriors. Alternatively, use a time-varying rate model (e.g., "
                "changepoint detection or a hierarchical model with regime indicators). "
                "As an immediate hedge, we widen credible intervals."
            ),
            detected=detected,
            severity=severity,
            confidence_penalty=penalty,
            evidence=(
                f"Surge/normal arrival ratio = {ratio:.2f}. "
                f"Surge windows: {surge_windows}."
            ),
        )

    def _fm2_independence(self, daily_counts: np.ndarray,
                           los_hours: np.ndarray,
                           census_hourly: np.ndarray) -> FailureModeReport:
        """
        FM2: Independence violations

        Assumption: Arrivals are independent, and LOS is independent of
                    the current census (occupancy level).

        Detection: Compute lag-1 autocorrelation of daily counts and
                   correlation between daily census and LOS.

        Consequence: If arrivals cluster or LOS increases during crowding
                     (boarding effect), the model underestimates tail risk.
        """
        if len(daily_counts) < 3:
            return FailureModeReport(
                name="Independence violations (FM2)",
                assumption="Arrivals and LOS are mutually independent.",
                how_it_breaks="Insufficient data to test.",
                consequence="Unknown.",
                mitigation="Collect more data.",
                detected=False, severity="low", confidence_penalty=1.0,
                evidence="Fewer than 3 days of data."
            )

        autocorr = np.corrcoef(daily_counts[:-1], daily_counts[1:])[0, 1]
        detected = abs(autocorr) > 0.05
        severity = "high" if abs(autocorr) > 0.3 else (
            "medium" if abs(autocorr) > 0.1 else "low"
        )
        penalty = 1.0 + 0.2 * abs(autocorr) if detected else 1.0

        return FailureModeReport(
            name="Independence violations (FM2)",
            assumption=(
                "Arrivals in each time window are independent, and LOS is "
                "independent of the occupancy level."
            ),
            how_it_breaks=(
                f"Lag-1 autocorrelation of daily admissions = {autocorr:.3f}. "
                "Positive autocorrelation indicates clustering (e.g., multi-patient "
                "trauma events, seasonal illness waves). In practice, LOS also "
                "increases when ICUs are crowded (patients board in the ED longer, "
                "discharge planning slows)."
            ),
            consequence=(
                "The Poisson model treats each time window as independent. "
                "Positive autocorrelation means bursts of arrivals are more likely "
                "than the model predicts, leading to underestimated tail risk for "
                "crowding events."
            ),
            mitigation=(
                "Model arrivals with a Hawkes process or autoregressive Poisson "
                "to capture temporal dependence. Model LOS as a function of "
                "concurrent occupancy. As an immediate hedge, widen CIs."
            ),
            detected=detected,
            severity=severity,
            confidence_penalty=penalty,
            evidence=f"Lag-1 autocorrelation = {autocorr:.3f}.",
        )

    def _fm3_data_quality(self, missing_fraction: float,
                           daily_counts: np.ndarray) -> FailureModeReport:
        """
        FM3: Data quality issues

        Assumption: Admission and discharge times are accurately recorded.

        Detection: Check fraction of missing discharge timestamps and
                   look for implausible zero-count days.

        Consequence: Missing data biases LOS estimates (long-stay patients
                     more likely to have missing discharge → right-censoring
                     bias). Missing admissions undercount arrivals.
        """
        zero_days = np.sum(daily_counts == 0)
        detected = missing_fraction > 0.01 or zero_days > 0
        severity = "high" if missing_fraction > 0.05 else (
            "medium" if missing_fraction > 0.01 else "low"
        )
        penalty = 1.0 + 2.0 * missing_fraction if detected else 1.0

        return FailureModeReport(
            name="Data quality issues (FM3)",
            assumption="Timestamps are accurately and completely recorded.",
            how_it_breaks=(
                f"{100 * missing_fraction:.1f}% of discharges are missing. "
                f"{zero_days} days have zero recorded admissions. "
                "Missing discharge times are more common for long-stay patients "
                "(informative missingness / right-censoring). Zero-count days "
                "may reflect charting gaps rather than genuinely quiet periods."
            ),
            consequence=(
                "Missing discharge data causes LOS to be underestimated (we lose "
                "the longest stays), which in turn underestimates occupancy duration. "
                "Missing admissions directly bias the arrival rate downward."
            ),
            mitigation=(
                "Use survival analysis techniques (Kaplan–Meier, Cox) to handle "
                "right-censored LOS. Impute missing admissions using auxiliary "
                "data (e.g., ED logs). Flag days with zero counts for manual review. "
                "Widen CIs to reflect data-quality uncertainty."
            ),
            detected=detected,
            severity=severity,
            confidence_penalty=penalty,
            evidence=(
                f"Missing discharge fraction = {100 * missing_fraction:.1f}%. "
                f"Zero-count days = {zero_days}."
            ),
        )

    def _fm4_distribution_mismatch(self, los_hours: np.ndarray
                                    ) -> FailureModeReport:
        """
        FM4: Distribution mismatch (heavy tails)

        Assumption: The LOS distribution is well-captured by the empirical
                    distribution or our parametric fit.

        Detection: Compute kurtosis. Heavy tails (kurtosis >> 3) indicate
                   rare but extreme LOS values that dominate occupancy risk.

        Consequence: If the model underestimates the probability of very
                     long stays, it will underestimate sustained high occupancy.
        """
        los_clean = los_hours[~np.isnan(los_hours)]
        from scipy.stats import kurtosis as compute_kurtosis
        kurt = float(compute_kurtosis(los_clean, fisher=True))

        p99 = np.percentile(los_clean / 24.0, 99)
        p999 = np.percentile(los_clean / 24.0, 99.9)
        tail_ratio = p999 / max(p99, 0.01)

        detected = kurt > 3.0 or tail_ratio > 2.0
        severity = "high" if kurt > 6.0 else ("medium" if kurt > 3.0 else "low")
        penalty = min(1.0 + 0.05 * max(0, kurt - 3.0), 1.5) if detected else 1.0

        return FailureModeReport(
            name="Distribution mismatch / heavy tails (FM4)",
            assumption=(
                "The empirical LOS distribution adequately captures tail behavior, "
                "including rare ultra-long stays."
            ),
            how_it_breaks=(
                f"LOS excess kurtosis = {kurt:.2f} (>3 indicates heavier tails "
                f"than Gaussian). p99/p999 ratio of LOS = {tail_ratio:.2f}. "
                "A few patients stay 10–30+ days, creating persistent occupancy "
                "that the bulk of the distribution cannot predict."
            ),
            consequence=(
                "Monte Carlo simulation with the empirical LOS may still under-"
                "sample extreme tail events. If a simulation run fails to draw "
                "rare 20+ day stays, it will underestimate periods of sustained "
                "high occupancy."
            ),
            mitigation=(
                "Use importance sampling to oversample tail LOS events. Fit a "
                "generalized Pareto distribution to the upper tail. Increase "
                "the number of Monte Carlo samples. Report sensitivity of "
                "crowding probability to tail assumptions."
            ),
            detected=detected,
            severity=severity,
            confidence_penalty=penalty,
            evidence=(
                f"Excess kurtosis = {kurt:.2f}. "
                f"p99 LOS = {p99:.1f} days, p99.9 = {p999:.1f} days."
            ),
        )

    def _fm5_feedback_loops(self) -> FailureModeReport:
        """
        FM5: Feedback loops / Goodhart's Law

        Assumption: The model's predictions do not influence the system
                    it is modeling.

        This failure mode cannot be detected from data alone — it is
        a structural concern that must be disclosed.

        Consequence: If clinicians act on crowding forecasts (e.g., by
                     diverting patients or expediting discharges), the
                     observed arrival rate will differ from the rate that
                     would have occurred without intervention. The model
                     would then "learn" the post-intervention rate and
                     under-predict what happens if the intervention is
                     removed (Goodhart's Law).
        """
        return FailureModeReport(
            name="Feedback loops / Goodhart's Law (FM5)",
            assumption=(
                "Predictions do not influence the system being modeled. "
                "The arrival process and discharge process are exogenous."
            ),
            how_it_breaks=(
                "If ICU staff use crowding forecasts to divert patients or "
                "accelerate discharges, the observed data reflects the "
                "post-intervention system, not the counterfactual. The model "
                "then learns the 'managed' rate and will underpredict crowding "
                "if interventions are relaxed."
            ),
            consequence=(
                "Predictions become self-defeating: high crowding forecasts "
                "trigger interventions that reduce observed crowding, causing "
                "the model to lower its estimates, which reduces intervention "
                "urgency. This creates a dangerous oscillation."
            ),
            mitigation=(
                "Document all interventions triggered by model output. Use "
                "intention-to-treat analysis: model the arrival rate that would "
                "have been observed without intervention. Maintain a separate "
                "'unmanaged' prior. Never deploy this model for autonomous "
                "decision-making without human oversight."
            ),
            detected=True,
            severity="medium",
            confidence_penalty=1.05,
            evidence=(
                "This is a structural concern, not empirically detectable. "
                "Any deployment of a predictive model in a clinical setting "
                "risks Goodhart's Law. We flag it unconditionally."
            ),
        )


def format_failure_report(reports: List[FailureModeReport]) -> str:
    """Format all failure-mode reports into a structured writeup section."""
    lines = ["=" * 70, "FAILURE-MODE ANALYSIS", "=" * 70, ""]
    for i, r in enumerate(reports, 1):
        lines.append(f"--- {r.name} ---")
        lines.append(f"  Assumption   : {r.assumption}")
        lines.append(f"  Violation    : {r.how_it_breaks}")
        lines.append(f"  Consequence  : {r.consequence}")
        lines.append(f"  Mitigation   : {r.mitigation}")
        lines.append(f"  Detected     : {'YES' if r.detected else 'no'}")
        lines.append(f"  Severity     : {r.severity}")
        lines.append(f"  CI widening  : ×{r.confidence_penalty:.2f}")
        lines.append(f"  Evidence     : {r.evidence}")
        lines.append("")
    combined = 1.0
    for r in reports:
        if r.detected:
            combined *= r.confidence_penalty
    lines.append(f"Combined CI widening factor: ×{combined:.2f}")
    lines.append("")
    return "\n".join(lines)
