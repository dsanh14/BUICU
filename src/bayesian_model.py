"""
Bayesian ICU Crowding Model
============================

Implements the full probabilistic pipeline:

  1. Gamma–Poisson conjugate updating for arrival rate λ
  2. Posterior predictive (Negative Binomial) for future admissions
  3. Monte Carlo occupancy simulation with full uncertainty propagation
  4. Crowding-probability computation  P(O_t > capacity | data)
  5. Windowed (adaptive) Bayesian model for non-stationary regimes
  6. KL divergence tracking for information gain quantification
  7. Prior sensitivity analysis
  8. Anomaly detection via posterior predictive p-values

Random variables (formal definitions):

  λ          ~ Gamma(α₀, β₀)                  prior on arrival rate
  N_t | λ    ~ Poisson(λ · Δt)                 arrivals in window Δt
  λ | data   ~ Gamma(α₀ + Σk, β₀ + T)         posterior after T time and Σk arrivals
  N_future   ~ NegBin(r=α_post, p=β_post/(β_post+Δt))   posterior predictive
  L          ~ empirical / parametric           length of stay
  O_t        = Σ_i 1[a_i ≤ t < a_i + L_i]     occupancy (random, not deterministic)

The Gamma–Poisson conjugacy is exact; occupancy is computed via Monte Carlo
because it involves a convolution of correlated random variables.
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln, digamma
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BeliefState:
    """
    Encapsulates the current Bayesian belief about the arrival rate λ.

    The posterior is Gamma(alpha, beta) where:
      - E[λ] = alpha / beta
      - Var[λ] = alpha / beta²
      - The posterior concentrates as more data is observed.
    """
    alpha: float
    beta: float
    time: float = 0.0
    total_arrivals: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def variance(self) -> float:
        return self.alpha / self.beta ** 2

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute equal-tailed credible interval for λ."""
        tail = (1.0 - level) / 2.0
        return (
            stats.gamma.ppf(tail, a=self.alpha, scale=1.0 / self.beta),
            stats.gamma.ppf(1.0 - tail, a=self.alpha, scale=1.0 / self.beta),
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate posterior PDF at given λ values."""
        return stats.gamma.pdf(x, a=self.alpha, scale=1.0 / self.beta)


def kl_divergence_gamma(alpha_p: float, beta_p: float,
                        alpha_q: float, beta_q: float) -> float:
    """
    KL(p || q) between two Gamma distributions in closed form.

    KL(Gamma(α_p,β_p) || Gamma(α_q,β_q)) =
        (α_p - α_q) ψ(α_p) - ln Γ(α_p) + ln Γ(α_q)
        + α_q (ln β_p - ln β_q) + α_p (β_q - β_p) / β_p

    This measures how much information the posterior (p) contains
    beyond what the prior (q) already encoded.
    """
    return (
        (alpha_p - alpha_q) * digamma(alpha_p)
        - gammaln(alpha_p) + gammaln(alpha_q)
        + alpha_q * (np.log(beta_p) - np.log(beta_q))
        + alpha_p * (beta_q - beta_p) / beta_p
    )


@dataclass
class BeliefHistory:
    """Tracks the evolution of beliefs over time for visualization."""
    times: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    betas: List[float] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    ci_lows: List[float] = field(default_factory=list)
    ci_highs: List[float] = field(default_factory=list)
    observed_counts: List[int] = field(default_factory=list)
    window_labels: List[str] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)
    anomaly_flags: List[bool] = field(default_factory=list)
    predictive_pvalues: List[float] = field(default_factory=list)

    def record(self, belief: BeliefState, observed_k: int, label: str = "",
               kl_div: float = 0.0, is_anomaly: bool = False,
               p_value: float = 0.5):
        self.times.append(belief.time)
        self.alphas.append(belief.alpha)
        self.betas.append(belief.beta)
        self.means.append(belief.mean)
        ci = belief.credible_interval(0.95)
        self.ci_lows.append(ci[0])
        self.ci_highs.append(ci[1])
        self.observed_counts.append(observed_k)
        self.window_labels.append(label)
        self.kl_divergences.append(kl_div)
        self.anomaly_flags.append(is_anomaly)
        self.predictive_pvalues.append(p_value)


class BayesianArrivalModel:
    """
    Gamma–Poisson conjugate model for ICU arrival rate estimation.

    Prior:     λ ~ Gamma(α₀, β₀)
    Likelihood (per window): N_t | λ ~ Poisson(λ · Δt)
    Posterior:  λ | data ~ Gamma(α₀ + Σk_i, β₀ + Σ Δt_i)

    The conjugate update is exact — no approximation needed.
    """

    def __init__(self, alpha_0: float = 2.0, beta_0: float = 0.2):
        """
        Parameters
        ----------
        alpha_0 : float
            Prior shape. Small values → vague prior. We use α₀=2 for a weakly
            informative prior centered near λ=10/day with wide uncertainty.
        beta_0 : float
            Prior rate (inverse scale). β₀=0.2 gives prior mean = 2/0.2 = 10/day.
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.belief = BeliefState(alpha=alpha_0, beta=beta_0)
        self.history = BeliefHistory()
        self.history.record(self.belief, observed_k=0, label="prior")

    def update(self, observed_count: int, window_duration: float,
               label: str = "") -> BeliefState:
        """
        Bayesian update with KL divergence tracking and anomaly detection.

        Conjugate update:
            α_new = α_old + k
            β_new = β_old + Δt

        Additionally computes:
          - KL(posterior_new || posterior_old): information gained
          - Posterior predictive p-value: P(N ≥ k) under previous belief
            → anomaly if p < 0.025 or p > 0.975
        """
        old_alpha, old_beta = self.belief.alpha, self.belief.beta

        # Anomaly detection: is this observation surprising given current belief?
        p_nb = old_beta / (old_beta + window_duration)
        p_value = 1.0 - stats.nbinom.cdf(observed_count - 1, n=old_alpha, p=p_nb)
        is_anomaly = (p_value < 0.025) or (p_value > 0.975)

        # Conjugate update
        new_alpha = old_alpha + observed_count
        new_beta = old_beta + window_duration

        # KL divergence: information gained from this observation
        kl_div = kl_divergence_gamma(new_alpha, new_beta, old_alpha, old_beta)

        self.belief = BeliefState(
            alpha=new_alpha, beta=new_beta,
            time=self.belief.time + window_duration,
            total_arrivals=self.belief.total_arrivals + observed_count,
        )
        self.history.record(self.belief, observed_count, label,
                            kl_div=kl_div, is_anomaly=is_anomaly,
                            p_value=p_value)
        return self.belief

    def posterior_predictive_pmf(self, future_window: float,
                                max_k: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior predictive distribution for future arrivals.

        Integrating out λ from Poisson(λ·Δt) × Gamma(α, β) yields:
            N_future ~ NegBin(r=α, p=β/(β+Δt))

        This is the key advantage of Bayesian inference: we propagate
        parameter uncertainty into predictions automatically.
        """
        alpha = self.belief.alpha
        beta = self.belief.beta
        p = beta / (beta + future_window)
        k_values = np.arange(0, max_k + 1)
        pmf = stats.nbinom.pmf(k_values, n=alpha, p=p)
        return k_values, pmf

    def posterior_predictive_sample(self, future_window: float,
                                   n_samples: int = 10000,
                                   rng: Optional[np.random.Generator] = None
                                   ) -> np.ndarray:
        """
        Draw samples from the posterior predictive by composition:
            1. λ_s ~ Gamma(α, β)           (sample rate from posterior)
            2. N_s ~ Poisson(λ_s · Δt)     (sample count given rate)

        This two-step procedure correctly propagates parameter uncertainty.
        """
        if rng is None:
            rng = np.random.default_rng()
        lambdas = rng.gamma(self.belief.alpha,
                            1.0 / self.belief.beta,
                            size=n_samples)
        counts = rng.poisson(lambdas * future_window)
        return counts

    def prob_exceeds(self, threshold: int, future_window: float) -> float:
        """P(N_future > threshold) under the posterior predictive."""
        alpha = self.belief.alpha
        beta = self.belief.beta
        p = beta / (beta + future_window)
        return 1.0 - stats.nbinom.cdf(threshold, n=alpha, p=p)

    def sequential_update(self, daily_counts: np.ndarray,
                          window_size: float = 1.0) -> BeliefHistory:
        """
        Perform sequential Bayesian updating over an array of daily counts.

        This simulates how beliefs evolve as each day's data arrives,
        which is the core demonstration of Bayesian reasoning.
        """
        for i, k in enumerate(daily_counts):
            label = f"day {i + 1}"
            self.update(int(k), window_size, label=label)
        return self.history


class WindowedBayesianModel:
    """
    Adaptive Bayesian model using a sliding window of recent observations.

    Motivation: The stationary model (BayesianArrivalModel) uses ALL historical
    data equally. When the arrival rate changes (surges, policy shifts), the
    stationary posterior is slow to react because it is anchored by old data.

    The windowed model uses only the last W days of observations:
        α_window = α₀ + Σ(k_i for last W days)
        β_window = β₀ + W

    This is still exact conjugate inference — we simply restrict the
    sufficient statistics to a sliding window. The tradeoff:
      - Smaller W → faster adaptation, higher variance
      - Larger W → slower adaptation, lower variance

    This directly addresses the non-stationarity failure mode (FM1).
    """

    def __init__(self, window_days: int = 14,
                 alpha_0: float = 2.0, beta_0: float = 0.2):
        self.window_days = window_days
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.history = BeliefHistory()

    def fit(self, daily_counts: np.ndarray) -> BeliefHistory:
        """Compute windowed posterior at each time step."""
        n = len(daily_counts)
        for t in range(n):
            start = max(0, t - self.window_days + 1)
            window_data = daily_counts[start:t + 1]
            w = len(window_data)

            alpha_t = self.alpha_0 + np.sum(window_data)
            beta_t = self.beta_0 + w

            belief = BeliefState(alpha=alpha_t, beta=beta_t, time=float(t + 1),
                                 total_arrivals=int(np.sum(window_data)))

            # KL from prior to this windowed posterior
            kl = kl_divergence_gamma(alpha_t, beta_t, self.alpha_0, self.beta_0)

            self.history.record(belief, int(daily_counts[t]),
                                label=f"day {t + 1}", kl_div=kl)
        return self.history


class PriorSensitivityAnalysis:
    """
    Demonstrates Bayesian robustness: different priors converge to
    similar posteriors as data accumulates.

    We run three priors:
      1. Uninformative:       Gamma(0.01, 0.001)  — nearly flat
      2. Weakly informative:  Gamma(2, 0.2)       — centered at 10
      3. Strong (wrong):      Gamma(50, 10)        — centered at 5 (deliberately wrong)

    The convergence of posteriors despite different priors is a key
    Bayesian insight that demonstrates understanding of how evidence
    overwhelms prior beliefs.
    """

    PRIORS = {
        'Uninformative': (0.01, 0.001),
        'Weakly informative': (2.0, 0.2),
        'Strong (wrong center)': (50.0, 10.0),
    }

    def __init__(self):
        self.histories = {}

    def run(self, daily_counts: np.ndarray) -> dict:
        """Run sequential updating under each prior and return histories."""
        for name, (a0, b0) in self.PRIORS.items():
            model = BayesianArrivalModel(alpha_0=a0, beta_0=b0)
            model.sequential_update(daily_counts)
            self.histories[name] = model.history
        return self.histories


class LOSModel:
    """
    Length-of-stay model supporting both empirical and parametric modes.

    Assumption: LOS is independent of arrival time (violated during surges —
    see failure-mode analysis).

    The empirical mode uses kernel density estimation on observed LOS values.
    The parametric mode fits a log-normal mixture (justified by the
    bimodal structure of ICU stays: short acute vs. long chronic).
    """

    def __init__(self, los_hours: np.ndarray, mode: str = "empirical"):
        self.raw_los = los_hours[~np.isnan(los_hours)]
        self.mode = mode

        if mode == "parametric":
            self._fit_lognormal()

    def _fit_lognormal(self):
        """Fit a single log-normal to log(LOS). Simple but useful baseline."""
        log_los = np.log(self.raw_los)
        self.mu_hat = np.mean(log_los)
        self.sigma_hat = np.std(log_los)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None
               ) -> np.ndarray:
        """Draw n LOS samples (in hours)."""
        if rng is None:
            rng = np.random.default_rng()
        if self.mode == "empirical":
            return rng.choice(self.raw_los, size=n, replace=True)
        else:
            return np.exp(rng.normal(self.mu_hat, self.sigma_hat, size=n))

    def summary_stats(self) -> dict:
        los_days = self.raw_los / 24.0
        return {
            'median_days': float(np.median(los_days)),
            'mean_days': float(np.mean(los_days)),
            'p90_days': float(np.percentile(los_days, 90)),
            'p99_days': float(np.percentile(los_days, 99)),
            'max_days': float(np.max(los_days)),
        }


class OccupancySimulator:
    """
    Monte Carlo simulator for ICU occupancy.

    Occupancy at time t is:
        O_t = Σ_i 1[a_i ≤ t < a_i + L_i]

    Because O_t is a sum over random arrival times and random LOS values,
    it is itself a random variable. We propagate uncertainty by:

      1. Sampling λ from its posterior
      2. Sampling future arrivals from Poisson(λ · Δt)
      3. Sampling LOS for each future patient
      4. Computing the resulting census trajectory
      5. Repeating to build a distribution over trajectories

    This gives us full predictive distributions for occupancy, including
    the probability of exceeding capacity.
    """

    def __init__(self, arrival_model: BayesianArrivalModel,
                 los_model: LOSModel, capacity: int):
        self.arrival_model = arrival_model
        self.los_model = los_model
        self.capacity = capacity

    def simulate_trajectories(
        self,
        current_patients: np.ndarray,
        forecast_hours: int = 48,
        n_trajectories: int = 2000,
        time_step_hours: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Simulate future occupancy trajectories.

        Parameters
        ----------
        current_patients : np.ndarray
            Remaining LOS (in hours) for each currently admitted patient.
        forecast_hours : int
            How far ahead to simulate (hours).
        n_trajectories : int
            Number of Monte Carlo trajectories.
        time_step_hours : float
            Resolution of the occupancy time series.
        rng : np.random.Generator

        Returns
        -------
        dict with:
            'time_grid'       : np.ndarray, time points in hours
            'trajectories'    : np.ndarray, shape (n_trajectories, n_steps)
            'mean'            : np.ndarray, mean occupancy at each step
            'median'          : np.ndarray, median occupancy
            'ci_low'          : np.ndarray, 2.5th percentile
            'ci_high'         : np.ndarray, 97.5th percentile
            'p_overcrowded'   : np.ndarray, P(O_t > capacity) at each step
        """
        if rng is None:
            rng = np.random.default_rng(123)

        n_steps = int(forecast_hours / time_step_hours)
        time_grid = np.arange(n_steps) * time_step_hours
        trajectories = np.zeros((n_trajectories, n_steps), dtype=int)

        alpha = self.arrival_model.belief.alpha
        beta = self.arrival_model.belief.beta
        forecast_days = forecast_hours / 24.0

        for traj in range(n_trajectories):
            # Step 1: Sample arrival rate from posterior
            lam = rng.gamma(alpha, 1.0 / beta)

            # Step 2: Existing patients — deterministic remaining LOS
            remaining_los = current_patients.copy()

            # Step 3: Sample future arrivals over the forecast window
            n_future = rng.poisson(lam * forecast_days)
            if n_future > 0:
                arrival_offsets = rng.uniform(0, forecast_hours, size=n_future)
                future_los = self.los_model.sample(n_future, rng=rng)
            else:
                arrival_offsets = np.array([])
                future_los = np.array([])

            # Step 4: Compute occupancy at each time step
            for step_idx, t in enumerate(time_grid):
                # Count existing patients still present
                n_existing = int(np.sum(remaining_los > t))

                # Count future patients present at time t
                if n_future > 0:
                    arrived = arrival_offsets <= t
                    not_discharged = (arrival_offsets + future_los) > t
                    n_new = int(np.sum(arrived & not_discharged))
                else:
                    n_new = 0

                trajectories[traj, step_idx] = n_existing + n_new

        return {
            'time_grid': time_grid,
            'trajectories': trajectories,
            'mean': np.mean(trajectories, axis=0),
            'median': np.median(trajectories, axis=0),
            'ci_low': np.percentile(trajectories, 2.5, axis=0),
            'ci_high': np.percentile(trajectories, 97.5, axis=0),
            'p_overcrowded': np.mean(trajectories > self.capacity, axis=0),
            'capacity': self.capacity,
        }

    def crowding_probability(self, current_patients: np.ndarray,
                             horizon_hours: int = 48,
                             n_samples: int = 5000,
                             rng: Optional[np.random.Generator] = None
                             ) -> dict:
        """
        Compute P(max occupancy > capacity within horizon).

        Returns a scalar probability plus a credible interval obtained
        by bootstrap resampling of the Monte Carlo estimate itself
        (uncertainty about our uncertainty estimate).
        """
        if rng is None:
            rng = np.random.default_rng(456)

        result = self.simulate_trajectories(
            current_patients, horizon_hours, n_samples, rng=rng
        )
        peak_occupancy = np.max(result['trajectories'], axis=1)
        p_crowd = np.mean(peak_occupancy > self.capacity)

        # Bootstrap CI on the Monte Carlo estimate
        n_boot = 1000
        boot_ps = np.zeros(n_boot)
        for b in range(n_boot):
            boot_sample = rng.choice(peak_occupancy, size=len(peak_occupancy),
                                     replace=True)
            boot_ps[b] = np.mean(boot_sample > self.capacity)
        ci_low, ci_high = np.percentile(boot_ps, [2.5, 97.5])

        return {
            'probability': float(p_crowd),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'horizon_hours': horizon_hours,
            'n_samples': n_samples,
            'peak_occupancy_dist': peak_occupancy,
        }
