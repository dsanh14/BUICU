"""
Synthetic MIMIC-IV ICU Dataset Generator
=========================================

Generates a synthetic dataset that faithfully replicates the statistical
structure of MIMIC-IV ICU data for ethical and accessibility reasons.

Statistical properties preserved:
  - Arrival process: non-homogeneous Poisson with diurnal + weekly modulation
  - Length-of-stay: log-normal mixture (short acute + heavy-tailed chronic)
  - Non-stationarity: capacity-surge windows with elevated arrival rates
  - Noise: jittered timestamps, occasional missing discharge times

Random variables modeled:
  N_t   ~ Poisson(lambda(t) * dt)   arrivals in window t
  L     ~ mixture of LogNormal       length of stay (hours)

References for calibration:
  - MIMIC-IV median ICU LOS ≈ 2.1 days, mean ≈ 4.3 days
  - Typical academic ICU: 20-30 beds, ~60-80 admissions/week
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyntheticICUConfig:
    """All parameters for synthetic data generation, with MIMIC-IV-calibrated defaults."""
    n_days: int = 180
    capacity: int = 50
    base_arrival_rate: float = 10.0        # admissions per day (baseline)
    surge_rate_multiplier: float = 1.8     # multiplier during surge windows
    surge_windows: list = field(default_factory=lambda: [(30, 50), (110, 130)])

    # Diurnal modulation: arrivals peak mid-morning, trough overnight
    diurnal_amplitude: float = 0.30
    diurnal_peak_hour: float = 11.0

    # Weekly modulation: slight dip on weekends
    weekly_amplitude: float = 0.10

    # LOS mixture: component 1 = short acute, component 2 = long chronic
    # Calibrated to MIMIC-IV: overall median ≈ 2 days, mean ≈ 4 days
    los_mixture_weight: float = 0.70       # P(short stay)
    los_short_mu: float = 3.5              # log-hours; exp(3.5) ≈ 33h ≈ 1.4d median
    los_short_sigma: float = 0.5
    los_long_mu: float = 5.1               # log-hours; exp(5.1) ≈ 164h ≈ 6.8d median
    los_long_sigma: float = 0.7

    # Noise
    missing_discharge_prob: float = 0.02   # fraction of stays with missing discharge
    timestamp_jitter_hours: float = 0.5

    seed: Optional[int] = 42


def _intensity(t_hours: float, config: SyntheticICUConfig) -> float:
    """
    Time-varying arrival intensity lambda(t) incorporating:
      1. base rate
      2. diurnal cycle  (sinusoidal, peak at config.diurnal_peak_hour)
      3. weekly cycle   (sinusoidal, trough on Sunday)
      4. surge windows  (step function multiplier)
    """
    day = t_hours / 24.0
    hour_of_day = t_hours % 24.0
    day_of_week = day % 7.0

    diurnal = 1.0 + config.diurnal_amplitude * np.cos(
        2.0 * np.pi * (hour_of_day - config.diurnal_peak_hour) / 24.0
    )
    weekly = 1.0 - config.weekly_amplitude * np.cos(
        2.0 * np.pi * day_of_week / 7.0
    )

    surge = 1.0
    for s_start, s_end in config.surge_windows:
        if s_start <= day < s_end:
            surge = config.surge_rate_multiplier
            break

    rate_per_hour = (config.base_arrival_rate / 24.0) * diurnal * weekly * surge
    return max(rate_per_hour, 0.0)


def _sample_los(rng: np.random.Generator, config: SyntheticICUConfig) -> float:
    """
    Sample a single length-of-stay from a two-component log-normal mixture.

    Component 1 (weight = config.los_mixture_weight):
        Short acute stays — median ≈ 1.1 days
    Component 2 (weight = 1 - config.los_mixture_weight):
        Long chronic stays — median ≈ 5.5 days, heavy tail

    Returns LOS in hours.
    """
    if rng.random() < config.los_mixture_weight:
        log_los = rng.normal(config.los_short_mu, config.los_short_sigma)
    else:
        log_los = rng.normal(config.los_long_mu, config.los_long_sigma)
    return np.exp(log_los)


def generate_dataset(config: Optional[SyntheticICUConfig] = None):
    """
    Generate a complete synthetic ICU dataset.

    Returns
    -------
    dict with keys:
        'admissions'   : np.ndarray, shape (n_patients,), admission times in hours
        'discharges'   : np.ndarray, shape (n_patients,), discharge times in hours (NaN if missing)
        'los_hours'    : np.ndarray, shape (n_patients,), length of stay in hours
        'capacity'     : int, ICU bed capacity
        'n_days'       : int, simulation horizon
        'config'       : SyntheticICUConfig used
        'surge_windows': list of (start_day, end_day) tuples
        'census_hourly': np.ndarray, shape (n_hours,), occupancy at each hour
    """
    if config is None:
        config = SyntheticICUConfig()

    rng = np.random.default_rng(config.seed)
    total_hours = config.n_days * 24

    # --- Step 1: Generate arrivals via thinning algorithm (Lewis-Shedler) ---
    # Upper bound on intensity for thinning
    lambda_max = (config.base_arrival_rate / 24.0) * \
                 (1.0 + config.diurnal_amplitude) * \
                 (1.0 + config.weekly_amplitude) * \
                 config.surge_rate_multiplier * 1.05  # safety margin

    admission_times = []
    t = 0.0
    while t < total_hours:
        u1 = rng.random()
        dt = -np.log(u1) / lambda_max
        t += dt
        if t >= total_hours:
            break
        u2 = rng.random()
        if u2 <= _intensity(t, config) / lambda_max:
            admission_times.append(t)

    admission_times = np.array(admission_times)
    n_patients = len(admission_times)

    # --- Step 2: Sample LOS for each patient ---
    los_hours = np.array([_sample_los(rng, config) for _ in range(n_patients)])

    # --- Step 3: Compute discharge times (with missingness) ---
    discharge_times = admission_times + los_hours
    missing_mask = rng.random(n_patients) < config.missing_discharge_prob
    discharge_times_observed = discharge_times.copy()
    discharge_times_observed[missing_mask] = np.nan

    # --- Step 4: Add timestamp jitter ---
    jitter = rng.normal(0, config.timestamp_jitter_hours, size=n_patients)
    admission_times_obs = admission_times + jitter
    admission_times_obs = np.clip(admission_times_obs, 0, total_hours)

    # --- Step 5: Compute hourly census (ground truth, using exact times) ---
    census = np.zeros(total_hours, dtype=int)
    for i in range(n_patients):
        a = int(np.floor(admission_times[i]))
        d = int(np.floor(discharge_times[i]))
        a = max(0, a)
        d = min(total_hours - 1, d)
        if a <= d:
            census[a:d + 1] += 1

    return {
        'admissions': admission_times_obs,
        'discharges': discharge_times_observed,
        'los_hours': los_hours,
        'capacity': config.capacity,
        'n_days': config.n_days,
        'config': config,
        'surge_windows': config.surge_windows,
        'census_hourly': census,
        'n_patients': n_patients,
        'admission_times_true': admission_times,
        'discharge_times_true': discharge_times,
    }


def summarize_dataset(data: dict) -> str:
    """Return a human-readable summary of the generated dataset."""
    los_days = data['los_hours'] / 24.0
    census = data['census_hourly']
    lines = [
        "=== Synthetic ICU Dataset Summary ===",
        f"Simulation horizon   : {data['n_days']} days ({data['n_days'] * 24} hours)",
        f"ICU capacity         : {data['capacity']} beds",
        f"Total admissions     : {data['n_patients']}",
        f"Avg admissions/day   : {data['n_patients'] / data['n_days']:.1f}",
        "",
        "Length-of-Stay (days):",
        f"  median = {np.nanmedian(los_days):.2f}",
        f"  mean   = {np.nanmean(los_days):.2f}",
        f"  p90    = {np.nanpercentile(los_days, 90):.2f}",
        f"  p99    = {np.nanpercentile(los_days, 99):.2f}",
        f"  max    = {np.nanmax(los_days):.2f}",
        "",
        "Census (hourly):",
        f"  mean occupancy = {np.mean(census):.1f}",
        f"  max occupancy  = {np.max(census)}",
        f"  hours over capacity = {np.sum(census > data['capacity'])} "
        f"({100 * np.mean(census > data['capacity']):.1f}%)",
        "",
        f"Surge windows: {data['surge_windows']}",
        f"Missing discharges: {np.sum(np.isnan(data['discharges']))} "
        f"({100 * np.mean(np.isnan(data['discharges'])):.1f}%)",
    ]
    return "\n".join(lines)
