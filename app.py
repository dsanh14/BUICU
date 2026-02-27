"""
BUICU — Bayesian ICU Crowding Under Uncertainty
=================================================
Interactive Streamlit dashboard for exploring Bayesian belief updating
applied to ICU crowding prediction.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.synthetic_data import SyntheticICUConfig, generate_dataset
from src.bayesian_model import (
    BayesianArrivalModel, LOSModel, OccupancySimulator, BeliefState,
    WindowedBayesianModel, PriorSensitivityAnalysis, VarianceDecomposition,
    MLEComparison, ModelComparisonScorer, kl_divergence_gamma,
)
from src.failure_modes import FailureModeAnalyzer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BUICU — Bayesian ICU Crowding",
    page_icon="\U0001F3E5",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
BLUE = "#2563EB"
ORANGE = "#F59E0B"
RED = "#DC2626"
GREEN = "#10B981"
GRAY = "#6B7280"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Cached computations (run once, persist across reruns)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    config = SyntheticICUConfig()
    data = generate_dataset(config)

    n_days = data["n_days"]
    daily_counts = np.zeros(n_days, dtype=int)
    for t in data["admissions"]:
        idx = int(t / 24.0)
        if 0 <= idx < n_days:
            daily_counts[idx] += 1

    return data, daily_counts, config


@st.cache_data
def fit_default_models(_daily_counts):
    dc = np.array(_daily_counts)
    model = BayesianArrivalModel(alpha_0=2.0, beta_0=0.2)
    model.sequential_update(dc)

    windowed = WindowedBayesianModel(window_days=14, alpha_0=2.0, beta_0=0.2)
    w_history = windowed.fit(dc)

    return model, w_history


@st.cache_data
def run_failure_modes(_daily_counts, _los_hours, _census_hourly,
                      _surge_windows, _capacity, _discharges):
    dc = np.array(_daily_counts)
    los = np.array(_los_hours)
    census = np.array(_census_hourly)
    dis = np.array(_discharges)
    missing_frac = float(np.mean(np.isnan(dis)))

    analyzer = FailureModeAnalyzer(_capacity)
    reports = analyzer.analyze_all(
        daily_counts=dc,
        los_hours=los,
        census_hourly=census,
        surge_windows=list(_surge_windows),
        missing_fraction=missing_frac,
    )
    penalty = analyzer.combined_confidence_penalty(reports)
    return reports, penalty


@st.cache_data
def run_mle_comparison(_daily_counts):
    dc = np.array(_daily_counts)
    return MLEComparison.compare_over_time(dc, 2.0, 0.2)


@st.cache_data
def run_prior_sensitivity(_daily_counts):
    dc = np.array(_daily_counts)
    psa = PriorSensitivityAnalysis()
    histories = psa.run(dc)
    return histories


@st.cache_data
def compute_log_scores(_daily_counts, _stat_alphas, _stat_betas,
                       _wind_alphas, _wind_betas):
    from src.bayesian_model import BeliefHistory
    dc = np.array(_daily_counts)

    stat_h = BeliefHistory()
    stat_h.alphas = list(_stat_alphas)
    stat_h.betas = list(_stat_betas)

    wind_h = BeliefHistory()
    wind_h.alphas = list(_wind_alphas)
    wind_h.betas = list(_wind_betas)

    return ModelComparisonScorer.compute_log_scores(dc, stat_h, wind_h)


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
data, daily_counts, config = load_data()
default_model, w_history = fit_default_models(daily_counts)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0'>BUICU</h1>"
    "<p style='text-align:center; color:#6B7280; font-size:1.15rem; "
    "margin-top:0'>Belief Updating for ICU Crowding Under Uncertainty</p>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
ci_default = default_model.belief.credible_interval(0.95)
col1.metric(
    "\u03BB Posterior Mean",
    f"{default_model.belief.mean:.2f} adm/day",
    f"95% CI: [{ci_default[0]:.2f}, {ci_default[1]:.2f}]",
)
col2.metric(
    "Total Observations",
    f"{int(default_model.belief.total_arrivals):,}",
    f"over {int(default_model.belief.time)} days",
)
n_anom = sum(default_model.history.anomaly_flags)
col3.metric("Anomalous Days", f"{n_anom}", f"of {len(daily_counts)} total")
col4.metric("CS109 Concepts", "16", "demonstrated in this project")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_model, tab_update, tab_forecast, tab_eval, tab_concepts = st.tabs([
    "\U0001F4D0 The Model",
    "\U0001F504 Interactive Belief Updating",
    "\U0001F3E5 Crowding Forecast",
    "\U0001F50D Model Evaluation",
    "\U0001F4DA CS109 Concepts",
])


# ===========================  TAB 1: THE MODEL  ============================
with tab_model:
    st.header("Probabilistic Model")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Random Variables")
        st.latex(r"""
        \begin{aligned}
        \lambda &\sim \mathrm{Gamma}(\alpha_0,\; \beta_0)
            && \text{prior on arrival rate} \\[4pt]
        N_t \mid \lambda &\sim \mathrm{Poisson}(\lambda \cdot \Delta t)
            && \text{arrivals in window } \Delta t \\[4pt]
        \lambda \mid \text{data} &\sim
            \mathrm{Gamma}\!\left(\alpha_0 + \textstyle\sum k_i,\;
            \beta_0 + T\right)
            && \text{posterior (conjugate)} \\[4pt]
        N_{\text{future}} &\sim \mathrm{NegBin}\!\left(\alpha_{\text{post}},\;
            \tfrac{\beta_{\text{post}}}{\beta_{\text{post}}+\Delta t}\right)
            && \text{posterior predictive} \\[4pt]
        O_t &= \textstyle\sum_i \mathbf{1}[a_i \le t < a_i + L_i]
            && \text{occupancy (random variable)}
        \end{aligned}
        """)

        st.subheader("Why Bayesian?")
        st.markdown(
            "The Gamma\u2013Poisson conjugate model gives **exact** posteriors "
            "\u2014 no MCMC approximation needed. Every prediction automatically "
            "propagates parameter uncertainty into the forecast. We never produce "
            "a point estimate without a credible interval."
        )

    with right:
        st.subheader("Synthetic Dataset")
        los_days = data["los_hours"] / 24.0
        valid_los = los_days[~np.isnan(los_days)]

        stats_md = f"""
| Statistic | Value |
|---|---|
| Simulation | **{config.n_days} days** |
| Capacity | **{config.capacity} beds** |
| Total admissions | **{len(data['admissions']):,}** |
| Mean adm/day | **{np.mean(daily_counts):.1f}** |
| LOS median | **{np.median(valid_los):.2f} days** |
| LOS mean | **{np.mean(valid_los):.2f} days** |
| LOS p99 | **{np.percentile(valid_los, 99):.1f} days** |
| Surge windows | **{config.surge_windows}** |
"""
        st.markdown(stats_md)

    # LOS distribution
    st.subheader("Length-of-Stay Distribution")
    fig_los, (ax_los1, ax_los2) = plt.subplots(1, 2, figsize=(12, 3.5))

    ax_los1.hist(valid_los, bins=80, density=True, color=BLUE, alpha=0.6,
                 edgecolor="white", linewidth=0.3)
    ax_los1.axvline(np.median(valid_los), color=RED, linestyle="--",
                    label=f"median = {np.median(valid_los):.1f}d")
    ax_los1.set_xlabel("Days")
    ax_los1.set_ylabel("Density")
    ax_los1.set_title("LOS Distribution (linear scale)")
    ax_los1.legend()

    ax_los2.hist(valid_los, bins=80, density=True, color=ORANGE, alpha=0.6,
                 edgecolor="white", linewidth=0.3)
    ax_los2.set_yscale("log")
    ax_los2.set_xlabel("Days")
    ax_los2.set_title("LOS Distribution (log scale \u2014 reveals heavy tail)")

    plt.tight_layout()
    st.pyplot(fig_los)
    plt.close(fig_los)


# ===================  TAB 2: INTERACTIVE BELIEF UPDATING  ===================
with tab_update:
    st.header("Interactive Bayesian Belief Updating")
    st.markdown(
        "Adjust the **prior** and scrub through **time** to watch the "
        "posterior concentrate as evidence accumulates. This is the core "
        "of Bayesian reasoning: beliefs update continuously, uncertainty "
        "narrows, and every prediction carries a credible interval."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    alpha_0 = ctrl1.slider(
        "\u03B1\u2080 (prior shape)", 0.1, 50.0, 2.0, 0.1,
        help="Controls prior strength. Small = vague, large = confident.",
    )
    beta_0 = ctrl2.slider(
        "\u03B2\u2080 (prior rate)", 0.01, 10.0, 0.2, 0.01,
        help="Prior rate parameter. Prior mean = \u03B1\u2080 / \u03B2\u2080.",
    )
    day_t = ctrl3.slider(
        "Observation day", 1, len(daily_counts), len(daily_counts),
        help="Scrub to see how belief evolves day by day.",
    )

    prior_mean = alpha_0 / beta_0
    st.caption(
        f"Prior: Gamma({alpha_0:.1f}, {beta_0:.2f})  \u2192  "
        f"E[\u03BB] = {prior_mean:.1f},  "
        f"Std[\u03BB] = {np.sqrt(alpha_0) / beta_0:.2f}"
    )

    # Fit model with user-chosen prior up to day_t
    user_model = BayesianArrivalModel(alpha_0=alpha_0, beta_0=beta_0)
    user_model.sequential_update(daily_counts[:day_t])
    belief = user_model.belief

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    user_ci = belief.credible_interval(0.95)
    m1.metric("Posterior Mean", f"{belief.mean:.3f}")
    m2.metric("95% Credible Interval", f"[{user_ci[0]:.2f}, {user_ci[1]:.2f}]")
    kl_total = sum(user_model.history.kl_divergences)
    m3.metric("Total Information Gain (KL)", f"{kl_total:.2f}")

    decomp_now = VarianceDecomposition.decompose_at_belief(belief)
    m4.metric(
        "Epistemic Fraction",
        f"{100 * decomp_now['parameter_fraction']:.1f}%",
        "of forecast variance",
    )

    # --- Two-panel: density + evolution ---
    fig_upd, (ax_dens, ax_evo) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: prior vs posterior density
    x_max = max(prior_mean * 3, belief.mean * 1.5, 20)
    x = np.linspace(0.01, x_max, 500)

    prior_pdf = stats.gamma.pdf(x, a=alpha_0, scale=1.0 / beta_0)
    post_pdf = stats.gamma.pdf(x, a=belief.alpha, scale=1.0 / belief.beta)

    ax_dens.fill_between(x, prior_pdf, alpha=0.25, color=RED, label="Prior")
    ax_dens.plot(x, prior_pdf, color=RED, linewidth=1.5, linestyle="--")
    ax_dens.fill_between(x, post_pdf, alpha=0.35, color=BLUE,
                         label=f"Posterior (day {day_t})")
    ax_dens.plot(x, post_pdf, color=BLUE, linewidth=2)
    ax_dens.axvline(belief.mean, color=BLUE, linestyle=":", alpha=0.5)
    ax_dens.set_xlabel("\u03BB (admissions/day)")
    ax_dens.set_ylabel("Density")
    ax_dens.set_title("Prior \u2192 Posterior Transformation")
    ax_dens.legend()

    # Right: belief evolution over time
    h = user_model.history
    times = np.array(h.times)
    means = np.array(h.means)
    ci_lo = np.array(h.ci_lows)
    ci_hi = np.array(h.ci_highs)
    obs = np.array(h.observed_counts)

    if len(times) > 1:
        ax_evo.fill_between(times[1:], ci_lo[1:], ci_hi[1:],
                            alpha=0.2, color=BLUE, label="95% CI")
        ax_evo.plot(times[1:], means[1:], color=BLUE, linewidth=2,
                    label="Posterior mean")
        ax_evo.scatter(times[1:], obs[1:], s=8, color=GRAY, alpha=0.4,
                       zorder=3, label="Observed counts")
        for s, e in config.surge_windows:
            if s < day_t:
                ax_evo.axvspan(s, min(e, day_t), alpha=0.1, color=ORANGE)
    ax_evo.set_xlabel("Day")
    ax_evo.set_ylabel("\u03BB (adm/day)")
    ax_evo.set_title("Belief Evolution Over Time")
    ax_evo.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_upd)
    plt.close(fig_upd)

    # --- Variance decomposition ---
    st.subheader("Law of Total Variance: Epistemic vs. Aleatoric Uncertainty")
    st.latex(r"""
    \mathrm{Var}[N_{\text{future}}] \;=\;
        \underbrace{E\bigl[\mathrm{Var}[N \mid \lambda]\bigr]}_{
            \text{stochastic (aleatoric)}}
        \;+\;
        \underbrace{\mathrm{Var}\bigl[E[N \mid \lambda]\bigr]}_{
            \text{parameter (epistemic)}}
    """)

    decomp_time = VarianceDecomposition.decompose_over_time(user_model.history)

    fig_var, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(14, 3.5))

    t_arr = decomp_time["times"]
    ax_v1.fill_between(t_arr, 0, decomp_time["stochastic"], alpha=0.4,
                       color=BLUE, label="Stochastic (irreducible)")
    ax_v1.fill_between(t_arr, decomp_time["stochastic"], decomp_time["total"],
                       alpha=0.4, color=ORANGE,
                       label="Parameter (reducible with data)")
    ax_v1.set_xlabel("Day")
    ax_v1.set_ylabel("Variance of $N_{\\mathrm{future}}$")
    ax_v1.set_title("Uncertainty Decomposition (absolute)")
    ax_v1.legend(fontsize=8)

    ax_v2.fill_between(t_arr, 0, decomp_time["stochastic_frac"],
                       alpha=0.5, color=BLUE, label="Stochastic fraction")
    ax_v2.fill_between(t_arr, decomp_time["stochastic_frac"], 1.0,
                       alpha=0.5, color=ORANGE, label="Parameter fraction")
    ax_v2.set_xlabel("Day")
    ax_v2.set_ylabel("Fraction of total variance")
    ax_v2.set_ylim(0, 1)
    ax_v2.set_title("Uncertainty Composition (relative)")
    ax_v2.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_var)
    plt.close(fig_var)

    if day_t >= len(daily_counts):
        st.info(
            f"**Key insight:** After {day_t} days, "
            f"**{100 * decomp_now['stochastic_fraction']:.0f}%** "
            "of forecast uncertainty is irreducible stochastic noise. "
            "More data cannot reduce this \u2014 only changing the underlying "
            "process (e.g., reducing arrival variability) can. This is the "
            "fundamental distinction between **epistemic** uncertainty "
            "(what we don't know) and **aleatoric** uncertainty "
            "(what is inherently random)."
        )


# =======================  TAB 3: CROWDING FORECAST  ========================
with tab_forecast:
    st.header("48-Hour Crowding Forecast")
    st.markdown(
        "Monte Carlo simulation propagates **both** parameter uncertainty "
        "(not knowing \u03BB) and stochastic uncertainty (Poisson randomness) "
        "into a full predictive distribution over future occupancy."
    )

    fc1, fc2 = st.columns(2)
    user_capacity = fc1.slider("ICU Capacity (beds)", 20, 80, config.capacity)
    forecast_hours = fc2.slider("Forecast Horizon (hours)", 12, 96, 48, 6)

    # Snapshot at day 36 (surge onset, near capacity)
    snapshot_day = 36
    snapshot_hour = snapshot_day * 24 + 12
    admissions = data["admissions"]
    discharges = data["discharges"]

    present_mask = admissions <= snapshot_hour
    for i in range(len(present_mask)):
        if present_mask[i] and not np.isnan(discharges[i]):
            if discharges[i] <= snapshot_hour:
                present_mask[i] = False

    patient_indices = np.where(present_mask)[0]
    current_occupancy = len(patient_indices)

    remaining_los = []
    for idx in patient_indices:
        if np.isnan(discharges[idx]):
            remaining_los.append(48.0)
        else:
            remaining_los.append(max(0, discharges[idx] - snapshot_hour))
    current_patients = np.array(remaining_los)

    los_model = LOSModel(data["los_hours"], mode="empirical")

    # Use beliefs from day 36
    sim_model = BayesianArrivalModel(alpha_0=2.0, beta_0=0.2)
    sim_model.sequential_update(daily_counts[:snapshot_day])
    simulator = OccupancySimulator(sim_model, los_model, user_capacity)

    with st.spinner("Running 2,000 Monte Carlo trajectories\u2026"):
        sim_result = simulator.simulate_trajectories(
            current_patients,
            forecast_hours=forecast_hours,
            n_trajectories=2000,
            rng=np.random.default_rng(42),
        )

    peak_occ = np.max(sim_result["trajectories"], axis=1)
    p_crowd = float(np.mean(peak_occ > user_capacity))

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "Current Occupancy",
        f"{current_occupancy} / {user_capacity}",
        f"Day {snapshot_day} (surge onset)",
    )
    mc2.metric(
        f"P(overcrowded) within {forecast_hours}h",
        f"{100 * p_crowd:.1f}%",
    )
    mc3.metric("Monte Carlo Trajectories", "2,000")

    # Fan chart + crowding probability timeline
    fig_fc, (ax_fc, ax_pc) = plt.subplots(
        1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]},
    )

    tg = sim_result["time_grid"]
    ax_fc.fill_between(tg, sim_result["ci_low"], sim_result["ci_high"],
                       alpha=0.25, color=BLUE, label="95% CI")
    ax_fc.plot(tg, sim_result["mean"], color=BLUE, linewidth=2,
               label="Mean occupancy")
    ax_fc.axhline(user_capacity, color=RED, linestyle="--", linewidth=1.5,
                  label=f"Capacity ({user_capacity})")
    ax_fc.set_xlabel("Hours from now")
    ax_fc.set_ylabel("Occupancy")
    ax_fc.set_title(f"{forecast_hours}h Occupancy Forecast (Monte Carlo)")
    ax_fc.legend()

    ax_pc.plot(tg, sim_result["p_overcrowded"] * 100, color=RED, linewidth=2)
    ax_pc.fill_between(tg, sim_result["p_overcrowded"] * 100,
                       alpha=0.2, color=RED)
    ax_pc.set_xlabel("Hours")
    ax_pc.set_ylabel("P(overcrowded) %")
    ax_pc.set_title("Crowding Probability")
    ax_pc.set_ylim(0, 100)

    plt.tight_layout()
    st.pyplot(fig_fc)
    plt.close(fig_fc)

    # Sensitivity to capacity
    st.subheader("Sensitivity: What Drives Crowding Risk?")
    st.markdown(
        "The same Monte Carlo output, re-evaluated under different capacity "
        "thresholds, reveals which assumptions dominate the crowding forecast. "
        "This is a direct demonstration of decision sensitivity."
    )

    caps = [int(user_capacity * 0.8), user_capacity, int(user_capacity * 1.2)]
    labels_s, vals_s, colors_s = [], [], []
    color_map = {0: RED, 1: BLUE, 2: GREEN}

    for i, c in enumerate(caps):
        p = 100 * float(np.mean(peak_occ > c))
        tag = " (current)" if c == user_capacity else ""
        labels_s.append(f"Capacity {c}{tag}: {p:.1f}%")
        vals_s.append(p)
        colors_s.append(color_map[i])

    fig_sens, ax_sens = plt.subplots(figsize=(9, 2.8))
    ax_sens.barh(labels_s, vals_s, color=colors_s, alpha=0.7, height=0.5)
    ax_sens.set_xlabel("P(overcrowded) %")
    ax_sens.set_xlim(0, 100)
    ax_sens.set_title("Capacity Assumption Drives Crowding Risk")
    plt.tight_layout()
    st.pyplot(fig_sens)
    plt.close(fig_sens)


# ========================  TAB 4: MODEL EVALUATION  ========================
with tab_eval:
    st.header("Model Evaluation & Failure Modes")

    eval_t1, eval_t2, eval_t3, eval_t4 = st.tabs([
        "Stationary vs. Windowed",
        "MLE vs. Bayesian",
        "Calibration & Scoring",
        "Failure Modes",
    ])

    # --- Stationary vs Windowed ---
    with eval_t1:
        st.subheader("Stationary vs. Windowed Model")
        st.markdown(
            "The **stationary** model uses all historical data equally. "
            "The **windowed** model (14-day window) adapts to regime changes. "
            "During surges, the windowed model tracks the elevated arrival "
            "rate while the stationary model is anchored to the historical "
            "average."
        )

        h_stat = default_model.history
        h_wind = w_history

        fig_cmp, ax_cmp = plt.subplots(figsize=(13, 4.5))
        t_s = np.array(h_stat.times)
        t_w = np.array(h_wind.times)

        ax_cmp.fill_between(
            t_s[1:], np.array(h_stat.ci_lows)[1:],
            np.array(h_stat.ci_highs)[1:], alpha=0.15, color=BLUE,
        )
        ax_cmp.plot(t_s[1:], np.array(h_stat.means)[1:], color=BLUE,
                    linewidth=2, label="Stationary")
        ax_cmp.plot(t_w, np.array(h_wind.means), color=GREEN,
                    linewidth=2, label="Windowed (14d)")
        ax_cmp.scatter(t_s[1:], np.array(h_stat.observed_counts)[1:],
                       s=10, color=GRAY, alpha=0.35, zorder=3,
                       label="Observed")
        for s, e in config.surge_windows:
            ax_cmp.axvspan(s, e, alpha=0.1, color=ORANGE)

        ax_cmp.set_xlabel("Day")
        ax_cmp.set_ylabel("\u03BB (adm/day)")
        ax_cmp.set_title("Stationary vs. Windowed Bayesian Model")
        ax_cmp.legend()
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

    # --- MLE vs Bayesian ---
    with eval_t2:
        st.subheader("Maximum Likelihood vs. Bayesian Estimation")
        st.markdown(
            "MLE gives a point estimate: "
            "$\\hat{\\lambda}_{\\text{MLE}} = \\sum k / T$. "
            "The Bayesian posterior is a full distribution. With limited data, "
            "the Bayesian credible interval is wider (more honest). As data "
            "accumulates, both converge (**Bernstein\u2013von Mises theorem**)."
        )

        mle_res = run_mle_comparison(daily_counts)

        fig_mle, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(14, 4))
        days_arr = mle_res["days"]

        ax_m1.plot(days_arr, mle_res["mle_means"], color=RED, linewidth=1.5,
                   label="MLE (frequentist)", alpha=0.8)
        ax_m1.plot(days_arr, mle_res["bayes_means"], color=BLUE,
                   linewidth=1.5, label="Bayesian posterior mean", alpha=0.8)
        for s, e in config.surge_windows:
            ax_m1.axvspan(s, e, alpha=0.1, color=ORANGE)
        ax_m1.set_xlabel("Day")
        ax_m1.set_ylabel("\u03BB estimate")
        ax_m1.set_title("Point Estimates Converge")
        ax_m1.legend()

        mle_w = mle_res["mle_ci_hi"] - mle_res["mle_ci_lo"]
        bay_w = mle_res["bayes_ci_hi"] - mle_res["bayes_ci_lo"]
        ax_m2.plot(days_arr, mle_w, color=RED, linewidth=1.5,
                   label="Frequentist 95% CI width")
        ax_m2.plot(days_arr, bay_w, color=BLUE, linewidth=1.5,
                   label="Bayesian 95% CI width")
        ax_m2.set_xlabel("Day")
        ax_m2.set_ylabel("CI Width")
        ax_m2.set_title("Interval Width Comparison")
        ax_m2.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_mle)
        plt.close(fig_mle)

    # --- Calibration & Log Scores ---
    with eval_t3:
        st.subheader("Formal Model Comparison: Log Predictive Score")
        st.markdown(
            "One-step-ahead **log predictive score**: for each day $t$, "
            "we compute $\\log P(y_t \\mid y_{1:t-1})$ under each model. "
            "Higher total score = better calibrated model. This is a "
            "*strict proper scoring rule* \u2014 it cannot be gamed."
        )

        stat_a = np.array(default_model.history.alphas)
        stat_b = np.array(default_model.history.betas)
        wind_a = np.array(w_history.alphas)
        wind_b = np.array(w_history.betas)
        score_res = compute_log_scores(
            daily_counts, stat_a, stat_b, wind_a, wind_b,
        )

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Stationary Score", f"{score_res['stationary_total']:.1f}")
        sc2.metric("Windowed Score", f"{score_res['windowed_total']:.1f}")
        winner = "Windowed" if score_res["difference"] > 0 else "Stationary"
        sc3.metric(
            "Winner", winner,
            f"+{abs(score_res['difference']):.1f} log-score units",
        )

        fig_sc, ax_sc = plt.subplots(figsize=(13, 4))
        ax_sc.plot(score_res["days"], score_res["cumulative_stationary"],
                   color=BLUE, linewidth=2, label="Stationary (cumulative)")
        ax_sc.plot(score_res["days"], score_res["cumulative_windowed"],
                   color=GREEN, linewidth=2, label="Windowed (cumulative)")
        for s, e in config.surge_windows:
            ax_sc.axvspan(s, e, alpha=0.1, color=ORANGE)
        ax_sc.set_xlabel("Day")
        ax_sc.set_ylabel("Cumulative Log Predictive Score")
        ax_sc.set_title(
            "Model Comparison: Windowed Model Wins During Surges"
        )
        ax_sc.legend()
        plt.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

        # Posterior predictive check
        st.subheader("Posterior Predictive Check")
        k_vals, pmf = default_model.posterior_predictive_pmf(1.0)

        fig_pp, ax_pp = plt.subplots(figsize=(9, 3.5))
        ax_pp.hist(daily_counts, bins=np.arange(-0.5, 40.5, 1), density=True,
                   alpha=0.5, color=BLUE, label="Empirical",
                   edgecolor="white")
        ax_pp.plot(k_vals, pmf, "o-", color=RED, markersize=3, linewidth=1.5,
                   label="NegBin posterior predictive")
        ax_pp.set_xlabel("Daily admissions")
        ax_pp.set_ylabel("Probability")
        ax_pp.set_title("Posterior Predictive vs. Empirical")
        ax_pp.legend()
        plt.tight_layout()
        st.pyplot(fig_pp)
        plt.close(fig_pp)

    # --- Failure Modes ---
    with eval_t4:
        st.subheader("Failure-Mode Analysis")
        st.markdown(
            "We systematically identify **5 failure modes** where our model's "
            "assumptions break. For each, we detect the violation, quantify "
            "its severity, and widen uncertainty accordingly."
        )

        reports, penalty = run_failure_modes(
            daily_counts, data["los_hours"], data["census_hourly"],
            config.surge_windows, config.capacity, data["discharges"],
        )

        st.metric("Combined CI Widening Factor", f"\u00d7{penalty:.2f}")

        severity_icon = {
            "low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534",
        }

        for r in reports:
            icon = severity_icon.get(r.severity, "\u26AA")
            det = "DETECTED" if r.detected else "not detected"
            with st.expander(
                f"{icon} **{r.name}** ({r.severity}) \u2014 {det}"
            ):
                st.markdown(f"**Assumption:** {r.assumption}")
                st.markdown(f"**How it breaks:** {r.how_it_breaks}")
                st.markdown(f"**Consequence:** {r.consequence}")
                st.markdown(f"**Mitigation:** {r.mitigation}")
                st.markdown(f"**Evidence:** {r.evidence}")
                st.markdown(
                    f"**CI widening:** \u00d7{r.confidence_penalty:.2f}"
                )


# ========================  TAB 5: CS109 CONCEPTS  ==========================
with tab_concepts:
    st.header("CS109 Concepts Demonstrated")
    st.markdown(
        "This project demonstrates **16 concepts** from the CS109 curriculum, "
        "spanning probability foundations, Bayesian inference, frequentist "
        "methods, simulation, model evaluation, and decision-making under "
        "uncertainty."
    )

    concepts = [
        ("Random Variables",
         "$N_t$, $L$, $O_t$, $\\lambda$",
         "Foundation of the model"),
        ("Distributions",
         "Poisson, Gamma, NegBin, LogNormal",
         "Each justified probabilistically"),
        ("Conditional Probability",
         "$P(N_t \\mid \\lambda)$, $P(\\lambda \\mid \\text{data})$",
         "Core of Bayesian update"),
        ("Bayes' Theorem",
         "Prior \u00d7 Likelihood = Posterior",
         "Gamma\u2013Poisson conjugacy"),
        ("Posterior Predictive",
         "Integrate out $\\lambda$ \u2192 NegBin",
         "Predictions with parameter uncertainty"),
        ("Conjugate Priors",
         "Gamma\u2013Poisson \u2192 exact posterior",
         "No MCMC needed"),
        ("Law of Total Variance",
         "Var = E[Var|$\\lambda$] + Var[E|$\\lambda$]",
         "Epistemic vs. aleatoric"),
        ("Monte Carlo Simulation",
         "2,000+ occupancy trajectories",
         "Full uncertainty propagation"),
        ("Maximum Likelihood",
         "$\\hat{\\lambda} = \\sum k / T$",
         "Compared against Bayesian"),
        ("Central Limit Theorem",
         "Frequentist CI via CLT",
         "MLE interval construction"),
        ("Information Theory",
         "KL divergence",
         "Quantifies learning per observation"),
        ("Hypothesis Testing",
         "Posterior predictive p-values",
         "Bayesian anomaly detection"),
        ("Model Comparison",
         "Log predictive score",
         "Proper scoring rule"),
        ("Calibration",
         "Coverage, PIT histograms",
         "Are intervals honest?"),
        ("Sensitivity Analysis",
         "Assumptions \u2192 P(overcrowded)",
         "Which choices matter?"),
        ("Prior Sensitivity",
         "3 priors \u2192 convergence",
         "Evidence overwhelms priors"),
    ]

    cols_per_row = 4
    for row_start in range(0, len(concepts), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(concepts):
                name, detail, desc = concepts[idx]
                col.markdown(
                    f"**{idx + 1}. {name}**  \n"
                    f"{detail}  \n"
                    f"_{desc}_"
                )

    st.divider()

    # Prior sensitivity demo
    st.subheader("Prior Sensitivity: Evidence Overwhelms Prior Beliefs")
    st.markdown(
        "Three very different priors \u2014 uninformative, weakly informative, "
        "and deliberately wrong \u2014 all converge to the same posterior "
        "after sufficient data. This is a fundamental Bayesian result."
    )

    sens_histories = run_prior_sensitivity(daily_counts)

    fig_ps, ax_ps = plt.subplots(figsize=(13, 4))
    color_ps = [RED, BLUE, GREEN]
    for (name, hist), color in zip(sens_histories.items(), color_ps):
        t = np.array(hist.times)
        m = np.array(hist.means)
        ax_ps.plot(t[1:], m[1:], color=color, linewidth=2, label=name)
    for s, e in config.surge_windows:
        ax_ps.axvspan(s, e, alpha=0.1, color=ORANGE)
    ax_ps.set_xlabel("Day")
    ax_ps.set_ylabel("\u03BB estimate")
    ax_ps.set_title("Prior Sensitivity: All Priors Converge to Same Posterior")
    ax_ps.legend()
    plt.tight_layout()
    st.pyplot(fig_ps)
    plt.close(fig_ps)

    st.divider()

    # Ethical reflection
    st.subheader("Ethical Reflection")
    st.markdown(
        """
- **Synthetic data** used for ethical reasons \u2014 real ICU data contains
  protected health information
- **Never a point estimate** \u2014 every prediction includes a credible interval
- **5 failure modes documented** with detection criteria and uncertainty widening
- **Not for autonomous deployment** \u2014 probabilistic forecasts should
  augment, never replace, clinical judgment
- **Goodhart's Law** flagged as an irreducible limitation: any deployed
  model that influences the system it measures risks becoming unreliable
"""
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "BUICU \u2014 CS109 Challenge Project  |  "
    "Belief Updating for ICU Crowding Under Uncertainty  |  "
    "Built with Bayesian inference, not black-box ML"
)
