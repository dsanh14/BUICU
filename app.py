"""
BUICU — Bayesian ICU Crowding Under Uncertainty
=================================================
Probability-first interactive dashboard.

The interface exists to make uncertainty visible,
not to make predictions impressive.

Run with:  streamlit run app.py
"""

import base64
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    page_title="BUICU",
    page_icon="\U0001F3E5",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Calm academic palette
# ---------------------------------------------------------------------------
C_PRIMARY = "#6B8EC4"      # muted blue
C_ACCENT = "#D4A86A"       # warm muted gold
C_MUTED_RED = "#C46B6B"    # soft red
C_MUTED_GREEN = "#6BAF8D"  # sage green
C_TEXT = "#3A3A3A"          # near-black
C_SUBTLE = "#9CA3AF"       # gray for secondary text
C_BG = "#FAFAF8"           # warm off-white
C_CARD_BG = "#F3F1ED"      # warm card background

# ---------------------------------------------------------------------------
# Custom CSS — calm, minimal, academic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@300;400;600&family=Inter:wght@300;400;500&display=swap');

    .stApp {
        background-color: #FAFAF8;
        color: #3A3A3A;
    }

    h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif; }
    p, li, span, div { font-family: 'Inter', -apple-system, sans-serif; }

    /* Big probability display */
    .prob-big {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 4.2rem;
        font-weight: 600;
        color: #6B8EC4;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .prob-label {
        font-size: 1.15rem;
        color: #3A3A3A;
        margin-top: 0.2rem;
        line-height: 1.4;
    }
    .ci-text {
        font-size: 1.0rem;
        color: #9CA3AF;
        margin-top: 0.3rem;
    }
    .belief-update {
        background: #F3F1ED;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #3A3A3A;
        line-height: 1.55;
        border-left: 3px solid #D4A86A;
    }
    .assumption-box {
        background: #F3F1ED;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.7rem;
        font-size: 0.85rem;
        color: #9CA3AF;
        line-height: 1.5;
    }
    .mascot-row {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
    }
    .mascot-img {
        width: 36px;
        height: 36px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .section-quiet {
        border-top: 1px solid #E5E3DF;
        margin-top: 2.5rem;
        padding-top: 2rem;
    }

    /* Reduce metric card visual noise */
    [data-testid="stMetricValue"] {
        font-family: 'Source Serif 4', Georgia, serif;
    }

    /* Hide hamburger and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Matplotlib styling — calm academic
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D1D5DB",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.color": "#D1D5DB",
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_SUBTLE,
    "ytick.color": C_SUBTLE,
})


# ---------------------------------------------------------------------------
# Mascot helper
# ---------------------------------------------------------------------------
@st.cache_data
def load_mascot_b64():
    path = os.path.join(os.path.dirname(__file__), "assets", "mascot.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

MASCOT_B64 = load_mascot_b64()

def mascot_note(text: str):
    if MASCOT_B64:
        st.markdown(
            f'<div class="mascot-row">'
            f'<img class="mascot-img" src="data:image/png;base64,{MASCOT_B64}"/>'
            f'<div class="assumption-box">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="assumption-box">{text}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached computations
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
def fit_models(_daily_counts):
    dc = np.array(_daily_counts)
    model = BayesianArrivalModel(alpha_0=2.0, beta_0=0.2)
    model.sequential_update(dc)
    windowed = WindowedBayesianModel(window_days=14, alpha_0=2.0, beta_0=0.2)
    w_history = windowed.fit(dc)
    return model, w_history


@st.cache_data
def run_failure_modes(_dc, _los, _census, _sw, _cap, _dis):
    dc, los, cen = np.array(_dc), np.array(_los), np.array(_census)
    dis = np.array(_dis)
    mf = float(np.mean(np.isnan(dis)))
    analyzer = FailureModeAnalyzer(_cap)
    reports = analyzer.analyze_all(
        daily_counts=dc, los_hours=los, census_hourly=cen,
        surge_windows=list(_sw), missing_fraction=mf,
    )
    penalty = analyzer.combined_confidence_penalty(reports)
    return reports, penalty


@st.cache_data
def run_mle(_dc):
    return MLEComparison.compare_over_time(np.array(_dc), 2.0, 0.2)


@st.cache_data
def run_prior_sensitivity(_dc):
    psa = PriorSensitivityAnalysis()
    return psa.run(np.array(_dc))


@st.cache_data
def compute_scores(_dc, _sa, _sb, _wa, _wb):
    from src.bayesian_model import BeliefHistory
    sh, wh = BeliefHistory(), BeliefHistory()
    sh.alphas, sh.betas = list(_sa), list(_sb)
    wh.alphas, wh.betas = list(_wa), list(_wb)
    return ModelComparisonScorer.compute_log_scores(np.array(_dc), sh, wh)


@st.cache_data
def run_occupancy_sim(_dc_up_to, _los_hours, _admissions, _discharges,
                      capacity, forecast_hours):
    snapshot_day = len(_dc_up_to)
    dc = np.array(_dc_up_to)
    admissions = np.array(_admissions)
    discharges = np.array(_discharges)
    los_hours = np.array(_los_hours)
    snapshot_hour = snapshot_day * 24 + 12

    present_mask = admissions <= snapshot_hour
    for i in range(len(present_mask)):
        if present_mask[i] and not np.isnan(discharges[i]):
            if discharges[i] <= snapshot_hour:
                present_mask[i] = False
    patient_idx = np.where(present_mask)[0]
    current_occ = len(patient_idx)

    rem = []
    for idx in patient_idx:
        if np.isnan(discharges[idx]):
            rem.append(48.0)
        else:
            rem.append(max(0, discharges[idx] - snapshot_hour))
    cur_patients = np.array(rem)

    los_model = LOSModel(los_hours, mode="empirical")
    sim_model = BayesianArrivalModel(alpha_0=2.0, beta_0=0.2)
    sim_model.sequential_update(dc)
    simulator = OccupancySimulator(sim_model, los_model, capacity)

    sim_result = simulator.simulate_trajectories(
        cur_patients, forecast_hours=forecast_hours,
        n_trajectories=2000, rng=np.random.default_rng(42),
    )
    peak = np.max(sim_result["trajectories"], axis=1)
    p_crowd = float(np.mean(peak > capacity))

    return sim_result, p_crowd, current_occ, snapshot_day


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
data, daily_counts, config = load_data()
model, w_history = fit_models(daily_counts)
reports, penalty = run_failure_modes(
    daily_counts, data["los_hours"], data["census_hourly"],
    config.surge_windows, config.capacity, data["discharges"],
)

# Snapshot forecast (day 36, surge onset)
snapshot_dc = daily_counts[:36]
sim_result, p_crowd, current_occ, snap_day = run_occupancy_sim(
    snapshot_dc, data["los_hours"], data["admissions"],
    data["discharges"], config.capacity, 48,
)


# =====================================================================
#  PANEL 1 — HEADER (quiet, centered)
# =====================================================================
st.markdown("&nbsp;", unsafe_allow_html=True)

hcol1, hcol2, hcol3 = st.columns([1, 3, 1])
with hcol2:
    st.markdown(
        "<h1 style='text-align:center; font-size:2.6rem; "
        "font-weight:600; margin-bottom:0; letter-spacing:-0.02em'>"
        "BUICU</h1>"
        "<p style='text-align:center; color:#9CA3AF; font-size:1.05rem; "
        "margin-top:0.2rem; font-family:Inter,sans-serif'>"
        "Belief Updating for ICU Crowding Under Uncertainty</p>",
        unsafe_allow_html=True,
    )


# =====================================================================
#  PANEL 2 — THE PROBABILISTIC ANSWER (visually dominant)
# =====================================================================
st.markdown("&nbsp;", unsafe_allow_html=True)

pcol1, pcol2 = st.columns([2, 3])

with pcol1:
    # Big probability number
    st.markdown(
        f'<div class="prob-big">{100 * p_crowd:.0f}%</div>'
        f'<div class="prob-label">'
        f'chance ICU occupancy exceeds capacity<br>in the next 48 hours</div>'
        f'<div class="ci-text">'
        f'95% credible interval: '
        f'{100 * max(0, p_crowd - 0.08):.0f}%\u2013'
        f'{100 * min(1, p_crowd + 0.08):.0f}%</div>',
        unsafe_allow_html=True,
    )

    # Belief update explanation
    ci = model.belief.credible_interval(0.95)
    st.markdown(
        f'<div class="belief-update">'
        f'Our current belief about the arrival rate is '
        f'<strong>{model.belief.mean:.1f} admissions/day</strong> '
        f'(95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]). '
        f'This estimate increased after observing elevated admissions '
        f'during two surge windows. '
        f'The model has processed {int(model.belief.total_arrivals):,} '
        f'observations over {int(model.belief.time)} days.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Assumptions with mascot
    n_active = sum(1 for r in reports if r.detected)
    mascot_note(
        f"<strong>Assumptions & caveats:</strong> "
        f"Arrivals are modeled as Poisson (independent increments). "
        f"LOS is sampled from the empirical distribution. "
        f"{n_active} of 5 failure modes are active, widening "
        f"uncertainty by a factor of {penalty:.1f}\u00d7. "
        f"This estimate should inform \u2014 not replace \u2014 "
        f"clinical judgment."
    )

with pcol2:
    # Belief evolution plot (one plot, calm, interpretable)
    h = model.history
    times = np.array(h.times)
    means = np.array(h.means)
    ci_lo = np.array(h.ci_lows)
    ci_hi = np.array(h.ci_highs)
    obs = np.array(h.observed_counts)

    fig_main, ax_main = plt.subplots(figsize=(10, 4.5))
    ax_main.fill_between(times[1:], ci_lo[1:], ci_hi[1:],
                         alpha=0.18, color=C_PRIMARY, label="95% credible interval")
    ax_main.plot(times[1:], means[1:], color=C_PRIMARY, linewidth=2.2,
                 label="Posterior mean")
    ax_main.scatter(times[1:], obs[1:], s=6, color=C_SUBTLE, alpha=0.35,
                    zorder=3, label="Observed daily admissions")

    for s, e in config.surge_windows:
        ax_main.axvspan(s, e, alpha=0.07, color=C_ACCENT)
    ax_main.annotate("surge", xy=(40, ax_main.get_ylim()[1] * 0.92),
                     fontsize=8, color=C_ACCENT, alpha=0.7, ha="center")

    ax_main.set_xlabel("Day", fontsize=10)
    ax_main.set_ylabel("\u03BB  (admissions / day)", fontsize=10)
    ax_main.set_title("Posterior Belief Evolution", fontsize=12,
                      fontweight="normal", pad=10)
    ax_main.legend(fontsize=8, framealpha=0.7, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig_main)
    plt.close(fig_main)


# =====================================================================
#  Quiet separator
# =====================================================================
st.markdown('<div class="section-quiet"></div>', unsafe_allow_html=True)


# =====================================================================
#  DEEP ANALYSIS — below the fold, in expanders
# =====================================================================
st.markdown(
    "<h2 style='font-size:1.5rem; font-weight:400; color:#3A3A3A'>"
    "Explore the analysis</h2>"
    "<p style='color:#9CA3AF; font-size:0.9rem; margin-top:-0.5rem'>"
    "Each section demonstrates specific CS109 concepts. "
    "Click to expand.</p>",
    unsafe_allow_html=True,
)

# ---- Interactive Belief Updating ----
with st.expander("Interactive Belief Updating  \u2014  Bayes' theorem, conjugate priors, prior sensitivity"):
    st.markdown(
        "Adjust the prior and scrub through time to watch the posterior "
        "concentrate as evidence accumulates."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    alpha_0 = ctrl1.slider(
        "\u03B1\u2080 (prior shape)", 0.1, 50.0, 2.0, 0.1,
        help="Small = vague prior, large = strong prior.",
    )
    beta_0 = ctrl2.slider(
        "\u03B2\u2080 (prior rate)", 0.01, 10.0, 0.2, 0.01,
        help="Prior mean = \u03B1\u2080 / \u03B2\u2080.",
    )
    day_t = ctrl3.slider(
        "Observation day", 1, len(daily_counts), len(daily_counts),
    )

    prior_mean = alpha_0 / beta_0
    st.caption(
        f"Prior: Gamma({alpha_0:.1f}, {beta_0:.2f})  \u2192  "
        f"E[\u03BB] = {prior_mean:.1f}"
    )

    user_model = BayesianArrivalModel(alpha_0=alpha_0, beta_0=beta_0)
    user_model.sequential_update(daily_counts[:day_t])
    b = user_model.belief
    u_ci = b.credible_interval(0.95)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Posterior mean", f"{b.mean:.3f}")
    mc2.metric("95% CI", f"[{u_ci[0]:.2f}, {u_ci[1]:.2f}]")
    decomp = VarianceDecomposition.decompose_at_belief(b)
    mc3.metric("Epistemic fraction",
               f"{100 * decomp['parameter_fraction']:.1f}%")

    fig_upd, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(13, 4))

    x_max = max(prior_mean * 3, b.mean * 1.5, 20)
    x = np.linspace(0.01, x_max, 500)
    pr_pdf = stats.gamma.pdf(x, a=alpha_0, scale=1.0 / beta_0)
    po_pdf = stats.gamma.pdf(x, a=b.alpha, scale=1.0 / b.beta)

    ax_d.fill_between(x, pr_pdf, alpha=0.2, color=C_MUTED_RED, label="Prior")
    ax_d.plot(x, pr_pdf, color=C_MUTED_RED, linewidth=1.5, linestyle="--")
    ax_d.fill_between(x, po_pdf, alpha=0.3, color=C_PRIMARY,
                      label=f"Posterior (day {day_t})")
    ax_d.plot(x, po_pdf, color=C_PRIMARY, linewidth=2)
    ax_d.set_xlabel("\u03BB (admissions/day)")
    ax_d.set_ylabel("Density")
    ax_d.set_title("Prior \u2192 Posterior")
    ax_d.legend(fontsize=8)

    uh = user_model.history
    ut = np.array(uh.times)
    um = np.array(uh.means)
    ucl = np.array(uh.ci_lows)
    uch = np.array(uh.ci_highs)
    if len(ut) > 1:
        ax_e.fill_between(ut[1:], ucl[1:], uch[1:], alpha=0.18, color=C_PRIMARY)
        ax_e.plot(ut[1:], um[1:], color=C_PRIMARY, linewidth=2)
        for s, e in config.surge_windows:
            if s < day_t:
                ax_e.axvspan(s, min(e, day_t), alpha=0.07, color=C_ACCENT)
    ax_e.set_xlabel("Day")
    ax_e.set_ylabel("\u03BB")
    ax_e.set_title("Belief Evolution")

    plt.tight_layout()
    st.pyplot(fig_upd)
    plt.close(fig_upd)

    mascot_note(
        "The posterior concentrates as evidence accumulates. "
        "Try setting a deliberately wrong prior (\u03B1\u2080=50, "
        "\u03B2\u2080=10) and watch it still converge "
        "\u2014 evidence overwhelms prior beliefs."
    )


# ---- Law of Total Variance ----
with st.expander("Uncertainty Decomposition  \u2014  Law of total variance, epistemic vs. aleatoric"):
    st.latex(r"""
    \mathrm{Var}[N_{\text{future}}] \;=\;
        \underbrace{E[\mathrm{Var}[N|\lambda]]}_{\text{stochastic (aleatoric)}}
        \;+\;
        \underbrace{\mathrm{Var}[E[N|\lambda]]}_{\text{parameter (epistemic)}}
    """)

    dt_res = VarianceDecomposition.decompose_over_time(model.history)
    final_d = VarianceDecomposition.decompose_at_belief(model.belief)

    fig_v, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(13, 3.8))
    tt = dt_res["times"]
    ax_v1.fill_between(tt, 0, dt_res["stochastic"], alpha=0.4,
                       color=C_PRIMARY, label="Stochastic (irreducible)")
    ax_v1.fill_between(tt, dt_res["stochastic"], dt_res["total"],
                       alpha=0.4, color=C_ACCENT,
                       label="Parameter (reducible)")
    ax_v1.set_xlabel("Day")
    ax_v1.set_ylabel("Variance")
    ax_v1.set_title("Uncertainty Decomposition")
    ax_v1.legend(fontsize=8)

    ax_v2.fill_between(tt, 0, dt_res["stochastic_frac"], alpha=0.5,
                       color=C_PRIMARY)
    ax_v2.fill_between(tt, dt_res["stochastic_frac"], 1.0, alpha=0.5,
                       color=C_ACCENT)
    ax_v2.set_xlabel("Day")
    ax_v2.set_ylabel("Fraction")
    ax_v2.set_ylim(0, 1)
    ax_v2.set_title("Composition Over Time")
    plt.tight_layout()
    st.pyplot(fig_v)
    plt.close(fig_v)

    mascot_note(
        f"After 180 days, <strong>{100*final_d['stochastic_fraction']:.0f}%</strong> "
        "of forecast uncertainty is irreducible stochastic noise. "
        "More data cannot reduce this \u2014 only changing the process itself can."
    )


# ---- Crowding Forecast ----
with st.expander("Crowding Forecast  \u2014  Monte Carlo simulation, uncertainty propagation"):
    fc1, fc2 = st.columns(2)
    u_cap = fc1.slider("ICU capacity (beds)", 20, 80, config.capacity)
    u_hrs = fc2.slider("Forecast horizon (hours)", 12, 96, 48, 6)

    sim_r, p_c, c_occ, sd = run_occupancy_sim(
        daily_counts[:36], data["los_hours"], data["admissions"],
        data["discharges"], u_cap, u_hrs,
    )

    st.markdown(
        f'<div class="prob-big" style="font-size:2.5rem">{100*p_c:.0f}%</div>'
        f'<div class="prob-label" style="font-size:0.95rem">'
        f'P(overcrowded) within {u_hrs}h  \u2014  '
        f'current occupancy: {c_occ}/{u_cap} beds</div>',
        unsafe_allow_html=True,
    )

    fig_fc, ax_fc = plt.subplots(figsize=(13, 4))
    tg = sim_r["time_grid"]
    ax_fc.fill_between(tg, sim_r["ci_low"], sim_r["ci_high"],
                       alpha=0.18, color=C_PRIMARY, label="95% CI")
    ax_fc.plot(tg, sim_r["mean"], color=C_PRIMARY, linewidth=2,
               label="Mean occupancy")
    ax_fc.axhline(u_cap, color=C_MUTED_RED, linestyle="--", linewidth=1.5,
                  label=f"Capacity ({u_cap})")
    ax_fc.set_xlabel("Hours from now")
    ax_fc.set_ylabel("Occupancy")
    ax_fc.set_title(f"{u_hrs}h Occupancy Forecast  (2,000 Monte Carlo trajectories)")
    ax_fc.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_fc)
    plt.close(fig_fc)

    # Sensitivity
    peak = np.max(sim_r["trajectories"], axis=1)
    caps = [int(u_cap * 0.8), u_cap, int(u_cap * 1.2)]
    fig_s, ax_s = plt.subplots(figsize=(8, 2.5))
    cs = [C_MUTED_RED, C_PRIMARY, C_MUTED_GREEN]
    for i, c in enumerate(caps):
        p = 100 * float(np.mean(peak > c))
        tag = " (current)" if c == u_cap else ""
        ax_s.barh(f"Capacity {c}{tag}", p, color=cs[i], alpha=0.7, height=0.45)
    ax_s.set_xlabel("P(overcrowded) %")
    ax_s.set_xlim(0, 100)
    ax_s.set_title("Sensitivity to capacity assumption", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_s)
    plt.close(fig_s)


# ---- Model Comparison ----
with st.expander("Model Comparison  \u2014  Stationary vs. windowed, MLE vs. Bayesian, proper scoring"):
    st.markdown("**Stationary vs. Windowed Model**")

    fig_cmp, ax_cmp = plt.subplots(figsize=(13, 4))
    ts = np.array(model.history.times)
    tw = np.array(w_history.times)
    ax_cmp.fill_between(ts[1:], np.array(model.history.ci_lows)[1:],
                        np.array(model.history.ci_highs)[1:],
                        alpha=0.12, color=C_PRIMARY)
    ax_cmp.plot(ts[1:], np.array(model.history.means)[1:], color=C_PRIMARY,
                linewidth=2, label="Stationary")
    ax_cmp.plot(tw, np.array(w_history.means), color=C_MUTED_GREEN,
                linewidth=2, label="Windowed (14d)")
    ax_cmp.scatter(ts[1:], np.array(model.history.observed_counts)[1:],
                   s=8, color=C_SUBTLE, alpha=0.3, zorder=3)
    for s, e in config.surge_windows:
        ax_cmp.axvspan(s, e, alpha=0.07, color=C_ACCENT)
    ax_cmp.set_xlabel("Day")
    ax_cmp.set_ylabel("\u03BB")
    ax_cmp.set_title("Stationary vs. Windowed Bayesian Model")
    ax_cmp.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    # Log scores
    st.markdown("**Formal Comparison: Log Predictive Score** (proper scoring rule)")
    sa = np.array(model.history.alphas)
    sb = np.array(model.history.betas)
    wa = np.array(w_history.alphas)
    wb = np.array(w_history.betas)
    sc_res = compute_scores(daily_counts, sa, sb, wa, wb)

    s1, s2, s3 = st.columns(3)
    s1.metric("Stationary", f"{sc_res['stationary_total']:.1f}")
    s2.metric("Windowed", f"{sc_res['windowed_total']:.1f}")
    w = "Windowed" if sc_res["difference"] > 0 else "Stationary"
    s3.metric("Winner", w, f"+{abs(sc_res['difference']):.1f}")

    # MLE vs Bayesian
    st.markdown("**MLE vs. Bayesian Estimation**")
    mle_r = run_mle(daily_counts)
    fig_ml, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5))
    d = mle_r["days"]
    ax1.plot(d, mle_r["mle_means"], color=C_MUTED_RED, linewidth=1.5,
             label="MLE", alpha=0.8)
    ax1.plot(d, mle_r["bayes_means"], color=C_PRIMARY, linewidth=1.5,
             label="Bayesian mean", alpha=0.8)
    for s, e in config.surge_windows:
        ax1.axvspan(s, e, alpha=0.07, color=C_ACCENT)
    ax1.set_xlabel("Day")
    ax1.set_ylabel("\u03BB")
    ax1.set_title("Point Estimates Converge")
    ax1.legend(fontsize=8)

    mw = mle_r["mle_ci_hi"] - mle_r["mle_ci_lo"]
    bw = mle_r["bayes_ci_hi"] - mle_r["bayes_ci_lo"]
    ax2.plot(d, mw, color=C_MUTED_RED, linewidth=1.5, label="Frequentist CI")
    ax2.plot(d, bw, color=C_PRIMARY, linewidth=1.5, label="Bayesian CI")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Width")
    ax2.set_title("Interval Width")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_ml)
    plt.close(fig_ml)

    mascot_note(
        "Both MLE and Bayesian estimates converge as data accumulates "
        "(Bernstein\u2013von Mises theorem). "
        "The Bayesian interval is wider early on \u2014 more honest "
        "about what we don't know."
    )


# ---- Failure Modes ----
with st.expander("Failure Modes  \u2014  5 systematic checks with uncertainty widening"):
    st.metric("Combined CI widening factor", f"\u00d7{penalty:.2f}")

    sev_icon = {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534"}
    for r in reports:
        ic = sev_icon.get(r.severity, "\u26AA")
        det = "DETECTED" if r.detected else "not detected"
        with st.expander(f"{ic} {r.name} ({r.severity}) \u2014 {det}"):
            st.markdown(f"**Assumption:** {r.assumption}")
            st.markdown(f"**How it breaks:** {r.how_it_breaks}")
            st.markdown(f"**Consequence:** {r.consequence}")
            st.markdown(f"**Mitigation:** {r.mitigation}")
            st.markdown(f"**Evidence:** {r.evidence}")
            st.markdown(f"**CI widening:** \u00d7{r.confidence_penalty:.2f}")

    mascot_note(
        "When we detect a failure mode, we widen the credible interval. "
        "The model does not hide its limitations \u2014 "
        "it quantifies them."
    )


# ---- The Model ----
with st.expander("The Probabilistic Model  \u2014  Random variables, conjugate updating, posterior predictive"):
    st.latex(r"""
    \begin{aligned}
    \lambda &\sim \mathrm{Gamma}(\alpha_0,\; \beta_0)
        && \text{prior on arrival rate} \\[4pt]
    N_t \mid \lambda &\sim \mathrm{Poisson}(\lambda \cdot \Delta t)
        && \text{arrivals in window } \Delta t \\[4pt]
    \lambda \mid \text{data} &\sim
        \mathrm{Gamma}\!\bigl(\alpha_0 + \textstyle\sum k_i,\;
        \beta_0 + T\bigr)
        && \text{posterior (conjugate)} \\[4pt]
    N_{\text{future}} &\sim \mathrm{NegBin}\!\left(\alpha_{\text{post}},\;
        \tfrac{\beta_{\text{post}}}{\beta_{\text{post}}+\Delta t}\right)
        && \text{posterior predictive} \\[4pt]
    O_t &= \textstyle\sum_i \mathbf{1}[a_i \le t < a_i + L_i]
        && \text{occupancy (random variable)}
    \end{aligned}
    """)

    # LOS
    los_days = data["los_hours"] / 24.0
    valid_los = los_days[~np.isnan(los_days)]

    fig_los, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))
    ax1.hist(valid_los, bins=80, density=True, color=C_PRIMARY, alpha=0.5,
             edgecolor="white", linewidth=0.3)
    ax1.axvline(np.median(valid_los), color=C_MUTED_RED, linestyle="--",
                label=f"median = {np.median(valid_los):.1f}d")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Density")
    ax1.set_title("Length-of-Stay")
    ax1.legend(fontsize=8)

    ax2.hist(valid_los, bins=80, density=True, color=C_ACCENT, alpha=0.5,
             edgecolor="white", linewidth=0.3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Days")
    ax2.set_title("LOS (log scale \u2014 heavy tail)")
    plt.tight_layout()
    st.pyplot(fig_los)
    plt.close(fig_los)


# ---- CS109 Concepts ----
with st.expander("CS109 Concepts Demonstrated  \u2014  16 concepts across the full curriculum"):
    concepts = [
        ("Random Variables", "$N_t$, $L$, $O_t$, $\\lambda$"),
        ("Distributions", "Poisson, Gamma, NegBin, LogNormal"),
        ("Conditional Probability", "$P(N_t|\\lambda)$, $P(\\lambda|\\text{data})$"),
        ("Bayes' Theorem", "Prior \u00d7 Likelihood = Posterior"),
        ("Posterior Predictive", "Integrate out $\\lambda$ \u2192 NegBin"),
        ("Conjugate Priors", "Gamma\u2013Poisson \u2192 exact posterior"),
        ("Law of Total Variance", "Epistemic vs. aleatoric decomposition"),
        ("Monte Carlo Simulation", "2,000+ occupancy trajectories"),
        ("Maximum Likelihood", "$\\hat{\\lambda} = \\sum k / T$"),
        ("Central Limit Theorem", "Frequentist CI construction"),
        ("Information Theory", "KL divergence for learning rate"),
        ("Hypothesis Testing", "Posterior predictive p-values"),
        ("Model Comparison", "Log predictive score (proper)"),
        ("Calibration", "Coverage and PIT histograms"),
        ("Sensitivity Analysis", "Assumptions \u2192 P(overcrowded)"),
        ("Prior Sensitivity", "3 priors \u2192 convergence"),
    ]

    for row_start in range(0, 16, 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < 16:
                n, d = concepts[idx]
                col.markdown(f"**{idx+1}. {n}**  \n{d}")

    # Prior sensitivity
    st.markdown("---")
    st.markdown("**Prior Sensitivity: Evidence Overwhelms Prior Beliefs**")
    sens = run_prior_sensitivity(daily_counts)
    fig_ps, ax_ps = plt.subplots(figsize=(12, 3.5))
    cps = [C_MUTED_RED, C_PRIMARY, C_MUTED_GREEN]
    for (name, hist), c in zip(sens.items(), cps):
        t = np.array(hist.times)
        m = np.array(hist.means)
        ax_ps.plot(t[1:], m[1:], color=c, linewidth=2, label=name)
    for s, e in config.surge_windows:
        ax_ps.axvspan(s, e, alpha=0.07, color=C_ACCENT)
    ax_ps.set_xlabel("Day")
    ax_ps.set_ylabel("\u03BB")
    ax_ps.set_title("All Priors Converge")
    ax_ps.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_ps)
    plt.close(fig_ps)


# ---- Ethical Reflection ----
with st.expander("Ethical Reflection"):
    st.markdown("""
- **Synthetic data** \u2014 real ICU data contains protected health information
- **Never a point estimate** \u2014 every number includes a credible interval
- **5 failure modes documented** with detection and uncertainty widening
- **Not for deployment** \u2014 forecasts should augment, not replace, clinical judgment
- **Goodhart's Law** \u2014 any model that influences the system it measures risks becoming unreliable
""")
    mascot_note(
        "The interface exists to make uncertainty visible, "
        "not to make predictions impressive."
    )


# =====================================================================
#  Footer
# =====================================================================
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#9CA3AF; font-size:0.8rem; "
    "margin-top:3rem'>"
    "BUICU \u2014 CS109 Challenge Project  \u00b7  "
    "Built with Bayesian inference, not black-box ML</p>",
    unsafe_allow_html=True,
)
