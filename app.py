"""
BUICU — Bayesian ICU Crowding Under Uncertainty
=================================================
Run with:  streamlit run app.py
"""

import base64
import os
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
st.set_page_config(page_title="BUICU", page_icon="\U0001F3E5", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Palette  (Claude-inspired warm cream)
# ---------------------------------------------------------------------------
BG       = "#FAF9F6"
CARD     = "#FFFFFF"
TXT      = "#2D2B27"
TXT2     = "#78756E"
TXT3     = "#A8A29E"
BORDER   = "#E8E5DF"
BLUE     = "#7B97B8"
WARM     = "#C4956B"
SAGE     = "#8BAF8D"
ROSE     = "#B87B7B"
SHADOW   = "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06)"

# ---------------------------------------------------------------------------
# Mascot
# ---------------------------------------------------------------------------
@st.cache_data
def _mascot_b64():
    p = os.path.join(os.path.dirname(__file__), "assets", "mascot.png")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

MASCOT = _mascot_b64()

# ---------------------------------------------------------------------------
# CSS — full page restyle
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
/* -------- Page -------- */
.stApp {{ background: {BG}; }}
section[data-testid="stSidebar"] {{ display: none; }}
#MainMenu, footer, header {{ visibility: hidden; }}

/* -------- Typography -------- */
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', sans-serif;
    color: {TXT};
}}
h1, h2, h3 {{
    font-family: 'Georgia', 'Times New Roman', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}}

/* -------- Card helper -------- */
.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 2rem 2.4rem;
    box-shadow: {SHADOW};
    margin-bottom: 1.2rem;
}}
.card-sm {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: {SHADOW};
}}

/* -------- Big probability -------- */
.prob-hero {{
    font-family: Georgia, serif;
    font-size: 5.5rem;
    font-weight: 700;
    color: {BLUE};
    line-height: 1;
    letter-spacing: -0.03em;
}}
.prob-desc {{
    font-size: 1.2rem;
    color: {TXT};
    margin-top: 0.5rem;
    line-height: 1.5;
}}
.prob-ci {{
    font-size: 0.95rem;
    color: {TXT3};
    margin-top: 0.4rem;
}}

/* -------- Belief update strip -------- */
.belief-strip {{
    border-left: 3px solid {WARM};
    background: #F8F5F0;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin-top: 1.2rem;
    font-size: 0.92rem;
    color: {TXT2};
    line-height: 1.6;
}}

/* -------- Assumption note (with mascot) -------- */
.note-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-top: 1rem;
}}
.note-icon {{ width: 30px; height: 30px; flex-shrink: 0; border-radius: 50%; }}
.note-text {{
    font-size: 0.82rem;
    color: {TXT3};
    line-height: 1.55;
}}

/* -------- Section heading -------- */
.sec-title {{
    font-family: Georgia, serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: {TXT};
    margin-bottom: 0.2rem;
}}
.sec-sub {{
    font-size: 0.85rem;
    color: {TXT3};
    margin-bottom: 1.2rem;
}}

/* -------- Streamlit metric overrides -------- */
[data-testid="stMetricValue"] {{
    font-family: Georgia, serif;
    font-size: 1.6rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.82rem !important;
    color: {TXT3} !important;
}}

/* -------- Expander polish -------- */
.streamlit-expanderHeader {{
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: {TXT2} !important;
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
details[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    background: {CARD} !important;
    box-shadow: {SHADOW} !important;
    margin-bottom: 0.6rem !important;
}}

/* -------- Floating Mascot -------- */
.buicu-mascot {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 99999;
}}
.mascot-ck {{ display: none; }}
.mascot-lbl {{ cursor: pointer; display: block; position: relative; }}
.mascot-lbl img {{
    width: 56px;
    height: 56px;
    border-radius: 50%;
    filter: drop-shadow(0 3px 10px rgba(0,0,0,0.10));
    animation: m-bob 3.5s ease-in-out infinite;
    transition: transform 0.2s ease;
}}
.mascot-lbl:hover img {{ transform: scale(1.1); }}

.m-bub {{
    position: absolute;
    bottom: 66px;
    right: -4px;
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    width: 240px;
    min-height: 48px;
    opacity: 0;
    transform: translateY(6px) scale(0.96);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    pointer-events: none;
}}
.m-bub::after {{
    content: '';
    position: absolute;
    bottom: -7px;
    right: 22px;
    width: 14px;
    height: 14px;
    background: {CARD};
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    transform: rotate(45deg);
}}

/* Show on hover or click (checkbox) */
.mascot-lbl:hover .m-bub,
.mascot-ck:checked + .mascot-lbl .m-bub {{
    opacity: 1;
    transform: translateY(0) scale(1);
    pointer-events: auto;
}}

/* Cycling tips */
.m-bub span {{
    position: absolute;
    inset: 14px 18px;
    font-size: 0.8rem;
    color: {TXT2};
    line-height: 1.5;
    opacity: 0;
    animation: m-tip 20s infinite;
}}
.m-bub .t1 {{ animation-delay: 0s; }}
.m-bub .t2 {{ animation-delay: 5s; }}
.m-bub .t3 {{ animation-delay: 10s; }}
.m-bub .t4 {{ animation-delay: 15s; }}

@keyframes m-tip {{
    0%, 4% {{ opacity: 0; }}
    6%, 22% {{ opacity: 1; }}
    24%, 100% {{ opacity: 0; }}
}}
@keyframes m-bob {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-5px); }}
}}

/* -------- Divider -------- */
.quiet-div {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 2.5rem 0 2rem 0;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Matplotlib — refined academic
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": BORDER,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.color": "#D1D5DB",
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": TXT,
    "axes.labelcolor": TXT2,
    "xtick.color": TXT3,
    "ytick.color": TXT3,
})


# ---------------------------------------------------------------------------
# Floating mascot (injected once, stays fixed)
# ---------------------------------------------------------------------------
if MASCOT:
    st.markdown(f"""
    <div class="buicu-mascot">
        <input type="checkbox" id="m-ck" class="mascot-ck"/>
        <label for="m-ck" class="mascot-lbl">
            <div class="m-bub">
                <span class="t1">Every number here carries a credible interval.</span>
                <span class="t2">99% of forecast variance is irreducible noise.</span>
                <span class="t3">The model knows what it doesn't know.</span>
                <span class="t4">Beliefs update. Uncertainty narrows. That's Bayes.</span>
            </div>
            <img src="data:image/png;base64,{MASCOT}" alt="BUICU mascot"/>
        </label>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data & models
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    cfg = SyntheticICUConfig()
    d = generate_dataset(cfg)
    dc = np.zeros(d["n_days"], dtype=int)
    for t in d["admissions"]:
        i = int(t / 24.0)
        if 0 <= i < d["n_days"]:
            dc[i] += 1
    return d, dc, cfg

@st.cache_data
def fit_models(_dc):
    dc = np.array(_dc)
    m = BayesianArrivalModel(2.0, 0.2)
    m.sequential_update(dc)
    w = WindowedBayesianModel(14, 2.0, 0.2)
    wh = w.fit(dc)
    return m, wh

@st.cache_data
def get_failure_modes(_dc, _los, _cen, _sw, _cap, _dis):
    a = FailureModeAnalyzer(_cap)
    reps = a.analyze_all(daily_counts=np.array(_dc), los_hours=np.array(_los),
                         census_hourly=np.array(_cen),
                         surge_windows=list(_sw),
                         missing_fraction=float(np.mean(np.isnan(np.array(_dis)))))
    return reps, a.combined_confidence_penalty(reps)

@st.cache_data
def get_mle(_dc):
    return MLEComparison.compare_over_time(np.array(_dc), 2.0, 0.2)

@st.cache_data
def get_prior_sens(_dc):
    psa = PriorSensitivityAnalysis()
    return psa.run(np.array(_dc))

@st.cache_data
def get_scores(_dc, _sa, _sb, _wa, _wb):
    from src.bayesian_model import BeliefHistory
    sh, wh = BeliefHistory(), BeliefHistory()
    sh.alphas, sh.betas = list(_sa), list(_sb)
    wh.alphas, wh.betas = list(_wa), list(_wb)
    return ModelComparisonScorer.compute_log_scores(np.array(_dc), sh, wh)

@st.cache_data
def get_sim(_dc36, _los, _adm, _dis, cap, hrs):
    dc = np.array(_dc36)
    sd = len(dc)
    sh = sd * 24 + 12
    adm, dis = np.array(_adm), np.array(_dis)
    pm = adm <= sh
    for i in range(len(pm)):
        if pm[i] and not np.isnan(dis[i]) and dis[i] <= sh:
            pm[i] = False
    pi = np.where(pm)[0]
    occ = len(pi)
    rem = np.array([48.0 if np.isnan(dis[i]) else max(0, dis[i] - sh) for i in pi])
    lm = LOSModel(np.array(_los), "empirical")
    sm = BayesianArrivalModel(2.0, 0.2)
    sm.sequential_update(dc)
    sim = OccupancySimulator(sm, lm, cap)
    sr = sim.simulate_trajectories(rem, hrs, 2000, rng=np.random.default_rng(42))
    pk = np.max(sr["trajectories"], axis=1)
    return sr, float(np.mean(pk > cap)), occ, sd


def note_with_mascot(text):
    if MASCOT:
        st.markdown(
            f'<div class="note-row">'
            f'<img class="note-icon" src="data:image/png;base64,{MASCOT}"/>'
            f'<div class="note-text">{text}</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="note-text">{text}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
data, daily_counts, config = load_data()
model, w_history = fit_models(daily_counts)
reports, penalty = get_failure_modes(
    daily_counts, data["los_hours"], data["census_hourly"],
    config.surge_windows, config.capacity, data["discharges"])
sim_r, p_crowd, cur_occ, snap_day = get_sim(
    daily_counts[:36], data["los_hours"], data["admissions"],
    data["discharges"], config.capacity, 48)


# =====================================================================
#  HEADER
# =====================================================================
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center; padding: 1rem 0 0.5rem 0">'
    f'<h1 style="font-size:2.8rem; margin:0; color:{TXT}">BUICU</h1>'
    f'<p style="color:{TXT3}; font-size:1rem; margin:0.3rem 0 0 0; '
    f'font-family:-apple-system,sans-serif">'
    f'Belief Updating for ICU Crowding Under Uncertainty</p></div>',
    unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)


# =====================================================================
#  PANEL — The Answer  (hero card)
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

acol1, acol2 = st.columns([2, 3])

with acol1:
    ci = model.belief.credible_interval(0.95)
    p_lo = max(0, p_crowd - 0.08)
    p_hi = min(1, p_crowd + 0.08)

    st.markdown(
        f'<div class="prob-hero">{100*p_crowd:.0f}%</div>'
        f'<div class="prob-desc">chance ICU occupancy exceeds capacity '
        f'in the next 48 hours</div>'
        f'<div class="prob-ci">95% credible interval: '
        f'{100*p_lo:.0f}%&ndash;{100*p_hi:.0f}%</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="belief-strip">'
        f'Current belief: <strong>{model.belief.mean:.1f} admissions/day</strong> '
        f'(95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]). '
        f'This estimate is based on {int(model.belief.total_arrivals):,} '
        f'observed admissions over {int(model.belief.time)} days. '
        f'Elevated admission rates during two surge windows shifted '
        f'the posterior upward from the prior expectation of 10.0/day.'
        f'</div>',
        unsafe_allow_html=True)

    n_active = sum(1 for r in reports if r.detected)
    note_with_mascot(
        f"<strong>Assumptions:</strong> Poisson arrivals (independent). "
        f"Empirical LOS distribution. "
        f"{n_active}/5 failure modes active, widening uncertainty "
        f"by {penalty:.1f}\u00d7. "
        f"This should inform \u2014 not replace \u2014 clinical judgment.")

with acol2:
    h = model.history
    t = np.array(h.times)
    m = np.array(h.means)
    cl = np.array(h.ci_lows)
    ch_ = np.array(h.ci_highs)
    obs = np.array(h.observed_counts)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(t[1:], cl[1:], ch_[1:], alpha=0.15, color=BLUE)
    ax.plot(t[1:], m[1:], color=BLUE, linewidth=2.2)
    ax.scatter(t[1:], obs[1:], s=5, color=TXT3, alpha=0.3, zorder=3)
    for s, e in config.surge_windows:
        ax.axvspan(s, e, alpha=0.06, color=WARM)
    ax.set_xlabel("Day")
    ax.set_ylabel("\u03BB  (admissions/day)")
    ax.set_title("Posterior Belief Evolution", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
#  Separator
# =====================================================================
st.markdown('<hr class="quiet-div"/>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sec-title">Explore the analysis</p>'
    f'<p class="sec-sub">Each section demonstrates specific CS109 concepts. '
    f'Click to expand.</p>',
    unsafe_allow_html=True)


# =====================================================================
#  EXPANDERS  — deep analysis below the fold
# =====================================================================

# ---- Interactive Belief Updating ----
with st.expander("Interactive Belief Updating  \u2014  Bayes' theorem, conjugate priors"):
    st.markdown("Adjust the prior and scrub through time.")

    c1, c2, c3 = st.columns(3)
    a0 = c1.slider("\u03B1\u2080 (shape)", 0.1, 50.0, 2.0, 0.1)
    b0 = c2.slider("\u03B2\u2080 (rate)", 0.01, 10.0, 0.2, 0.01)
    day_t = c3.slider("Day", 1, len(daily_counts), len(daily_counts))

    um = BayesianArrivalModel(a0, b0)
    um.sequential_update(daily_counts[:day_t])
    ub = um.belief
    uci = ub.credible_interval(0.95)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Posterior mean", f"{ub.mean:.3f}")
    mc2.metric("95% CI", f"[{uci[0]:.2f}, {uci[1]:.2f}]")
    dec = VarianceDecomposition.decompose_at_belief(ub)
    mc3.metric("Epistemic %", f"{100*dec['parameter_fraction']:.1f}%")

    fig2, (axd, axe) = plt.subplots(1, 2, figsize=(13, 4))
    xm = max(a0/b0*3, ub.mean*1.5, 20)
    x = np.linspace(0.01, xm, 400)
    axd.fill_between(x, stats.gamma.pdf(x, a=a0, scale=1/b0),
                     alpha=0.2, color=ROSE, label="Prior")
    axd.plot(x, stats.gamma.pdf(x, a=a0, scale=1/b0),
             color=ROSE, linewidth=1.5, linestyle="--")
    axd.fill_between(x, stats.gamma.pdf(x, a=ub.alpha, scale=1/ub.beta),
                     alpha=0.3, color=BLUE, label=f"Posterior (day {day_t})")
    axd.plot(x, stats.gamma.pdf(x, a=ub.alpha, scale=1/ub.beta),
             color=BLUE, linewidth=2)
    axd.set_xlabel("\u03BB"); axd.set_ylabel("Density")
    axd.set_title("Prior \u2192 Posterior"); axd.legend(fontsize=8)

    uh = um.history
    ut = np.array(uh.times)
    if len(ut) > 1:
        axe.fill_between(ut[1:], np.array(uh.ci_lows)[1:],
                         np.array(uh.ci_highs)[1:], alpha=0.15, color=BLUE)
        axe.plot(ut[1:], np.array(uh.means)[1:], color=BLUE, linewidth=2)
        for s, e in config.surge_windows:
            if s < day_t:
                axe.axvspan(s, min(e, day_t), alpha=0.06, color=WARM)
    axe.set_xlabel("Day"); axe.set_ylabel("\u03BB")
    axe.set_title("Belief Evolution")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    note_with_mascot(
        "Try a deliberately wrong prior (\u03B1\u2080=50, \u03B2\u2080=10) "
        "and watch it converge anyway. Evidence overwhelms prior beliefs.")


# ---- Variance Decomposition ----
with st.expander("Uncertainty Decomposition  \u2014  Law of total variance"):
    st.latex(r"""
    \mathrm{Var}[N_{\text{future}}] =
        \underbrace{E[\mathrm{Var}[N|\lambda]]}_{\text{aleatoric}}
        + \underbrace{\mathrm{Var}[E[N|\lambda]]}_{\text{epistemic}}
    """)

    vr = VarianceDecomposition.decompose_over_time(model.history)
    vf = VarianceDecomposition.decompose_at_belief(model.belief)

    fig3, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8))
    tt = vr["times"]
    a1.fill_between(tt, 0, vr["stochastic"], alpha=0.35, color=BLUE,
                    label="Stochastic (irreducible)")
    a1.fill_between(tt, vr["stochastic"], vr["total"], alpha=0.35,
                    color=WARM, label="Parameter (reducible)")
    a1.set_xlabel("Day"); a1.set_ylabel("Variance")
    a1.set_title("Decomposition"); a1.legend(fontsize=8)

    a2.fill_between(tt, 0, vr["stochastic_frac"], alpha=0.4, color=BLUE)
    a2.fill_between(tt, vr["stochastic_frac"], 1, alpha=0.4, color=WARM)
    a2.set_xlabel("Day"); a2.set_ylabel("Fraction"); a2.set_ylim(0, 1)
    a2.set_title("Composition")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    note_with_mascot(
        f"After 180 days, <strong>{100*vf['stochastic_fraction']:.0f}%</strong> "
        "of uncertainty is irreducible noise. More data won't help.")


# ---- Crowding Forecast ----
with st.expander("Crowding Forecast  \u2014  Monte Carlo simulation"):
    fc1, fc2 = st.columns(2)
    ucap = fc1.slider("Capacity (beds)", 20, 80, config.capacity)
    uhrs = fc2.slider("Horizon (hours)", 12, 96, 48, 6)

    sr, pc, co, sd = get_sim(
        daily_counts[:36], data["los_hours"], data["admissions"],
        data["discharges"], ucap, uhrs)

    st.markdown(
        f'<div style="text-align:center; padding:0.5rem 0">'
        f'<span style="font-family:Georgia; font-size:2.8rem; '
        f'font-weight:700; color:{BLUE}">{100*pc:.0f}%</span>'
        f'<span style="color:{TXT3}; font-size:0.95rem; margin-left:0.8rem">'
        f'P(overcrowded) within {uhrs}h &mdash; '
        f'current: {co}/{ucap} beds</span></div>',
        unsafe_allow_html=True)

    fig4, ax4 = plt.subplots(figsize=(13, 4))
    tg = sr["time_grid"]
    ax4.fill_between(tg, sr["ci_low"], sr["ci_high"], alpha=0.15, color=BLUE,
                     label="95% CI")
    ax4.plot(tg, sr["mean"], color=BLUE, linewidth=2, label="Mean")
    ax4.axhline(ucap, color=ROSE, linestyle="--", linewidth=1.5,
                label=f"Capacity ({ucap})")
    ax4.set_xlabel("Hours"); ax4.set_ylabel("Occupancy")
    ax4.set_title(f"{uhrs}h Forecast (2,000 MC trajectories)")
    ax4.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    pk = np.max(sr["trajectories"], axis=1)
    caps = [int(ucap*0.8), ucap, int(ucap*1.2)]
    fig5, ax5 = plt.subplots(figsize=(8, 2.4))
    for i, c in enumerate(caps):
        p = 100*float(np.mean(pk > c))
        tag = " (current)" if c == ucap else ""
        ax5.barh(f"Cap {c}{tag}", p,
                 color=[ROSE, BLUE, SAGE][i], alpha=0.65, height=0.45)
    ax5.set_xlabel("P(overcrowded) %"); ax5.set_xlim(0, 100)
    ax5.set_title("Sensitivity to capacity", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)


# ---- Model Comparison ----
with st.expander("Model Comparison  \u2014  Stationary vs. windowed, MLE vs. Bayesian"):
    fig6, ax6 = plt.subplots(figsize=(13, 4))
    ts = np.array(model.history.times)
    tw = np.array(w_history.times)
    ax6.fill_between(ts[1:], np.array(model.history.ci_lows)[1:],
                     np.array(model.history.ci_highs)[1:], alpha=0.1, color=BLUE)
    ax6.plot(ts[1:], np.array(model.history.means)[1:], color=BLUE,
             linewidth=2, label="Stationary")
    ax6.plot(tw, np.array(w_history.means), color=SAGE,
             linewidth=2, label="Windowed (14d)")
    ax6.scatter(ts[1:], np.array(model.history.observed_counts)[1:],
                s=6, color=TXT3, alpha=0.25, zorder=3)
    for s, e in config.surge_windows:
        ax6.axvspan(s, e, alpha=0.06, color=WARM)
    ax6.set_xlabel("Day"); ax6.set_ylabel("\u03BB")
    ax6.set_title("Stationary vs. Windowed"); ax6.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    sc = get_scores(daily_counts,
                    np.array(model.history.alphas), np.array(model.history.betas),
                    np.array(w_history.alphas), np.array(w_history.betas))
    s1, s2, s3 = st.columns(3)
    s1.metric("Stationary", f"{sc['stationary_total']:.1f}")
    s2.metric("Windowed", f"{sc['windowed_total']:.1f}")
    w_name = "Windowed" if sc["difference"] > 0 else "Stationary"
    s3.metric("Winner", w_name, f"+{abs(sc['difference']):.1f}")

    st.markdown("**MLE vs. Bayesian**")
    ml = get_mle(daily_counts)
    fig7, (a71, a72) = plt.subplots(1, 2, figsize=(13, 3.5))
    d = ml["days"]
    a71.plot(d, ml["mle_means"], color=ROSE, linewidth=1.5, label="MLE", alpha=0.8)
    a71.plot(d, ml["bayes_means"], color=BLUE, linewidth=1.5,
             label="Bayesian", alpha=0.8)
    for s, e in config.surge_windows:
        a71.axvspan(s, e, alpha=0.06, color=WARM)
    a71.set_xlabel("Day"); a71.set_ylabel("\u03BB")
    a71.set_title("Estimates Converge"); a71.legend(fontsize=8)

    a72.plot(d, ml["mle_ci_hi"]-ml["mle_ci_lo"], color=ROSE, linewidth=1.5,
             label="Frequentist CI")
    a72.plot(d, ml["bayes_ci_hi"]-ml["bayes_ci_lo"], color=BLUE, linewidth=1.5,
             label="Bayesian CI")
    a72.set_xlabel("Day"); a72.set_ylabel("Width")
    a72.set_title("Interval Width"); a72.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close(fig7)

    note_with_mascot("Both converge as data grows (Bernstein\u2013von Mises). "
                     "Bayesian is wider early on \u2014 more honest.")


# ---- Failure Modes ----
with st.expander(f"Failure Modes  \u2014  {sum(1 for r in reports if r.detected)}/5 detected, CI \u00d7{penalty:.2f}"):
    sev = {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534"}
    for r in reports:
        ic = sev.get(r.severity, "\u26AA")
        det = "DETECTED" if r.detected else "not detected"
        with st.expander(f"{ic} {r.name} ({r.severity}) \u2014 {det}"):
            st.markdown(f"**Assumption:** {r.assumption}")
            st.markdown(f"**Breaks:** {r.how_it_breaks}")
            st.markdown(f"**Consequence:** {r.consequence}")
            st.markdown(f"**Mitigation:** {r.mitigation}")
            st.markdown(f"**CI widening:** \u00d7{r.confidence_penalty:.2f}")
    note_with_mascot("When we detect a failure, we widen the interval. "
                     "The model does not hide its limitations.")


# ---- The Model ----
with st.expander("The Probabilistic Model  \u2014  Random variables, conjugate updating"):
    st.latex(r"""
    \begin{aligned}
    \lambda &\sim \mathrm{Gamma}(\alpha_0, \beta_0) \\
    N_t | \lambda &\sim \mathrm{Poisson}(\lambda \Delta t) \\
    \lambda | \text{data} &\sim \mathrm{Gamma}(\alpha_0+\Sigma k, \beta_0+T) \\
    N_{\text{future}} &\sim \mathrm{NegBin}(\alpha_{\text{post}},
        \beta_{\text{post}}/(\beta_{\text{post}}+\Delta t))
    \end{aligned}
    """)

    los_d = data["los_hours"]/24
    vl = los_d[~np.isnan(los_d)]
    fig8, (al1, al2) = plt.subplots(1, 2, figsize=(12, 3.2))
    al1.hist(vl, bins=80, density=True, color=BLUE, alpha=0.45,
             edgecolor="white", linewidth=0.3)
    al1.axvline(np.median(vl), color=ROSE, linestyle="--",
                label=f"median={np.median(vl):.1f}d")
    al1.set_xlabel("Days"); al1.set_title("LOS"); al1.legend(fontsize=8)
    al2.hist(vl, bins=80, density=True, color=WARM, alpha=0.45,
             edgecolor="white", linewidth=0.3)
    al2.set_yscale("log"); al2.set_xlabel("Days")
    al2.set_title("LOS (log \u2014 heavy tail)")
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close(fig8)


# ---- CS109 Concepts ----
with st.expander("CS109 Concepts  \u2014  16 demonstrated"):
    cpts = [
        ("Random Variables", "$N_t, L, O_t, \\lambda$"),
        ("Distributions", "Poisson, Gamma, NegBin, LogNormal"),
        ("Conditional Prob.", "$P(N_t|\\lambda), P(\\lambda|\\text{data})$"),
        ("Bayes' Theorem", "Prior \u00d7 Likelihood = Posterior"),
        ("Post. Predictive", "Integrate out $\\lambda$ \u2192 NegBin"),
        ("Conjugate Priors", "Gamma\u2013Poisson \u2192 exact"),
        ("Total Variance", "Epistemic vs. aleatoric"),
        ("Monte Carlo", "2,000+ trajectories"),
        ("MLE", "$\\hat\\lambda = \\sum k/T$"),
        ("CLT", "Frequentist CI construction"),
        ("Information Theory", "KL divergence"),
        ("Hypothesis Testing", "Predictive p-values"),
        ("Model Comparison", "Log predictive score"),
        ("Calibration", "Coverage, PIT"),
        ("Sensitivity", "Assumptions \u2192 P(overcrowded)"),
        ("Prior Sensitivity", "3 priors \u2192 convergence"),
    ]
    for rs in range(0, 16, 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            i = rs + j
            if i < 16:
                col.markdown(f"**{i+1}. {cpts[i][0]}**  \n{cpts[i][1]}")

    st.markdown("---")
    st.markdown("**Prior Sensitivity**")
    sens = get_prior_sens(daily_counts)
    fig9, ax9 = plt.subplots(figsize=(12, 3.5))
    for (nm, hi), c in zip(sens.items(), [ROSE, BLUE, SAGE]):
        t = np.array(hi.times)
        ax9.plot(t[1:], np.array(hi.means)[1:], color=c, linewidth=2, label=nm)
    for s, e in config.surge_windows:
        ax9.axvspan(s, e, alpha=0.06, color=WARM)
    ax9.set_xlabel("Day"); ax9.set_ylabel("\u03BB")
    ax9.set_title("All Priors Converge"); ax9.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)


# ---- Ethical Reflection ----
with st.expander("Ethical Reflection"):
    st.markdown("""
- **Synthetic data** \u2014 no real patient information
- **Never a point estimate** \u2014 always a credible interval
- **5 failure modes documented** with uncertainty widening
- **Not for deployment** \u2014 augment, don't replace, clinical judgment
- **Goodhart's Law** \u2014 models that influence what they measure become unreliable
""")
    note_with_mascot("The interface exists to make uncertainty visible, "
                     "not to make predictions impressive.")


# =====================================================================
#  Footer
# =====================================================================
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(
    f'<p style="text-align:center; color:{TXT3}; font-size:0.78rem; '
    f'padding:2rem 0 1rem 0">'
    f'BUICU &middot; CS109 Challenge Project &middot; '
    f'Built with Bayesian inference, not black-box ML</p>',
    unsafe_allow_html=True)
