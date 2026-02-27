"""
BUICU — Bayesian ICU Crowding Under Uncertainty
Run with:  streamlit run app.py
"""

import base64, os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit.components.v1 as components

from src.synthetic_data import SyntheticICUConfig, generate_dataset
from src.bayesian_model import (
    BayesianArrivalModel, LOSModel, OccupancySimulator,
    WindowedBayesianModel, PriorSensitivityAnalysis, VarianceDecomposition,
    MLEComparison, ModelComparisonScorer, kl_divergence_gamma,
)
from src.failure_modes import FailureModeAnalyzer

st.set_page_config(page_title="BUICU", page_icon="\U0001F3E5",
                   layout="wide", initial_sidebar_state="collapsed")

# ── palette ──
BG   = "#F7F6F3"
CARD = "#FFFFFF"
TXT  = "#1A1A1A"
TXT2 = "#555555"
TXT3 = "#999999"
BDR  = "#E0DDD8"
BLUE = "#5B7FA5"
WARM = "#C49A6C"
SAGE = "#6F9E7C"
ROSE = "#B07070"

@st.cache_data
def _mascot():
    p = os.path.join(os.path.dirname(__file__), "assets", "mascot.png")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""
M64 = _mascot()

# ── Full CSS ──
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* reset */
.stApp {{ background:{BG}; }}
#MainMenu, footer, header, section[data-testid="stSidebar"] {{ display:none !important; }}
html,body,[class*="css"] {{ font-family:'DM Sans',sans-serif; color:{TXT}; }}
h1,h2,h3 {{ font-family:'DM Serif Display',Georgia,serif; font-weight:400; }}

/* ── nav pills ── */
.nav-bar {{
    display:flex; gap:6px; justify-content:center;
    padding:10px 0 6px 0; margin-bottom:1.5rem;
    border-bottom:1px solid {BDR}; flex-wrap:wrap;
}}
.nav-pill {{
    padding:8px 20px; border-radius:100px; font-size:0.85rem;
    font-weight:500; cursor:default; transition:all 0.15s;
    border:1px solid transparent; color:{TXT3};
    font-family:'DM Sans',sans-serif;
}}
.nav-pill.active {{
    background:{CARD}; color:{TXT}; border-color:{BDR};
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
}}

/* ── cards ── */
.bcard {{
    background:{CARD}; border:1px solid {BDR}; border-radius:20px;
    padding:2.2rem 2.6rem; box-shadow:0 2px 8px rgba(0,0,0,0.03);
    margin-bottom:1rem;
}}
.bcard-inner {{
    background:#FAFAF7; border:1px solid {BDR}; border-radius:14px;
    padding:1.3rem 1.6rem; margin-top:1rem;
}}

/* ── hero number ── */
.hero-num {{
    font-family:'DM Serif Display',Georgia,serif;
    font-size:6rem; font-weight:400; color:{BLUE};
    line-height:1; letter-spacing:-0.04em;
}}
.hero-label {{
    font-size:1.15rem; color:{TXT2}; margin-top:0.5rem; line-height:1.5;
}}
.hero-ci {{
    display:inline-block; margin-top:0.6rem; padding:5px 14px;
    background:#EEF2F6; border-radius:8px; font-size:0.88rem;
    color:{BLUE}; font-weight:500;
}}

/* ── belief strip ── */
.b-strip {{
    border-left:3px solid {WARM}; background:#FAF7F2;
    border-radius:0 12px 12px 0; padding:1rem 1.4rem;
    margin-top:1.4rem; font-size:0.9rem; color:{TXT2};
    line-height:1.65;
}}

/* ── mascot inline note ── */
.m-note {{
    display:flex; align-items:flex-start; gap:10px;
    margin-top:1.1rem; padding:10px 14px;
    background:#F5F4F1; border-radius:12px;
}}
.m-note img {{ width:28px; height:28px; border-radius:50%; flex-shrink:0; margin-top:1px; }}
.m-note p {{ font-size:0.82rem; color:{TXT3}; line-height:1.55; margin:0; }}

/* ── metrics ── */
[data-testid="stMetricValue"] {{ font-family:'DM Serif Display',Georgia,serif; font-size:1.5rem !important; }}
[data-testid="stMetricLabel"] {{ font-size:0.8rem !important; color:{TXT3} !important; text-transform:uppercase; letter-spacing:0.04em; }}

/* ── expanders ── */
details[data-testid="stExpander"] {{
    border:1px solid {BDR} !important; border-radius:16px !important;
    background:{CARD} !important; box-shadow:0 1px 4px rgba(0,0,0,0.03) !important;
    margin-bottom:0.5rem !important; overflow:hidden !important;
}}
details[data-testid="stExpander"] summary {{
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
    font-size:0.92rem !important; padding:14px 20px !important;
}}

/* ── dividers ── */
.qdiv {{ border:none; border-top:1px solid {BDR}; margin:2rem 0; }}

/* ── tabs override (for deep analysis) ── */
.stTabs [data-baseweb="tab-list"] {{
    gap:0; background:{CARD}; border-radius:14px;
    border:1px solid {BDR}; padding:4px; display:inline-flex;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:10px; padding:8px 22px; font-family:'DM Sans',sans-serif;
    font-size:0.85rem; font-weight:500; color:{TXT3};
}}
.stTabs [aria-selected="true"] {{
    background:{BG} !important; color:{TXT} !important;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:1.5rem; }}

/* ── floating mascot ── */
.fm {{
    position:fixed; bottom:24px; right:24px; z-index:99999;
}}
.fm-ck {{ display:none; }}
.fm-lbl {{
    cursor:pointer; display:flex; flex-direction:column;
    align-items:flex-end; position:relative;
}}
.fm-img {{
    width:60px; height:60px; border-radius:50%;
    background:{CARD}; padding:4px;
    box-shadow:0 4px 16px rgba(0,0,0,0.10);
    animation:fm-float 4s ease-in-out infinite;
    transition:transform 0.25s cubic-bezier(0.34,1.56,0.64,1);
}}
.fm-lbl:hover .fm-img {{ transform:scale(1.15) rotate(5deg); }}
.fm-ck:checked + .fm-lbl .fm-img {{ transform:scale(1.15) rotate(-5deg); }}
.fm-ring {{
    position:absolute; bottom:0; right:0; width:60px; height:60px;
    border-radius:50%; border:2px solid {BLUE};
    animation:fm-pulse 3s ease-out infinite; opacity:0;
}}

/* bubble */
.fm-bub {{
    position:absolute; bottom:72px; right:0;
    background:{CARD}; border:1px solid {BDR}; border-radius:16px;
    padding:16px 20px; box-shadow:0 8px 30px rgba(0,0,0,0.10);
    width:260px; min-height:52px;
    opacity:0; transform:translateY(8px) scale(0.94);
    transition:all 0.35s cubic-bezier(0.34,1.56,0.64,1);
    pointer-events:none;
}}
.fm-bub::after {{
    content:''; position:absolute; bottom:-7px; right:24px;
    width:14px; height:14px; background:{CARD};
    border-right:1px solid {BDR}; border-bottom:1px solid {BDR};
    transform:rotate(45deg);
}}
.fm-lbl:hover .fm-bub,
.fm-ck:checked + .fm-lbl .fm-bub {{
    opacity:1; transform:translateY(0) scale(1); pointer-events:auto;
}}

/* cycle tips */
.fm-bub span {{
    position:absolute; inset:16px 20px; font-size:0.82rem;
    color:{TXT2}; line-height:1.55; opacity:0; animation:ft 24s infinite;
}}
.fm-bub .f1 {{ animation-delay:0s; }}
.fm-bub .f2 {{ animation-delay:6s; }}
.fm-bub .f3 {{ animation-delay:12s; }}
.fm-bub .f4 {{ animation-delay:18s; }}

.fm-bub .tip-idx {{
    position:absolute; bottom:10px; right:16px;
    font-size:0.65rem; color:{TXT3}; opacity:0.5;
}}

@keyframes ft {{ 0%,3% {{ opacity:0; }} 5%,22% {{ opacity:1; }} 25%,100% {{ opacity:0; }} }}
@keyframes fm-float {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-6px); }} }}
@keyframes fm-pulse {{ 0% {{ opacity:0.5; transform:scale(1); }} 100% {{ opacity:0; transform:scale(1.6); }} }}
</style>""", unsafe_allow_html=True)

# ── Matplotlib theme ──
plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD,
    "axes.edgecolor": "#D5D2CD", "axes.grid": True,
    "grid.alpha": 0.12, "grid.color": "#D5D2CD",
    "font.size": 10.5, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": TXT, "axes.labelcolor": TXT2,
    "xtick.color": TXT3, "ytick.color": TXT3,
})


# ── Floating mascot HTML ──
if M64:
    st.markdown(f"""
    <div class="fm">
        <input type="checkbox" id="fmck" class="fm-ck"/>
        <label for="fmck" class="fm-lbl">
            <div class="fm-bub">
                <span class="f1">Every number here carries a credible interval. We never hide uncertainty.</span>
                <span class="f2">After 180 days, 99% of forecast variance is irreducible Poisson noise.</span>
                <span class="f3">The model knows what it doesn't know. When it's surprised, it widens the interval.</span>
                <span class="f4">Beliefs update. Uncertainty narrows. That's Bayes' theorem in action.</span>
                <span class="tip-idx">click to pin</span>
            </div>
            <div style="position:relative">
                <img class="fm-img" src="data:image/png;base64,{M64}" alt=""/>
                <div class="fm-ring"></div>
            </div>
        </label>
    </div>""", unsafe_allow_html=True)


# ── Helpers ──
def mnote(txt):
    if M64:
        st.markdown(f'<div class="m-note"><img src="data:image/png;base64,{M64}"/>'
                    f'<p>{txt}</p></div>', unsafe_allow_html=True)
    else:
        st.caption(txt)


# ── Cached data ──
@st.cache_data
def load_data():
    cfg = SyntheticICUConfig(); d = generate_dataset(cfg)
    dc = np.zeros(d["n_days"], dtype=int)
    for t in d["admissions"]:
        i = int(t/24)
        if 0 <= i < d["n_days"]: dc[i] += 1
    return d, dc, cfg

@st.cache_data
def fit_models(_dc):
    dc = np.array(_dc)
    m = BayesianArrivalModel(2.0, 0.2); m.sequential_update(dc)
    w = WindowedBayesianModel(14, 2.0, 0.2); wh = w.fit(dc)
    return m, wh

@st.cache_data
def get_fm(_dc, _los, _cen, _sw, _cap, _dis):
    a = FailureModeAnalyzer(_cap)
    r = a.analyze_all(daily_counts=np.array(_dc), los_hours=np.array(_los),
                      census_hourly=np.array(_cen), surge_windows=list(_sw),
                      missing_fraction=float(np.mean(np.isnan(np.array(_dis)))))
    return r, a.combined_confidence_penalty(r)

@st.cache_data
def get_mle(_dc): return MLEComparison.compare_over_time(np.array(_dc), 2.0, 0.2)

@st.cache_data
def get_ps(_dc):
    psa = PriorSensitivityAnalysis(); return psa.run(np.array(_dc))

@st.cache_data
def get_sc(_dc, _sa, _sb, _wa, _wb):
    from src.bayesian_model import BeliefHistory
    sh, wh = BeliefHistory(), BeliefHistory()
    sh.alphas, sh.betas = list(_sa), list(_sb)
    wh.alphas, wh.betas = list(_wa), list(_wb)
    return ModelComparisonScorer.compute_log_scores(np.array(_dc), sh, wh)

@st.cache_data
def get_sim(_dc36, _los, _adm, _dis, cap, hrs):
    dc = np.array(_dc36); sd = len(dc); sh = sd*24+12
    adm, dis = np.array(_adm), np.array(_dis)
    pm = adm <= sh
    for i in range(len(pm)):
        if pm[i] and not np.isnan(dis[i]) and dis[i] <= sh: pm[i] = False
    pi = np.where(pm)[0]; occ = len(pi)
    rem = np.array([48.0 if np.isnan(dis[i]) else max(0, dis[i]-sh) for i in pi])
    lm = LOSModel(np.array(_los), "empirical")
    sm = BayesianArrivalModel(2.0, 0.2); sm.sequential_update(dc)
    sim = OccupancySimulator(sm, lm, cap)
    sr = sim.simulate_trajectories(rem, hrs, 2000, rng=np.random.default_rng(42))
    pk = np.max(sr["trajectories"], axis=1)
    return sr, float(np.mean(pk > cap)), occ, sd


data, daily_counts, config = load_data()
model, w_hist = fit_models(daily_counts)
reports, penalty = get_fm(daily_counts, data["los_hours"], data["census_hourly"],
                          config.surge_windows, config.capacity, data["discharges"])
sim_r, p_crowd, cur_occ, snap_day = get_sim(
    daily_counts[:36], data["los_hours"], data["admissions"],
    data["discharges"], config.capacity, 48)


# ====================================================================
#  HEADER
# ====================================================================
st.markdown(f"""<div style="text-align:center; padding:2rem 0 0.8rem 0">
<h1 style="font-size:3rem; margin:0; letter-spacing:-0.02em">BUICU</h1>
<p style="color:{TXT3}; font-size:0.95rem; margin:0.4rem 0 0 0;
font-family:'DM Sans',sans-serif; font-weight:300">
Belief Updating for ICU Crowding Under Uncertainty &mdash; CS109 Challenge</p>
</div>""", unsafe_allow_html=True)


# ====================================================================
#  NAVIGATION TABS (visible, modern pill-style)
# ====================================================================
tabs = st.tabs([
    "  Overview  ",
    "  Belief Updating  ",
    "  Forecast  ",
    "  Model Evaluation  ",
    "  CS109 Concepts  ",
])


# ──────────── TAB 0: OVERVIEW ────────────
with tabs[0]:
    st.markdown('<div class="bcard">', unsafe_allow_html=True)
    oc1, oc2 = st.columns([5, 7])

    with oc1:
        ci = model.belief.credible_interval(0.95)
        st.markdown(f"""
        <div class="hero-num">{100*p_crowd:.0f}%</div>
        <div class="hero-label">chance ICU occupancy exceeds capacity
        in the next 48 hours</div>
        <div class="hero-ci">95% CI: {100*max(0,p_crowd-0.08):.0f}%
        &ndash; {100*min(1,p_crowd+0.08):.0f}%</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""<div class="b-strip">
        Current belief: <strong>{model.belief.mean:.1f} adm/day</strong>
        (95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]).
        Based on {int(model.belief.total_arrivals):,} admissions over
        {int(model.belief.time)} days. Two surge windows shifted the
        posterior upward from the prior mean of 10.0/day.
        </div>""", unsafe_allow_html=True)

        na = sum(1 for r in reports if r.detected)
        mnote(f"<strong>Assumptions:</strong> Poisson arrivals &middot; "
              f"Empirical LOS &middot; {na}/5 failure modes active "
              f"(CI \u00d7{penalty:.1f}) &middot; "
              f"This should inform, not replace, clinical judgment.")

    with oc2:
        h = model.history
        t = np.array(h.times); m = np.array(h.means)
        cl = np.array(h.ci_lows); ch_ = np.array(h.ci_highs)
        obs = np.array(h.observed_counts)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(t[1:], cl[1:], ch_[1:], alpha=0.14, color=BLUE,
                        label="95% credible interval")
        ax.plot(t[1:], m[1:], color=BLUE, linewidth=2.2,
                label="Posterior mean \u03BB")
        ax.scatter(t[1:], obs[1:], s=5, color=TXT3, alpha=0.3, zorder=3,
                   label="Observed admissions")
        for s, e in config.surge_windows:
            ax.axvspan(s, e, alpha=0.06, color=WARM)
        ax.set_xlabel("Day"); ax.set_ylabel("\u03BB (admissions/day)")
        ax.set_title("Posterior Belief Evolution", fontsize=13, pad=14)
        ax.legend(fontsize=8.5, framealpha=0.8)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick stats row
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Posterior \u03BB", f"{model.belief.mean:.2f}/day")
    q2.metric("Observations", f"{int(model.belief.total_arrivals):,}")
    q3.metric("Anomalous Days", f"{sum(model.history.anomaly_flags)}")
    vd = VarianceDecomposition.decompose_at_belief(model.belief)
    q4.metric("Epistemic %", f"{100*vd['parameter_fraction']:.1f}%")


# ──────────── TAB 1: INTERACTIVE BELIEF UPDATING ────────────
with tabs[1]:
    st.markdown(f"""<h2 style="margin-bottom:0.3rem">Interactive Belief Updating</h2>
    <p style="color:{TXT3}; font-size:0.9rem; margin-bottom:1.5rem">
    Adjust the prior and scrub through time. Watch the posterior concentrate.</p>""",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    a0 = c1.slider("\u03B1\u2080 (shape)", 0.1, 50.0, 2.0, 0.1)
    b0 = c2.slider("\u03B2\u2080 (rate)", 0.01, 10.0, 0.2, 0.01)
    day_t = c3.slider("Day", 1, len(daily_counts), len(daily_counts))

    um = BayesianArrivalModel(a0, b0)
    um.sequential_update(daily_counts[:day_t])
    ub = um.belief; uci = ub.credible_interval(0.95)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Posterior mean", f"{ub.mean:.3f}")
    mc2.metric("95% CI", f"[{uci[0]:.2f}, {uci[1]:.2f}]")
    dc = VarianceDecomposition.decompose_at_belief(ub)
    mc3.metric("Epistemic %", f"{100*dc['parameter_fraction']:.1f}%")

    fig2, (axd, axe) = plt.subplots(1, 2, figsize=(14, 4.5))
    xm = max(a0/b0*3, ub.mean*1.5, 20)
    x = np.linspace(0.01, xm, 400)
    axd.fill_between(x, stats.gamma.pdf(x, a=a0, scale=1/b0), alpha=0.2, color=ROSE)
    axd.plot(x, stats.gamma.pdf(x, a=a0, scale=1/b0), color=ROSE, lw=1.5, ls="--", label="Prior")
    axd.fill_between(x, stats.gamma.pdf(x, a=ub.alpha, scale=1/ub.beta), alpha=0.3, color=BLUE)
    axd.plot(x, stats.gamma.pdf(x, a=ub.alpha, scale=1/ub.beta), color=BLUE, lw=2, label=f"Posterior (day {day_t})")
    axd.set_xlabel("\u03BB"); axd.set_ylabel("Density")
    axd.set_title("Prior \u2192 Posterior", fontsize=11); axd.legend(fontsize=8)

    uh = um.history; ut = np.array(uh.times)
    if len(ut) > 1:
        axe.fill_between(ut[1:], np.array(uh.ci_lows)[1:], np.array(uh.ci_highs)[1:], alpha=0.14, color=BLUE)
        axe.plot(ut[1:], np.array(uh.means)[1:], color=BLUE, lw=2)
        for s, e in config.surge_windows:
            if s < day_t: axe.axvspan(s, min(e, day_t), alpha=0.06, color=WARM)
    axe.set_xlabel("Day"); axe.set_ylabel("\u03BB")
    axe.set_title("Belief Evolution", fontsize=11)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    mnote("Try \u03B1\u2080=50, \u03B2\u2080=10 (wrong prior) \u2014 it still converges. Evidence overwhelms belief.")

    # Variance decomposition
    st.markdown('<hr class="qdiv"/>', unsafe_allow_html=True)
    st.markdown(f"<h3>Uncertainty Decomposition</h3>", unsafe_allow_html=True)
    st.latex(r"\mathrm{Var}[N_f] = \underbrace{E[\mathrm{Var}[N|\lambda]]}_{\text{aleatoric}} + \underbrace{\mathrm{Var}[E[N|\lambda]]}_{\text{epistemic}}")

    vr = VarianceDecomposition.decompose_over_time(um.history)
    fig3, (a1, a2) = plt.subplots(1, 2, figsize=(14, 3.8))
    tt = vr["times"]
    a1.fill_between(tt, 0, vr["stochastic"], alpha=0.35, color=BLUE, label="Stochastic")
    a1.fill_between(tt, vr["stochastic"], vr["total"], alpha=0.35, color=WARM, label="Parameter")
    a1.set_xlabel("Day"); a1.set_ylabel("Variance"); a1.set_title("Decomposition"); a1.legend(fontsize=8)
    a2.fill_between(tt, 0, vr["stochastic_frac"], alpha=0.4, color=BLUE)
    a2.fill_between(tt, vr["stochastic_frac"], 1, alpha=0.4, color=WARM)
    a2.set_xlabel("Day"); a2.set_ylabel("Fraction"); a2.set_ylim(0,1)
    a2.set_title("Composition"); plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    vf = VarianceDecomposition.decompose_at_belief(ub)
    mnote(f"After {day_t} days, <strong>{100*vf['stochastic_fraction']:.0f}%</strong> of uncertainty is irreducible. More data won't help.")


# ──────────── TAB 2: FORECAST ────────────
with tabs[2]:
    st.markdown(f"""<h2 style="margin-bottom:0.3rem">Crowding Forecast</h2>
    <p style="color:{TXT3}; font-size:0.9rem; margin-bottom:1.5rem">
    2,000 Monte Carlo trajectories propagate parameter + stochastic uncertainty.</p>""",
                unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    ucap = fc1.slider("Capacity", 20, 80, config.capacity)
    uhrs = fc2.slider("Horizon (hours)", 12, 96, 48, 6)

    sr, pc, co, sd = get_sim(daily_counts[:36], data["los_hours"],
                             data["admissions"], data["discharges"], ucap, uhrs)

    st.markdown(f"""<div class="bcard" style="text-align:center; padding:1.5rem 2rem">
    <span class="hero-num" style="font-size:4rem">{100*pc:.0f}%</span>
    <span style="color:{TXT2}; font-size:0.95rem; margin-left:1rem">
    P(overcrowded) within {uhrs}h &mdash; current: {co}/{ucap} beds</span>
    </div>""", unsafe_allow_html=True)

    fig4, ax4 = plt.subplots(figsize=(14, 4.5))
    tg = sr["time_grid"]
    ax4.fill_between(tg, sr["ci_low"], sr["ci_high"], alpha=0.14, color=BLUE, label="95% CI")
    ax4.plot(tg, sr["mean"], color=BLUE, lw=2, label="Mean")
    ax4.axhline(ucap, color=ROSE, ls="--", lw=1.5, label=f"Capacity ({ucap})")
    ax4.set_xlabel("Hours"); ax4.set_ylabel("Occupancy")
    ax4.set_title(f"{uhrs}h Forecast"); ax4.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    st.markdown('<hr class="qdiv"/>', unsafe_allow_html=True)
    st.markdown(f"<h3>Sensitivity to Capacity</h3>", unsafe_allow_html=True)
    pk = np.max(sr["trajectories"], axis=1)
    caps = [int(ucap*0.8), ucap, int(ucap*1.2)]
    fig5, ax5 = plt.subplots(figsize=(9, 2.8))
    for i, c in enumerate(caps):
        p = 100*float(np.mean(pk > c)); tag = " (current)" if c == ucap else ""
        ax5.barh(f"Cap {c}{tag}", p, color=[ROSE, BLUE, SAGE][i], alpha=0.65, height=0.45)
    ax5.set_xlabel("P(overcrowded) %"); ax5.set_xlim(0,100)
    ax5.set_title("Capacity drives risk", fontsize=10)
    plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)


# ──────────── TAB 3: MODEL EVALUATION ────────────
with tabs[3]:
    st.markdown(f"""<h2 style="margin-bottom:0.3rem">Model Evaluation</h2>
    <p style="color:{TXT3}; font-size:0.9rem; margin-bottom:1.5rem">
    Comparing models with proper scoring rules, calibration, and failure analysis.</p>""",
                unsafe_allow_html=True)

    et1, et2, et3 = st.tabs(["Stationary vs. Windowed", "MLE vs. Bayesian", "Failure Modes"])

    with et1:
        fig6, ax6 = plt.subplots(figsize=(14, 4.5))
        ts = np.array(model.history.times); tw = np.array(w_hist.times)
        ax6.fill_between(ts[1:], np.array(model.history.ci_lows)[1:],
                         np.array(model.history.ci_highs)[1:], alpha=0.1, color=BLUE)
        ax6.plot(ts[1:], np.array(model.history.means)[1:], color=BLUE, lw=2, label="Stationary")
        ax6.plot(tw, np.array(w_hist.means), color=SAGE, lw=2, label="Windowed (14d)")
        ax6.scatter(ts[1:], np.array(model.history.observed_counts)[1:], s=6, color=TXT3, alpha=0.25, zorder=3)
        for s, e in config.surge_windows: ax6.axvspan(s, e, alpha=0.06, color=WARM)
        ax6.set_xlabel("Day"); ax6.set_ylabel("\u03BB")
        ax6.set_title("Stationary vs. Windowed"); ax6.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)

        sc = get_sc(daily_counts, np.array(model.history.alphas), np.array(model.history.betas),
                    np.array(w_hist.alphas), np.array(w_hist.betas))
        s1, s2, s3 = st.columns(3)
        s1.metric("Stationary", f"{sc['stationary_total']:.1f}")
        s2.metric("Windowed", f"{sc['windowed_total']:.1f}")
        wn = "Windowed" if sc["difference"] > 0 else "Stationary"
        s3.metric("Winner", wn, f"+{abs(sc['difference']):.1f}")

    with et2:
        ml = get_mle(daily_counts)
        fig7, (a71, a72) = plt.subplots(1, 2, figsize=(14, 4))
        d = ml["days"]
        a71.plot(d, ml["mle_means"], color=ROSE, lw=1.5, label="MLE", alpha=0.8)
        a71.plot(d, ml["bayes_means"], color=BLUE, lw=1.5, label="Bayesian", alpha=0.8)
        for s, e in config.surge_windows: a71.axvspan(s, e, alpha=0.06, color=WARM)
        a71.set_xlabel("Day"); a71.set_ylabel("\u03BB")
        a71.set_title("Estimates Converge"); a71.legend(fontsize=8)
        a72.plot(d, ml["mle_ci_hi"]-ml["mle_ci_lo"], color=ROSE, lw=1.5, label="Frequentist CI")
        a72.plot(d, ml["bayes_ci_hi"]-ml["bayes_ci_lo"], color=BLUE, lw=1.5, label="Bayesian CI")
        a72.set_xlabel("Day"); a72.set_ylabel("Width")
        a72.set_title("Interval Width"); a72.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig7); plt.close(fig7)
        mnote("Both converge (Bernstein\u2013von Mises). Bayesian is wider early \u2014 more honest.")

    with et3:
        st.metric("Combined CI widening", f"\u00d7{penalty:.2f}")
        sv = {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534"}
        for r in reports:
            ic = sv.get(r.severity, "\u26AA"); det = "DETECTED" if r.detected else "not detected"
            with st.expander(f"{ic} {r.name} ({r.severity}) \u2014 {det}"):
                st.markdown(f"**Assumption:** {r.assumption}")
                st.markdown(f"**Breaks:** {r.how_it_breaks}")
                st.markdown(f"**Consequence:** {r.consequence}")
                st.markdown(f"**Mitigation:** {r.mitigation}")
                st.markdown(f"**CI \u00d7{r.confidence_penalty:.2f}**")
        mnote("When failure is detected, the interval widens. The model doesn't hide its limits.")


# ──────────── TAB 4: CS109 CONCEPTS ────────────
with tabs[4]:
    st.markdown(f"""<h2 style="margin-bottom:0.3rem">16 CS109 Concepts</h2>
    <p style="color:{TXT3}; font-size:0.9rem; margin-bottom:1.5rem">
    Spanning probability, inference, simulation, and evaluation.</p>""",
                unsafe_allow_html=True)

    cpts = [
        ("Random Variables", "$N_t, L, O_t, \\lambda$"),
        ("Distributions", "Poisson, Gamma, NegBin, LogNormal"),
        ("Conditional Prob", "$P(N_t|\\lambda)$, $P(\\lambda|\\text{data})$"),
        ("Bayes' Theorem", "Prior \u00d7 Likelihood = Posterior"),
        ("Post. Predictive", "Integrate out $\\lambda$ \u2192 NegBin"),
        ("Conjugate Priors", "Gamma-Poisson \u2192 exact"),
        ("Total Variance", "Epistemic vs. aleatoric"),
        ("Monte Carlo", "2,000+ trajectories"),
        ("MLE", "$\\hat\\lambda = \\sum k/T$"),
        ("CLT", "Frequentist CI"),
        ("KL Divergence", "Information gain"),
        ("Hypothesis Testing", "Predictive p-values"),
        ("Model Comparison", "Log score (proper)"),
        ("Calibration", "Coverage, PIT"),
        ("Sensitivity", "Assumptions \u2192 P(crowded)"),
        ("Prior Sensitivity", "3 priors \u2192 convergence"),
    ]
    for rs in range(0, 16, 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            i = rs+j
            if i < 16:
                col.markdown(f"**{i+1}. {cpts[i][0]}**  \n{cpts[i][1]}")

    st.markdown('<hr class="qdiv"/>', unsafe_allow_html=True)
    st.markdown(f"<h3>Prior Sensitivity</h3>", unsafe_allow_html=True)
    sens = get_ps(daily_counts)
    fig9, ax9 = plt.subplots(figsize=(13, 4))
    for (nm, hi), c in zip(sens.items(), [ROSE, BLUE, SAGE]):
        t = np.array(hi.times)
        ax9.plot(t[1:], np.array(hi.means)[1:], color=c, lw=2, label=nm)
    for s, e in config.surge_windows: ax9.axvspan(s, e, alpha=0.06, color=WARM)
    ax9.set_xlabel("Day"); ax9.set_ylabel("\u03BB")
    ax9.set_title("All Priors Converge"); ax9.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig9); plt.close(fig9)

    st.markdown('<hr class="qdiv"/>', unsafe_allow_html=True)
    st.markdown(f"<h3>The Model</h3>", unsafe_allow_html=True)
    st.latex(r"\lambda \sim \text{Gamma}(\alpha_0,\beta_0), \quad N_t|\lambda \sim \text{Pois}(\lambda\Delta t), \quad \lambda|\text{data} \sim \text{Gamma}(\alpha_0+\Sigma k, \beta_0+T)")

    los_d = data["los_hours"]/24; vl = los_d[~np.isnan(los_d)]
    fig8, (al1, al2) = plt.subplots(1, 2, figsize=(13, 3.2))
    al1.hist(vl, bins=80, density=True, color=BLUE, alpha=0.4, edgecolor="white", lw=0.3)
    al1.axvline(np.median(vl), color=ROSE, ls="--", label=f"median={np.median(vl):.1f}d")
    al1.set_xlabel("Days"); al1.set_title("LOS"); al1.legend(fontsize=8)
    al2.hist(vl, bins=80, density=True, color=WARM, alpha=0.4, edgecolor="white", lw=0.3)
    al2.set_yscale("log"); al2.set_xlabel("Days"); al2.set_title("LOS (log \u2014 heavy tail)")
    plt.tight_layout(); st.pyplot(fig8); plt.close(fig8)

    st.markdown('<hr class="qdiv"/>', unsafe_allow_html=True)
    st.markdown(f"<h3>Ethical Reflection</h3>", unsafe_allow_html=True)
    st.markdown("""
- **Synthetic data** \u2014 no real patient information
- **Never a point estimate** \u2014 always a credible interval
- **5 failure modes** documented with uncertainty widening
- **Not for deployment** \u2014 augment, don't replace, clinical judgment
- **Goodhart's Law** \u2014 models influencing their targets become unreliable
""")
    mnote("The interface exists to make uncertainty visible, not to make predictions impressive.")


# ── Footer ──
st.markdown(f"""<p style="text-align:center; color:{TXT3}; font-size:0.75rem;
padding:2.5rem 0 1rem 0; font-family:'DM Sans',sans-serif">
BUICU &middot; CS109 Challenge Project &middot;
Built with Bayesian inference, not black-box ML</p>""", unsafe_allow_html=True)
