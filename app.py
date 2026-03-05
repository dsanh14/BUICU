"""
BUICU — Belief Updating for ICU Crowding Under Uncertainty
Narrative-driven, scroll-based interactive experience.
Run:  streamlit run app.py
"""

import base64, os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
BG   = "#1A202C"       # Deep slate/navy
CARD = "#2D3748"       # Lighter elevation
TXT  = "#F1F5F9"       # Off-white
TXT2 = "#94A3B8"       # Muted slate/gray
TXT3 = "#64748B"       # Darker slate
BDR  = "rgba(255, 255, 255, 0.15)" # Subtle border
BLUE = "#22D3EE"       # Cyan accent
WARM = "#F59E0B"       # Warm accent (Amber)
SAGE = "#4ADE80"       # Mint green accent (like robots.rmrm.io)
ROSE = "#F43F5E"       # Rose red accent

SERIF = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace" # Changed to Monospace
SANS  = "'Space Grotesk', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

def _mascot():
    p = os.path.join(os.path.dirname(__file__), "assets", "mascot.png")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""
M64 = _mascot()

# =====================================================================
# CSS
# =====================================================================
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

::selection {{ background: {BLUE}; color: #000; }}
::-moz-selection {{ background: {BLUE}; color: #000; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255, 255, 255, 0.2); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {BLUE}; }}

/* ── reset & animated background ── */
@keyframes bg-drift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
.stApp {{ 
    background: linear-gradient(-45deg, {BG}, #121620, {BG}, #1A2130);
    background-size: 400% 400%;
    animation: bg-drift 15s ease infinite;
}}
#MainMenu, footer, header, section[data-testid="stSidebar"] {{ display:none !important; }}

html, body, [class*="css"],
.stApp, .stApp p, .stApp div, .stApp label,
.stMarkdown, .stMarkdown p,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em {{
    font-family: {SANS} !important;
    color: {TXT} !important;
    font-weight: 400;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
/* preserve material icon fonts */
.stApp span[class*="material"], .stApp [data-testid="stExpanderToggleIcon"],
.stApp .material-symbols-rounded, .stApp .material-symbols-outlined {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
}}
h1,h2,h3 {{
    font-family: {SERIF} !important;
    font-weight: 500; color: {TXT} !important;
    letter-spacing: -0.02em;
}}
h4,h5,h6 {{ font-family: {SANS} !important; font-weight: 600; color: {TXT} !important; }}

/* ── force Streamlit internal text visible ── */
.stApp [data-testid="stText"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"] label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"],
.stApp [data-testid="stThumbValue"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp summary p {{
    color: {TXT} !important;
}}
/* keep spans inheriting unless they're custom-styled */
.stApp span {{ color: inherit; }}

/* ── metrics ── */
[data-testid="stMetricValue"] {{ 
    font-family:{SERIF} !important; 
    font-size:1.8rem !important; 
    font-weight:700 !important; 
    color:{SAGE} !important; 
    text-shadow: 0 0 12px rgba(74, 222, 128, 0.4); 
    letter-spacing: -0.05em;
}}
[data-testid="stMetricLabel"] {{ 
    font-family:{SANS} !important; 
    font-size:0.75rem !important; 
    font-weight:600 !important; 
    color:{BLUE} !important; 
    text-transform:uppercase; 
    letter-spacing:0.08em; 
}}
[data-testid="stMetricDelta"] {{ color:{TXT} !important; opacity:0.8; }}

/* ── expanders ── */
details[data-testid="stExpander"] {{
    border:1px solid rgba(255, 255, 255, 0.1) !important; 
    border-radius:14px !important;
    background:rgba(30,36,48,0.5) !important; 
    backdrop-filter: blur(8px);
    box-shadow:0 4px 15px rgba(0,0,0,0.1) !important;
    margin-bottom:0.8rem !important;
    transition: all 0.3s ease;
}}
details[data-testid="stExpander"]:hover {{
    border-color: rgba(34, 211, 238, 0.3) !important;
    box-shadow: 0 4px 20px rgba(34, 211, 238, 0.1) !important;
}}
details[data-testid="stExpander"] summary {{
    font-family:{SANS} !important; font-weight:600 !important;
    font-size:0.95rem !important; color: {TXT} !important;
}}
details[data-testid="stExpander"] summary p {{
    font-family:{SANS} !important; font-size:0.95rem !important;
}}

/* ── sub-tabs (model eval) ── */
.stTabs [data-baseweb="tab-list"] {{
    gap:0; background:rgba(0,0,0,0.2); border-radius:12px;
    border:1px solid rgba(255,255,255,0.08); padding:4px; display:inline-flex;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:9px; padding:8px 22px; font-family:{SANS} !important;
    font-size:0.85rem !important; font-weight:600 !important; color:{TXT3} !important;
    transition: all 0.2s ease;
}}
.stTabs [data-baseweb="tab"] p {{ color:{TXT3} !important; }}
.stTabs [aria-selected="true"] {{
    background: rgba(34, 211, 238, 0.15) !important; 
    border: 1px solid rgba(34, 211, 238, 0.3) !important;
    color:{BLUE} !important;
    text-shadow: 0 0 10px rgba(34, 211, 238, 0.3);
}}
.stTabs [aria-selected="true"] p {{ color:{BLUE} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color:transparent !important; }}
.stTabs [data-baseweb="tab-border"] {{ display:none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:1.5rem; }}

/* ── Streamlit Form / Button / Slider adjustments ── */
.stSlider [data-testid="stThumbValue"] {{ 
    color: {BLUE} !important; font-family: {SERIF} !important; font-weight: bold; text-shadow: 0 0 8px rgba(34, 211, 238, 0.5); 
}}
.stButton>button {{
    background: rgba(34, 211, 238, 0.1) !important;
    border: 1px solid rgba(34, 211, 238, 0.5) !important;
    color: {BLUE} !important;
    font-family: {SANS} !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 999px !important;
    transition: all 0.3s ease !important;
}}
.stButton>button p {{
    font-weight: 800 !important;
    color: inherit !important;
}}
.stButton>button:hover {{
    background: {BLUE} !important;
    color: #000 !important;
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.6) !important;
    transform: translateY(-2px);
}}

/* ── custom classes & animations ── */
@keyframes fade-in-up {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.card {{
    background: rgba(30, 36, 48, 0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius:24px;
    padding:2.5rem 3rem; 
    box-shadow: 0 4px 15px -1px rgba(0, 0, 0, 0.2), 0 2px 8px -1px rgba(0, 0, 0, 0.15);
    margin-bottom:1.5rem;
    animation: fade-in-up 0.8s ease-out forwards;
}}
.hero-num {{
    font-family:{SERIF}; font-size:5.5rem; font-weight:800;
    color:{BLUE} !important; line-height:1; letter-spacing:-0.05em;
    text-shadow: 0 0 25px rgba(34, 211, 238, 0.4);
}}
.hero-label {{ font-family:{SANS}; font-size:1.1rem; font-weight:300; color:{TXT} !important; margin-top:0.6rem; line-height:1.6; opacity:0.9; }}
.hero-ci {{
    display:inline-block; margin-top:0.5rem; padding:5px 14px;
    background:#EEF2F6; border-radius:8px; font-size:0.85rem;
    color:{BLUE} !important; font-weight:500;
}}
.b-strip {{
    font-family:{SANS}; border-left:3px solid {WARM}; background:rgba(245, 158, 11, 0.1);
    border-radius:0 12px 12px 0; padding:1rem 1.3rem;
    margin-top:1.2rem; font-size:0.88rem; font-weight:400;
    color:{TXT} !important; line-height:1.7;
}}
.narr {{
    font-family:{SANS}; font-size:1.05rem; font-weight:300;
    color:{TXT2} !important; line-height:1.8;
    max-width:700px; margin:0 auto 2rem auto;
}}
.narr strong {{ color:{TXT} !important; }}
.sec-head {{
    text-align:center; margin:3.5rem 0 0.8rem 0;
}}
.sec-head h2 {{ margin:0; font-size:2rem; font-weight:500; }}
.sec-head p {{ font-family:{SANS}; color:{TXT3} !important; font-size:0.88rem; font-weight:400; margin:0.4rem 0 0 0; letter-spacing:0.01em; }}
.qdiv {{ border:none; border-top:1px solid {BDR}; margin:3rem auto; max-width:200px; }}
.content-shell {{
    max-width: 1120px;
    margin: 0 auto;
    padding: 0 1rem 2rem 1rem;
}}
.anchor {{
    display:block;
    position:relative;
    top:-82px;
    visibility:hidden;
}}
.jump-nav {{
    position: sticky;
    top: 16px;
    z-index: 99;
    margin: 0 auto 1.5rem auto;
    background: rgba(45, 55, 72, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    padding: 8px;
    display: flex;
    gap: 8px;
    justify-content: center;
    max-width: fit-content;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}}
.jump-nav a {{
    text-decoration: none !important;
    font-family: {SANS};
    font-size: 0.73rem;
    font-weight: 600;
    color: {TXT2} !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 7px 12px;
    border-radius: 999px;
}}
.jump-nav a:hover {{
    background: {CARD};
    color: {TXT} !important;
}}
.sec-kicker {{
    font-family:{SANS};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.68rem;
    font-weight: 700;
    color: {TXT3} !important;
    margin-bottom: 0.4rem;
}}
.viz-card {{
    background: rgba(30, 36, 48, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.2rem 1.2rem 0.5rem 1.2rem;
    box-shadow: 0 4px 20px -1px rgba(0, 0, 0, 0.3);
    margin: 1rem 0 1.5rem 0;
    transition: all 0.3s ease;
    animation: fade-in-up 1s ease-out forwards;
}}
.viz-card:hover {{
    box-shadow: 0 10px 30px -5px rgba(34, 211, 238, 0.15);
    transform: translateY(-4px);
    border-color: rgba(34, 211, 238, 0.3);
}}

/* ── concept cards ── */
.cpt-grid {{
    display:grid; grid-template-columns:repeat(4, 1fr); gap:12px;
    margin:1.5rem 0;
}}
@media (max-width:768px) {{ .cpt-grid {{ grid-template-columns:repeat(2, 1fr); }} }}
.cpt {{
    background:{CARD}; border:1px solid {BDR}; border-radius:14px;
    padding:1.1rem 1.2rem; position:relative;
    transition:transform 0.15s ease, box-shadow 0.15s ease;
}}
.cpt:hover {{
    transform:translateY(-2px);
    box-shadow:0 4px 12px rgba(0,0,0,0.06);
}}
.cpt-num {{
    display:inline-flex; align-items:center; justify-content:center;
    width:26px; height:26px; border-radius:8px;
    font-family:{SANS}; font-size:0.7rem; font-weight:600;
    margin-bottom:0.6rem; color:{CARD} !important;
}}
.cpt-name {{
    font-family:{SERIF}; font-size:0.95rem; font-weight:500;
    color:{TXT} !important; margin-bottom:0.25rem; line-height:1.3;
}}
.cpt-desc {{
    font-family:{SANS}; font-size:0.78rem; font-weight:400;
    color:{TXT3} !important; line-height:1.45;
}}
.cpt-cat {{
    font-family:{SANS}; font-size:0.7rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.06em;
    margin:2rem 0 0.6rem 0;
}}

/* ── mascot note (small) ── */
.m-note {{
    display:flex; align-items:flex-start; gap:10px;
    margin-top:1rem; padding:10px 14px;
    background:rgba(255,255,255,0.05); border-radius:12px;
}}
.m-note img {{
    width:28px; height:28px; border-radius:50%; flex-shrink:0; margin-top:1px;
    animation: mn-peek 0.5s ease-out both;
}}
.m-note:hover img {{ animation: mn-wiggle 0.4s ease-in-out; }}
.m-note p {{ font-size:0.82rem; color:{TXT} !important; line-height:1.55; margin:0; }}

/* ── mascot dialogue (bigger, inline) ── */
.m-dialog {{
    display:flex; gap:16px; align-items:flex-start;
    background:rgba(255,255,255,0.03); border:1px solid {BDR}; border-radius:16px;
    padding:1.3rem 1.6rem; margin:1.5rem 0;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);
    animation: md-slide 0.45s ease-out both;
}}
.m-dialog img {{
    width:48px; height:48px; border-radius:50%; flex-shrink:0;
    background: {CARD};
    padding: 2px;
    animation: md-bounce 0.6s cubic-bezier(0.34,1.56,0.64,1) 0.15s both;
}}
.m-dialog:hover img {{ animation: md-excited 0.5s ease-in-out; }}
.m-dialog .m-txt {{
    font-family:{SANS}; font-size:0.9rem; font-weight:400;
    color:{TXT2} !important; line-height:1.65;
    animation: md-fade 0.5s ease-out 0.2s both;
}}
.m-dialog .m-txt strong {{ color:{TXT} !important; }}

/* mascot inline animations */
@keyframes mn-peek {{
    0% {{ transform:translateX(-8px) scale(0.8); opacity:0; }}
    100% {{ transform:translateX(0) scale(1); opacity:1; }}
}}
@keyframes mn-wiggle {{
    0%,100% {{ transform:rotate(0deg); }}
    25% {{ transform:rotate(-8deg); }}
    75% {{ transform:rotate(8deg); }}
}}
@keyframes md-slide {{
    0% {{ transform:translateY(12px); opacity:0; }}
    100% {{ transform:translateY(0); opacity:1; }}
}}
@keyframes md-bounce {{
    0% {{ transform:scale(0) rotate(-10deg); }}
    60% {{ transform:scale(1.15) rotate(3deg); }}
    100% {{ transform:scale(1) rotate(0deg); }}
}}
@keyframes md-fade {{
    0% {{ opacity:0; transform:translateX(6px); }}
    100% {{ opacity:1; transform:translateX(0); }}
}}
@keyframes md-excited {{
    0%,100% {{ transform:scale(1) rotate(0deg); }}
    20% {{ transform:scale(1.1) rotate(-6deg); }}
    40% {{ transform:scale(1.15) rotate(6deg); }}
    60% {{ transform:scale(1.1) rotate(-4deg); }}
    80% {{ transform:scale(1.05) rotate(2deg); }}
}}

/* ── interactive guess ── */
.guess-card {{
    background:rgba(30,36,48,0.3); border:2px dashed rgba(34, 211, 238, 0.4); border-radius:18px;
    padding:2rem 2.4rem; text-align:center; margin:1rem 0 2rem 0;
    box-shadow: inset 0 0 20px rgba(34, 211, 238, 0.05);
}}
.guess-card h3 {{ margin:0 0 0.3rem 0; font-size:1.3rem; color:{BLUE} !important; font-family:{SERIF}; text-shadow:0 0 8px rgba(34,211,238,0.3); }}
.guess-card p {{ color:{TXT} !important; font-size:0.9rem; opacity:0.8; }}

/* ── reveal card ── */
.reveal {{
    background:linear-gradient(135deg, rgba(34,211,238,0.1) 0%, rgba(74,222,128,0.15) 100%);
    border:1px solid rgba(74,222,128,0.3); border-radius:16px;
    padding:1.6rem 2rem; margin:1rem 0;
    box-shadow: 0 0 20px rgba(74,222,128,0.1);
}}
.reveal .r-num {{
    font-family:{SERIF}; font-size:3.5rem; font-weight:800;
    color:{SAGE} !important; letter-spacing:-0.05em;
    text-shadow: 0 0 15px rgba(74, 222, 128, 0.4);
}}
.reveal .r-label {{ font-size:0.88rem; color:{TXT} !important; margin-top:0.2rem; font-family:{SANS}; text-transform:uppercase; letter-spacing:0.06em; opacity:0.8; }}

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
    width:56px; height:56px; border-radius:50%;
    background:{CARD}; padding:3px;
    box-shadow:0 4px 16px rgba(0,0,0,0.10);
    animation: fm-float 4s ease-in-out infinite, fm-wiggle 8s ease-in-out 2s infinite;
    transition:transform 0.25s cubic-bezier(0.34,1.56,0.64,1);
}}
.fm-lbl:hover .fm-img {{
    animation: fm-wave 0.6s ease-in-out;
    transform:scale(1.15);
}}
.fm-ck:checked + .fm-lbl .fm-img {{
    animation: fm-happy 0.7s ease-in-out;
    transform:scale(1.15);
}}
.fm-ring {{
    position:absolute; bottom:0; right:0; width:56px; height:56px;
    border-radius:50%; border:2px solid {BLUE};
    animation:fm-pulse 3s ease-out infinite; opacity:0;
}}
/* sparkle dots around mascot */
.fm-spark {{
    position:absolute; width:6px; height:6px; border-radius:50%;
    background:{WARM}; opacity:0;
}}
.fm-spark.s1 {{ top:-4px; right:20px; animation: fm-sparkle 4s ease-out 0s infinite; }}
.fm-spark.s2 {{ top:8px; right:-6px; animation: fm-sparkle 4s ease-out 1.3s infinite; }}
.fm-spark.s3 {{ bottom:4px; right:-4px; animation: fm-sparkle 4s ease-out 2.6s infinite; background:{SAGE}; }}
.fm-bub {{
    position:absolute; bottom:68px; right:0;
    background:{CARD}; border:1px solid {BDR}; border-radius:14px;
    padding:14px 18px 24px 18px; box-shadow:0 8px 30px rgba(0,0,0,0.50);
    width:280px; min-height:48px; height:auto;
    opacity:0; transform:translateY(8px) scale(0.94);
    transition:all 0.35s cubic-bezier(0.34,1.56,0.64,1);
    pointer-events:none;
}}
.fm-bub::after {{
    content:''; position:absolute; bottom:-7px; right:22px;
    width:12px; height:12px; background:{CARD};
    border-right:1px solid {BDR}; border-bottom:1px solid {BDR};
    transform:rotate(45deg);
}}
.fm-lbl:hover .fm-bub,
.fm-ck:checked + .fm-lbl .fm-bub {{
    opacity:1; transform:translateY(0) scale(1); pointer-events:auto;
}}
.fm-bub-inner {{
    display: grid;
    width: 100%;
}}
.fm-bub-inner span {{
    grid-area: 1 / 1;
    font-size:0.82rem;
    color:{TXT} !important; line-height:1.5; opacity:0;
}}
.fm-bub-inner span.anim {{
    animation:ft 24s infinite;
}}
.fm-bub-inner span.static {{
    opacity:1; animation:none; font-weight: 500;
}}
.fm-bub .f1 {{ animation-delay:0s; }}
.fm-bub .f2 {{ animation-delay:6s; }}
.fm-bub .f3 {{ animation-delay:12s; }}
.fm-bub .f4 {{ animation-delay:18s; }}
.fm-bub .tip-tag {{
    position:absolute; bottom:8px; right:14px;
    font-size:0.6rem; color:{TXT3} !important; opacity:0.5; animation:none !important;
}}

@keyframes ft {{ 0%,3% {{ opacity:0; }} 5%,22% {{ opacity:1; }} 25%,100% {{ opacity:0; }} }}
@keyframes fm-float {{ 
    0%,100% {{ transform:translateY(0); }} 
    50% {{ transform:translateY(-8px); }} 
}}
@keyframes fm-breathe {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
}}
@keyframes fm-pulse {{ 0% {{ opacity:0.8;transform:scale(0.9); }} 100% {{ opacity:0;transform:scale(1.8); }} }}
@keyframes fm-wiggle {{
    0%,90%,100% {{ transform:rotate(0deg); }}
    93% {{ transform:rotate(-8deg); }}
    96% {{ transform:rotate(8deg); }}
}}
@keyframes fm-wave {{
    0% {{ transform:scale(1) rotate(0deg); }}
    25% {{ transform:scale(1.2) rotate(-15deg); }}
    50% {{ transform:scale(1.15) rotate(15deg); }}
    75% {{ transform:scale(1.2) rotate(-10deg); }}
    100% {{ transform:scale(1.2) rotate(0deg); }}
}}
@keyframes fm-happy {{
    0% {{ transform:scale(1); }}
    20% {{ transform:scale(1.25) translateY(-12px) rotate(-10deg); }}
    40% {{ transform:scale(1.1) translateY(0) rotate(10deg); }}
    60% {{ transform:scale(1.2) translateY(-8px) rotate(-5deg); }}
    80% {{ transform:scale(1.1) translateY(-2px) rotate(5deg); }}
    100% {{ transform:scale(1.15) translateY(0); }}
}}
@keyframes fm-sparkle {{
    0%,85%,100% {{ opacity:0; transform:scale(0) rotate(0deg); }}
    90% {{ opacity:1; transform:scale(1.2) rotate(45deg); }}
    95% {{ opacity:0; transform:scale(1.8) translateY(-10px) rotate(90deg); }}
}}
</style>""", unsafe_allow_html=True)

# ── Matplotlib ──
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none",
    "axes.edgecolor": "#FFFFFF33", "axes.grid": True,  # 20% opacity white
    "grid.alpha": 0.2, "grid.color": "#FFFFFF66", "grid.linestyle": "--", # 40% opacity white
    "font.size": 11, "font.family": "monospace", "font.monospace": ["JetBrains Mono", "Fira Code", "Courier New"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": True,
    "text.color": TXT, "axes.labelcolor": TXT,
    "xtick.color": TXT2, "ytick.color": TXT2,
    "xtick.bottom": False, "ytick.left": False,
})

# ── Mascot Helpers ──
# ── Helpers ──
def mnote(txt):
    if M64:
        st.markdown(f'<div class="m-note"><img src="data:image/png;base64,{M64}"/>'
                    f'<p>{txt}</p></div>', unsafe_allow_html=True)
    else:
        st.caption(txt)

def mascot_says(txt):
    if M64:
        st.markdown(f'<div class="m-dialog"><img src="data:image/png;base64,{M64}"/>'
                    f'<div class="m-txt">{txt}</div></div>', unsafe_allow_html=True)
    else:
        st.info(txt)

def section(title, sub="", anchor="", kicker="SECTION"):
    html = ""
    if anchor:
        html += f'<span id="{anchor}" class="anchor"></span>'
    html += '<div class="sec-head">'
    if kicker:
        html += f'<div class="sec-kicker">{kicker}</div>'
    html += f'<h2>{title}</h2>'
    if sub:
        html += f'<p>{sub}</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="qdiv"/>', unsafe_allow_html=True)

def narrate(txt):
    st.markdown(f'<div class="narr">{txt}</div>', unsafe_allow_html=True)


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


# =====================================================================
#  1. HERO
# =====================================================================
st.markdown('<div class="content-shell">', unsafe_allow_html=True)
st.markdown(f"""<div style="text-align:center; padding:3.5rem 1rem 1rem 1rem">
<h1 style="font-size:3.2rem; margin:0; letter-spacing:-0.02em; 
color:{TXT}; text-shadow: 0 0 10px {BLUE}, 0 0 20px {BLUE}, 0 0 30px {BLUE};">
Belief Updating for ICU Crowding Under Uncertainty</h1>
<p style="color:{TXT3} !important; font-size:0.78rem; margin:0.3rem 0 0 0;
font-family:{SANS}; font-weight:300">
CS109 Challenge Project</p>
</div>""", unsafe_allow_html=True)
st.markdown("""
<div class="jump-nav">
  <a href="#guess">Guess</a>
  <a href="#belief">Beliefs</a>
  <a href="#forecast">Forecast</a>
  <a href="#evaluation">Evaluation</a>
  <a href="#concepts">Concepts</a>
  <a href="#ethics">Ethics</a>
</div>
""", unsafe_allow_html=True)

narrate(
    "An ICU has 50 beds and an unpredictable stream of patients. "
    "How full will it be tomorrow? Next week? "
    "We can't know for sure \u2014 but we can <strong>quantify our uncertainty</strong> "
    "and update it as new data arrives. That's the core idea."
)

divider()

# =====================================================================
#  2. INTERACTIVE: GUESS THE RATE
# =====================================================================
section("Can you guess the arrival rate?",
        "Before seeing any data, what's your intuition?",
        anchor="guess", kicker="")

narrate(
    "This ICU sees a stream of admissions every day. "
    "Some days are quiet, others bring a surge. "
    "What do you think the average daily admission rate is?"
)

if "guess_locked" not in st.session_state:
    st.session_state.guess_locked = False

user_guess = st.slider(
    "Your guess: average admissions per day",
    min_value=1.0, max_value=30.0, value=10.0, step=0.5,
    disabled=st.session_state.guess_locked,
)

if not st.session_state.guess_locked:
    st.button("\U0001F512 Lock in my guess", key="lock_btn",
              on_click=lambda: setattr(st.session_state, "guess_locked", True),
              use_container_width=True)
    st.markdown(f"""<div style="text-align:center; padding:2rem 1.5rem;
    background:{CARD}; border:2px dashed {BDR}; border-radius:16px;
    margin-top:0.5rem">
    <p style="color:{TXT3} !important; font-size:1rem; margin:0">
    \U0001F50D The Bayesian answer is hidden until you lock in your guess.</p>
    </div>""", unsafe_allow_html=True)
else:
    true_lambda = model.belief.mean
    ci = model.belief.credible_interval(0.95)
    diff = abs(user_guess - true_lambda)

    if diff < 1.0:
        verdict = "Impressive intuition!"
        v_color = SAGE
    elif diff < 3.0:
        verdict = "Not bad \u2014 close."
        v_color = WARM
    else:
        verdict = "The data tells a different story."
        v_color = ROSE

    st.markdown(f"""<div class="reveal">
    <div class="r-num">{true_lambda:.1f}</div>
    <div class="r-label">admissions/day &mdash; posterior mean after 180 days</div>
    <div style="margin-top:0.6rem; font-size:0.9rem; color:{v_color} !important; font-weight:600">{verdict}</div>
    <div style="font-size:0.8rem; color:{TXT3} !important; margin-top:0.3rem">
    Your guess: {user_guess:.1f} &nbsp;&middot;&nbsp; Bayesian: {true_lambda:.1f}
    &nbsp;&middot;&nbsp; 95% Credible Interval: [{ci[0]:.1f}, {ci[1]:.1f}]</div>
    </div>""", unsafe_allow_html=True)

    mascot_says(
        f"Your guess was <strong>{user_guess:.1f}</strong>. "
        f"The model started with a vague prior (mean 10.0/day) and after observing "
        f"<strong>{int(model.belief.total_arrivals):,} admissions</strong> over 180 days, "
        f"the posterior concentrated around <strong>{true_lambda:.1f}</strong>. "
        f"That's Bayesian updating \u2014 beliefs shift toward evidence."
    )

    _, c_btn, _ = st.columns([1, 1, 1])
    with c_btn:
        if st.button("\U0001F504 Try again", key="reset_btn", use_container_width=True):
            st.session_state.guess_locked = False
            st.rerun()

divider()

# =====================================================================
#  3. BELIEF EVOLUTION (the main story)
# =====================================================================
section("Watch beliefs update",
        "180 days of data, one day at a time",
        anchor="belief", kicker="BELIEF DYNAMICS")

narrate(
    "We start with a vague prior belief about the arrival rate \u03BB. "
    "Each day, new admissions arrive and we update. "
    "The credible interval narrows as evidence accumulates \u2014 "
    "except during <strong>surges</strong>, where the model detects "
    "something has changed."
)

h = model.history
t_arr = np.array(h.times); m_arr = np.array(h.means)
cl_arr = np.array(h.ci_lows); ch_arr = np.array(h.ci_highs)
obs_arr = np.array(h.observed_counts)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(t_arr[1:], cl_arr[1:], ch_arr[1:], alpha=0.15, color=BLUE,
                label="95% credible interval")
ax.plot(t_arr[1:], m_arr[1:], color=BLUE, linewidth=2.2,
        label="Posterior mean \u03BB")
ax.scatter(t_arr[1:], obs_arr[1:], s=6, color=TXT3, alpha=0.25, zorder=3,
           label="Observed counts")
for s, e in config.surge_windows:
    ax.axvspan(s, e, alpha=0.07, color=WARM, label="Surge" if s == config.surge_windows[0][0] else "")
ax.set_xlabel("Day"); ax.set_ylabel("\u03BB (admissions/day)")
ax.set_title("Posterior Belief Evolution", fontsize=14, pad=14)
ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
plt.tight_layout()
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
plt.close(fig)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Final \u03BB", f"{model.belief.mean:.2f}/day")
m2.metric("Observations", f"{int(model.belief.total_arrivals):,}")
m3.metric("Anomalous days", f"{sum(model.history.anomaly_flags)}")
vd_now = VarianceDecomposition.decompose_at_belief(model.belief)
m4.metric("Epistemic %", f"{100*vd_now['parameter_fraction']:.1f}%")

mnote(
    "<strong>Key insight:</strong> The interval narrows over time "
    "(we're learning \u03BB) but never collapses to zero "
    "(arrivals are inherently random)."
)

divider()

# =====================================================================
#  4. INTERACTIVE BELIEF UPDATING
# =====================================================================
section("Explore the prior \u2192 posterior update",
        "Change the prior. Scrub through time. See it converge.",
        kicker="EXPERIMENT")

narrate(
    "What happens if you start with a completely wrong prior? "
    "Try setting \u03B1\u2080=50, \u03B2\u2080=10 (a prior believing \u03BB\u22485). "
    "Even then, after enough data, the posterior converges to the truth. "
    "<strong>Evidence overwhelms belief.</strong>"
)

ic1, ic2, ic3 = st.columns(3)
a0 = ic1.slider("\u03B1\u2080 (prior shape)", 0.1, 50.0, 2.0, 0.1)
b0 = ic2.slider("\u03B2\u2080 (prior rate)", 0.01, 10.0, 0.2, 0.01)
day_t = ic3.slider("Observe through day", 1, len(daily_counts), len(daily_counts))

um = BayesianArrivalModel(a0, b0)
um.sequential_update(daily_counts[:day_t])
ub = um.belief; uci = ub.credible_interval(0.95)

ic_m1, ic_m2, ic_m3 = st.columns(3)
ic_m1.metric("Posterior mean", f"{ub.mean:.3f}")
ic_m2.metric("95% CI", f"[{uci[0]:.2f}, {uci[1]:.2f}]")
ud = VarianceDecomposition.decompose_at_belief(ub)
ic_m3.metric("Epistemic %", f"{100*ud['parameter_fraction']:.1f}%")

fig2, (axd, axe) = plt.subplots(1, 2, figsize=(14, 4.5))
xm = max(a0/b0*3, ub.mean*1.5, 20)
x = np.linspace(0.01, xm, 400)
axd.fill_between(x, stats.gamma.pdf(x, a=a0, scale=1/b0), alpha=0.18, color=ROSE)
axd.plot(x, stats.gamma.pdf(x, a=a0, scale=1/b0), color=ROSE, lw=1.5, ls="--", label="Prior")
axd.fill_between(x, stats.gamma.pdf(x, a=ub.alpha, scale=1/ub.beta), alpha=0.3, color=BLUE)
axd.plot(x, stats.gamma.pdf(x, a=ub.alpha, scale=1/ub.beta), color=BLUE, lw=2, label=f"Posterior (day {day_t})")
axd.set_xlabel("\u03BB"); axd.set_ylabel("Density")
axd.set_title("Prior \u2192 Posterior", fontsize=12); axd.legend(fontsize=8.5)

uh = um.history; ut = np.array(uh.times)
if len(ut) > 1:
    axe.fill_between(ut[1:], np.array(uh.ci_lows)[1:], np.array(uh.ci_highs)[1:], alpha=0.15, color=BLUE)
    axe.plot(ut[1:], np.array(uh.means)[1:], color=BLUE, lw=2)
    for s, e in config.surge_windows:
        if s < day_t: axe.axvspan(s, min(e, day_t), alpha=0.07, color=WARM)
axe.set_xlabel("Day"); axe.set_ylabel("\u03BB")
axe.set_title("Belief Trajectory", fontsize=12)
plt.tight_layout()
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)
plt.close(fig2)

mascot_says(
    f"With your prior (\u03B1\u2080={a0:.1f}, \u03B2\u2080={b0:.1f}), "
    f"the starting mean was <strong>{a0/b0:.1f}</strong>/day. "
    f"After {day_t} days, the posterior mean is <strong>{ub.mean:.2f}</strong>. "
    f"The KL divergence from prior to posterior is "
    f"<strong>{kl_divergence_gamma(ub.alpha, ub.beta, a0, b0):.1f} nats</strong> "
    f"\u2014 that's how much you learned."
)

divider()

# =====================================================================
#  5. UNCERTAINTY DECOMPOSITION
# =====================================================================
section("Where does uncertainty come from?",
        "Law of Total Variance: epistemic vs. aleatoric",
        kicker="UNCERTAINTY")

narrate(
    "Not all uncertainty is created equal. "
    "<strong>Epistemic</strong> uncertainty (about \u03BB) shrinks with more data. "
    "<strong>Aleatoric</strong> uncertainty (Poisson randomness) is irreducible. "
    "The Law of Total Variance separates them."
)

st.latex(r"\mathrm{Var}[N] = \underbrace{E[\mathrm{Var}[N|\lambda]]}_{\text{aleatoric}} + \underbrace{\mathrm{Var}[E[N|\lambda]]}_{\text{epistemic}}")

vr = VarianceDecomposition.decompose_over_time(um.history)
fig3, (va1, va2) = plt.subplots(1, 2, figsize=(14, 4))
tt = vr["times"]
va1.fill_between(tt, 0, vr["stochastic"], alpha=0.35, color=BLUE, label="Aleatoric (stochastic)")
va1.fill_between(tt, vr["stochastic"], vr["total"], alpha=0.35, color=WARM, label="Epistemic (parameter)")
va1.set_xlabel("Day"); va1.set_ylabel("Variance"); va1.set_title("Variance Over Time", fontsize=12)
va1.legend(fontsize=8.5)
va2.fill_between(tt, 0, vr["stochastic_frac"], alpha=0.4, color=BLUE, label="Aleatoric")
va2.fill_between(tt, vr["stochastic_frac"], 1, alpha=0.4, color=WARM, label="Epistemic")
va2.set_xlabel("Day"); va2.set_ylabel("Fraction"); va2.set_ylim(0, 1)
va2.set_title("Composition", fontsize=12); va2.legend(fontsize=8.5)
plt.tight_layout()
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.pyplot(fig3)
st.markdown('</div>', unsafe_allow_html=True)
plt.close(fig3)

vf = VarianceDecomposition.decompose_at_belief(ub)
mascot_says(
    f"After {day_t} days, <strong>{100*vf['stochastic_fraction']:.0f}%</strong> of "
    f"forecast variance is irreducible Poisson noise. "
    f"Collecting more data won't help \u2014 only changing the underlying process can. "
    f"This is the fundamental epistemic/aleatoric distinction."
)

divider()

# =====================================================================
#  6. CROWDING FORECAST
# =====================================================================
section("Will the ICU be overcrowded?",
        "2,000 Monte Carlo trajectories propagate all uncertainty",
        anchor="forecast", kicker="FORECAST")

narrate(
    "This is the question that matters. We sample \u03BB from the posterior, "
    "simulate arrivals, draw lengths-of-stay, and track occupancy. "
    "Each trajectory is one possible future. "
    "The fraction exceeding capacity <strong>is</strong> the probability."
)

fc1, fc2 = st.columns(2)
ucap = fc1.slider("ICU capacity (beds)", 20, 80, config.capacity)
uhrs = fc2.slider("Forecast horizon (hours)", 12, 96, 48, 6)

sr, pc, co, sd = get_sim(daily_counts[:36], data["los_hours"],
                         data["admissions"], data["discharges"], ucap, uhrs)

st.markdown(f"""<div style="text-align:center; margin:1.5rem 0">
<div class="hero-num">{100*pc:.0f}%</div>
<div class="hero-label">probability of exceeding {ucap} beds within {uhrs} hours</div>
<div style="font-size:0.85rem; color:{TXT3} !important; margin-top:0.4rem">
Currently {co} of {ucap} beds occupied (day {sd})</div>
</div>""", unsafe_allow_html=True)

fig4, ax4 = plt.subplots(figsize=(14, 4.5))
tg = sr["time_grid"]
ax4.fill_between(tg, sr["ci_low"], sr["ci_high"], alpha=0.14, color=BLUE, label="95% CI")
ax4.plot(tg, sr["mean"], color=BLUE, lw=2, label="Mean trajectory")
ax4.axhline(ucap, color=ROSE, ls="--", lw=1.5, label=f"Capacity ({ucap})")
ax4.set_xlabel("Hours ahead"); ax4.set_ylabel("Occupancy")
ax4.set_title(f"{uhrs}-Hour Occupancy Forecast", fontsize=12)
ax4.legend(fontsize=8.5)
plt.tight_layout()
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.pyplot(fig4)
st.markdown('</div>', unsafe_allow_html=True)
plt.close(fig4)

pk = np.max(sr["trajectories"], axis=1)
with st.expander("Sensitivity: how does capacity change the risk?"):
    caps = [int(ucap*0.8), ucap, int(ucap*1.2)]
    fig5, ax5 = plt.subplots(figsize=(10, 2.5))
    for i, c in enumerate(caps):
        p = 100*float(np.mean(pk > c)); tag = " (current)" if c == ucap else ""
        ax5.barh(f"{c} beds{tag}", p, color=[ROSE, BLUE, SAGE][i], alpha=0.6, height=0.5)
    ax5.set_xlabel("P(overcrowded) %"); ax5.set_xlim(0, 100)
    ax5.set_title("Capacity Sensitivity", fontsize=10)
    plt.tight_layout()
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.pyplot(fig5)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig5)

mascot_says(
    f"This probability is <strong>not</strong> a point estimate. "
    f"It's the fraction of {sr['trajectories'].shape[0]:,} simulated futures "
    f"where peak occupancy exceeded capacity. "
    f"Each trajectory drew a different \u03BB, different arrivals, "
    f"and different lengths-of-stay."
)

divider()

# =====================================================================
#  7. MODEL EVALUATION
# =====================================================================
section("How good is this model?",
        "Proper scoring, calibration, and honest failure analysis",
        anchor="evaluation", kicker="VALIDATION")

narrate(
    "A probability model that can't be evaluated is just a story. "
    "We compare two models (stationary vs. windowed), "
    "pit Bayesian against frequentist MLE, "
    "and document every way this model could break."
)

et1, et2, et3 = st.tabs(["Stationary vs. Windowed", "MLE vs. Bayesian", "Failure Modes"])

with et1:
    fig6, ax6 = plt.subplots(figsize=(14, 4.5))
    ts = np.array(model.history.times); tw = np.array(w_hist.times)
    ax6.fill_between(ts[1:], np.array(model.history.ci_lows)[1:],
                     np.array(model.history.ci_highs)[1:], alpha=0.12, color=BLUE)
    ax6.plot(ts[1:], np.array(model.history.means)[1:], color=BLUE, lw=2, label="Stationary")
    ax6.plot(tw, np.array(w_hist.means), color=SAGE, lw=2, label="Windowed (14d)")
    ax6.scatter(ts[1:], np.array(model.history.observed_counts)[1:], s=6, color=TXT3, alpha=0.2, zorder=3)
    for s, e in config.surge_windows: ax6.axvspan(s, e, alpha=0.07, color=WARM)
    ax6.set_xlabel("Day"); ax6.set_ylabel("\u03BB")
    ax6.set_title("Two Bayesian Models", fontsize=12); ax6.legend(fontsize=8.5)
    plt.tight_layout()
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.pyplot(fig6)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig6)

    sc = get_sc(daily_counts, np.array(model.history.alphas), np.array(model.history.betas),
                np.array(w_hist.alphas), np.array(w_hist.betas))
    s1, s2, s3 = st.columns(3)
    s1.metric("Stationary score", f"{sc['stationary_total']:.1f}")
    s2.metric("Windowed score", f"{sc['windowed_total']:.1f}")
    wn = "Windowed" if sc["difference"] > 0 else "Stationary"
    s3.metric("Winner", wn, f"+{abs(sc['difference']):.1f}")

    mnote("Log predictive score is a <strong>proper scoring rule</strong> \u2014 "
          "the model can't game it by being overconfident or underconfident.")

with et2:
    ml = get_mle(daily_counts)
    fig7, (a71, a72) = plt.subplots(1, 2, figsize=(14, 4))
    d = ml["days"]
    a71.plot(d, ml["mle_means"], color=ROSE, lw=1.5, label="MLE", alpha=0.8)
    a71.plot(d, ml["bayes_means"], color=BLUE, lw=1.5, label="Bayesian", alpha=0.8)
    for s, e in config.surge_windows: a71.axvspan(s, e, alpha=0.07, color=WARM)
    a71.set_xlabel("Day"); a71.set_ylabel("\u03BB"); a71.set_title("Estimates Converge", fontsize=12)
    a71.legend(fontsize=8.5)
    a72.plot(d, ml["mle_ci_hi"]-ml["mle_ci_lo"], color=ROSE, lw=1.5, label="Frequentist CI")
    a72.plot(d, ml["bayes_ci_hi"]-ml["bayes_ci_lo"], color=BLUE, lw=1.5, label="Bayesian CI")
    a72.set_xlabel("Day"); a72.set_ylabel("Width"); a72.set_title("Interval Width", fontsize=12)
    a72.legend(fontsize=8.5)
    plt.tight_layout()
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.pyplot(fig7)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig7)

    mascot_says(
        "Both converge \u2014 that's the <strong>Bernstein\u2013von Mises theorem</strong>. "
        "But the Bayesian interval is wider early on. "
        "That's not a flaw \u2014 it's <strong>more honest</strong> about what we don't yet know."
    )

with et3:
    st.metric("Combined CI widening", f"\u00d7{penalty:.2f}")
    sv = {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534"}
    for r in reports:
        ic = sv.get(r.severity, "\u26AA")
        det = "DETECTED" if r.detected else "not detected"
        with st.expander(f"{ic} {r.name} ({r.severity}) \u2014 {det}"):
            st.markdown(f"**Assumption:** {r.assumption}")
            st.markdown(f"**How it breaks:** {r.how_it_breaks}")
            st.markdown(f"**Consequence:** {r.consequence}")
            st.markdown(f"**Mitigation:** {r.mitigation}")
            st.markdown(f"**CI widening: \u00d7{r.confidence_penalty:.2f}**")

    mascot_says(
        "When a failure mode is detected, we <strong>widen the credible interval</strong> "
        f"by \u00d7{penalty:.2f}. The model doesn't pretend its assumptions hold when they don't."
    )

divider()

# =====================================================================
#  8. CS109 CONCEPTS
# =====================================================================
section("16 CS109 concepts in one project",
        "Spanning probability, inference, simulation, and evaluation",
        anchor="concepts", kicker="LEARNING OUTCOMES")

categories = [
    ("Probability Foundations", BLUE, [
        ("Random Variables", "N\u209C, L, O\u209C, \u03BB"),
        ("Distributions", "Poisson, Gamma, NegBin, LogNormal"),
        ("Conditional Prob", "P(N\u209C|\u03BB), P(\u03BB|data)"),
        ("Bayes\u2019 Theorem", "Prior \u00d7 Likelihood = Posterior"),
    ]),
    ("Bayesian Inference", WARM, [
        ("Post. Predictive", "Integrate out \u03BB \u2192 NegBin"),
        ("Conjugate Priors", "Gamma-Poisson \u2192 exact"),
        ("Total Variance", "Epistemic vs. aleatoric"),
        ("Monte Carlo", "2,000+ trajectories"),
    ]),
    ("Frequentist Methods", SAGE, [
        ("MLE", "\u03BB\u0302 = \u03A3k / T"),
        ("CLT", "Frequentist confidence interval"),
        ("KL Divergence", "Information gain quantification"),
        ("Hypothesis Testing", "Posterior predictive p-values"),
    ]),
    ("Model Evaluation", ROSE, [
        ("Model Comparison", "Log predictive score (proper)"),
        ("Calibration", "Coverage probability, PIT"),
        ("Sensitivity", "Assumptions \u2192 P(crowded)"),
        ("Prior Sensitivity", "3 priors \u2192 convergence"),
    ]),
]

n = 1
for cat_name, cat_color, items in categories:
    st.markdown(f'<div class="cpt-cat" style="color:{cat_color} !important">{cat_name}</div>',
                unsafe_allow_html=True)
    html = '<div class="cpt-grid">'
    for name, desc in items:
        html += (f'<div class="cpt">'
                 f'<div class="cpt-num" style="background:{cat_color}">{n}</div>'
                 f'<div class="cpt-name">{name}</div>'
                 f'<div class="cpt-desc">{desc}</div>'
                 f'</div>')
        n += 1
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

with st.expander("Prior sensitivity: 3 different priors, same convergence"):
    sens = get_ps(daily_counts)
    fig9, ax9 = plt.subplots(figsize=(13, 4))
    for (nm, hi), c in zip(sens.items(), [ROSE, BLUE, SAGE]):
        t = np.array(hi.times)
        ax9.plot(t[1:], np.array(hi.means)[1:], color=c, lw=2, label=nm)
    for s, e in config.surge_windows: ax9.axvspan(s, e, alpha=0.07, color=WARM)
    ax9.set_xlabel("Day"); ax9.set_ylabel("\u03BB")
    ax9.set_title("All Priors Converge", fontsize=12); ax9.legend(fontsize=8.5)
    plt.tight_layout()
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.pyplot(fig9)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig9)
    mnote("This is perhaps the most reassuring result: the data speaks louder than the prior.")

with st.expander("The probabilistic model"):
    st.latex(r"\lambda \sim \text{Gamma}(\alpha_0,\beta_0), \quad N_t|\lambda \sim \text{Pois}(\lambda \Delta t), \quad \lambda|\text{data} \sim \text{Gamma}(\alpha_0+\Sigma k,\;\beta_0+T)")
    los_d = data["los_hours"]/24; vl = los_d[~np.isnan(los_d)]
    fig8, (al1, al2) = plt.subplots(1, 2, figsize=(13, 3.2))
    al1.hist(vl, bins=80, density=True, color=BLUE, alpha=0.4, edgecolor="white", lw=0.3)
    al1.axvline(np.median(vl), color=ROSE, ls="--", label=f"median={np.median(vl):.1f}d")
    al1.set_xlabel("Days"); al1.set_title("Length of Stay", fontsize=10); al1.legend(fontsize=8)
    al2.hist(vl, bins=80, density=True, color=WARM, alpha=0.4, edgecolor="white", lw=0.3)
    al2.set_yscale("log"); al2.set_xlabel("Days"); al2.set_title("LOS (log scale \u2014 heavy tail)", fontsize=10)
    plt.tight_layout()
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.pyplot(fig8)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig8)

divider()

# =====================================================================
#  9. ETHICAL REFLECTION
# =====================================================================
section("Ethical reflection", "What this model should and shouldn't do",
        anchor="ethics", kicker="RESPONSIBILITY")

narrate(
    "A model that influences ICU staffing carries real consequences. "
    "We built BUICU to make uncertainty visible, not to make predictions impressive."
)

st.markdown("""
- **Synthetic data** \u2014 no real patient information was used
- **Never a point estimate** \u2014 every output carries a credible interval
- **5 failure modes** documented, each widening the uncertainty
- **Not for deployment** \u2014 this augments, never replaces, clinical judgment
- **Goodhart's Law** \u2014 a model that influences its target becomes unreliable
""")



st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──
st.markdown(f"""
<div style="text-align:center; padding:3rem 0 1.5rem 0">
<hr class="qdiv"/>
<p style="color:{TXT3} !important; font-size:0.75rem; font-family:{SANS}">
BUICU &middot; CS109 Challenge Project &middot;
Built with Bayesian inference, not black-box ML</p>
</div>""", unsafe_allow_html=True)

# ── Mascot Chatbot ──
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

ans = ""
q = st.chat_input("💬 Ask the Mascot about terms (e.g., Prior, Poisson, Surge, ICU, Census)...")

if q:
    with st.spinner("The Mascot is thinking..."):
        try:
            from google import genai
            import os
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key and hasattr(st.secrets, "GEMINI_API_KEY"):
                api_key = st.secrets.GEMINI_API_KEY
                
            # Verify API key exists
            if not api_key:
                ans = "Oops! I encountered an error connecting to my brain. Please make sure your GEMINI_API_KEY environment variable is set or secrets.toml is configured!"
            else:
                client = genai.Client(api_key=api_key)
                prompt = f"""You are a helpful robotic mascot for a web app called BUICU (Belief Updating for ICU Crowding Under Uncertainty), a CS109 challenge project.
BUICU's goal is to simulate how a hospital Intensive Care Unit (ICU) with 50 beds can forecast patient crowding using Bayesian statistics. It demonstrates how models can quantify uncertainty rather than just predicting a single number.
Your job is to answer the user's question concisely in 1-4 sentences.
Focus on explaining the BUICU project, its goals, or terms related to Bayesian statistics, forecasting, and hospital operations. 
User question: {q}"""
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                ans = response.text
                st.session_state.chat_open = True
        except Exception as e:
            ans = f"My brain had a hiccup! Error: {str(e)}"
            st.session_state.chat_open = True

if M64:
    # Mascot toggle logic
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("\U0001F916 Chat", key="mascot_btn", help="Click to chat with the Mascot!", use_container_width=True):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

    checked_attr = 'checked="checked"' if q or st.session_state.chat_open else ''
    
    if st.session_state.chat_open:
        st.markdown("""<style>
        .stChatInput { display: block !important; margin-bottom: 80px; }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
        .stChatInput { display: none !important; }
        </style>""", unsafe_allow_html=True)

    if ans:
        bubble_html = f'''
<div class="fm-bub-inner">
    <span class="static">{ans}</span>
</div>
<span class="tip-tag">Dismiss</span>'''
    else:
        bubble_html = '''
<div class="fm-bub-inner">
    <span class="anim f1">Every number carries a credible interval. We never hide uncertainty.</span>
    <span class="anim f2">After 180 days, 99% of forecast variance is irreducible Poisson noise.</span>
    <span class="anim f3">The model knows what it doesn't know. Surprises widen the interval.</span>
    <span class="anim f4">Beliefs update. Uncertainty narrows. That's Bayes' theorem.</span>
</div>
<span class="tip-tag" style="animation:none !important; opacity:0.5;">click to pin</span>'''

    st.markdown(f"""
<div class="fm">
    <input type="checkbox" id="fmck" class="fm-ck" {checked_attr}/>
    <label for="fmck" class="fm-lbl">
        <div class="fm-bub">
            {bubble_html}
        </div>
        <div style="position:relative">
            <img class="fm-img" src="data:image/png;base64,{M64}" alt=""/>
            <div class="fm-ring"></div>
            <div class="fm-spark s1"></div>
            <div class="fm-spark s2"></div>
            <div class="fm-spark s3"></div>
        </div>
    </label>
</div>""", unsafe_allow_html=True)
