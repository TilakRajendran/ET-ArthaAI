"""
ET ArthaAI — Autonomous Money Mentor
Hackathon-ready, single-file Streamlit application.

Dependencies:
    pip install streamlit crewai plotly pandas python-dotenv

Set GOOGLE_API_KEY in a .env file or as an environment variable.
"""

import os
import io
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Load `.env` next to this file — Streamlit’s cwd is often not the project folder.
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Optional heavy imports (graceful degradation if not installed) ──────────
try:
    # CrewAI expects `crewai.LLM` (or a model string), not LangChain chat models.
    from crewai import Agent, Crew, LLM, Process, Task

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


def _crewai_gemini_model() -> str:
    """LiteLLM id for Google AI Studio, e.g. gemini/gemini-2.5-flash."""
    raw = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    if not raw:
        raw = "gemini-2.5-flash"
    if "/" in raw:
        return raw
    return f"gemini/{raw}"

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ET ArthaAI — Autonomous Money Mentor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  (dark luxury theme)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-main:    #0B0F19;
    --bg-sidebar: #111827;
    --bg-card:    #141A27;
    --bg-card2:   #1A2235;
    --et-red:     #DA250A;
    --et-gold:    #C9A84C;
    --et-blue:    #3B82F6;
    --text-prim:  #F0F4FF;
    --text-sec:   #8A95B0;
    --border:     rgba(201,168,76,0.18);
    --glow-red:   0 0 18px rgba(218,37,10,0.45);
    --glow-gold:  0 0 18px rgba(201,168,76,0.40);
    --radius:     16px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-main) !important;
    color: var(--text-prim) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-prim) !important; }

/* ── Headings ── */
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

/* ── Cards ── */
.artha-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 20px;
    transition: box-shadow .25s ease;
}
.artha-card:hover { box-shadow: var(--glow-gold); }
.artha-card-active {
    border-color: var(--et-red) !important;
    box-shadow: var(--glow-red) !important;
}

/* ── Metric tiles ── */
.metric-tile {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--et-gold);
    margin: 0;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-sec);
    letter-spacing: .08em;
    text-transform: uppercase;
}

/* ── Terminal / Reasoning log ── */
.terminal-box {
    background: #050810;
    border: 1px solid rgba(218,37,10,0.30);
    border-radius: 10px;
    padding: 18px 22px;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem;
    color: #7EFFA0;
    line-height: 1.7;
    max-height: 480px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.terminal-box .t-red   { color: #FF5C5C; }
.terminal-box .t-gold  { color: var(--et-gold); }
.terminal-box .t-blue  { color: #60C0FF; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--et-red) 0%, #A81B07 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: .04em;
    box-shadow: var(--glow-red);
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 28px rgba(218,37,10,0.7);
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-sec) !important;
    border-bottom: 2px solid transparent;
    padding-bottom: 6px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--et-gold) !important;
    border-bottom-color: var(--et-gold) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .stSlider > div > div > div > div {
    background: var(--et-red) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Selectbox / Multiselect ── */
[data-testid="stMultiSelect"] > div, [data-testid="stSelectbox"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}

/* ── Section divider ── */
.et-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--et-gold), transparent);
    margin: 28px 0;
}

/* ── Logo / Brand ── */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
}
.brand-badge {
    background: var(--et-red);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.95rem;
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: .06em;
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-prim);
}
.brand-name span { color: var(--et-gold); }

/* ── Agent status pill ── */
.agent-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Share Tech Mono', monospace;
    margin-left: 8px;
}
.pill-idle    { background: rgba(138,149,176,0.15); color: var(--text-sec); }
.pill-running { background: rgba(218,37,10,0.20);  color: #FF7A6A; border: 1px solid var(--et-red); }
.pill-done    { background: rgba(201,168,76,0.15); color: var(--et-gold); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
defaults = {
    "agent_output": "",
    "reasoning_log": "",
    "agents_ran": False,
    "age": 30,
    "income": 1200000,
    "target_retirement": 55,
    "selected_funds": [],
    "tax_regime": "New",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & HARDCODED DATA
# ═══════════════════════════════════════════════════════════════════════════════
FUND_HOLDINGS: dict[str, list[str]] = {
    "Parag Parikh Flexi Cap": ["HDFC Bank", "ITC", "Bajaj Holdings", "Microsoft"],
    "HDFC Top 100":           ["HDFC Bank", "ICICI Bank", "Reliance", "Infosys"],
    "Quant Active":           ["Reliance", "HDFC Bank", "Adani Power"],
}

# Approximate equal-weight % within each fund
FUND_WEIGHTS: dict[str, dict[str, float]] = {
    "Parag Parikh Flexi Cap": {"HDFC Bank": 9.2,  "ITC": 8.5,  "Bajaj Holdings": 6.3, "Microsoft": 11.4},
    "HDFC Top 100":           {"HDFC Bank": 12.7, "ICICI Bank": 10.1, "Reliance": 9.8, "Infosys": 8.9},
    "Quant Active":           {"Reliance": 11.2,  "HDFC Bank": 10.5, "Adani Power": 8.3},
}

INFLATION_RATE   = 0.06   # 6 %
WITHDRAWAL_RATE  = 0.04   # 4 % (FIRE rule)
EXPECTED_RETURNS = 0.12   # 12 % CAGR assumption

# Old-regime tax slabs FY 2024-25 (with standard deduction ₹50k)
OLD_SLABS = [
    (250_000, 0.00),
    (500_000, 0.05),
    (1_000_000, 0.20),
    (float("inf"), 0.30),
]

# New-regime tax slabs FY 2024-25 (standard deduction ₹75k from Budget 2024)
NEW_SLABS = [
    (300_000, 0.00),
    (600_000, 0.05),
    (900_000, 0.10),
    (1_200_000, 0.15),
    (1_500_000, 0.20),
    (float("inf"), 0.30),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_inr(n: float) -> str:
    """Format large numbers in Indian crore/lakh notation."""
    if n >= 1e7:
        return f"₹{n/1e7:.2f} Cr"
    if n >= 1e5:
        return f"₹{n/1e5:.2f} L"
    return f"₹{n:,.0f}"


# ── FIRE Calculator ─────────────────────────────────────────────────────────

def compute_fire(age: int, income: float, target_retirement: int,
                 monthly_savings_pct: float = 0.30) -> dict:
    years_to_retire = max(target_retirement - age, 1)
    annual_savings   = income * monthly_savings_pct
    # Future value of savings at EXPECTED_RETURNS
    fv_savings = annual_savings * (((1 + EXPECTED_RETURNS) ** years_to_retire - 1) / EXPECTED_RETURNS)
    # Inflation-adjusted annual expenses at retirement
    current_expenses = income * 0.70
    future_expenses  = current_expenses * ((1 + INFLATION_RATE) ** years_to_retire)
    fire_corpus      = future_expenses / WITHDRAWAL_RATE

    # Year-by-year wealth journey
    years = list(range(age, target_retirement + 1))
    corpus = []
    accumulated = 0.0
    for _ in years:
        accumulated = accumulated * (1 + EXPECTED_RETURNS) + annual_savings
        corpus.append(accumulated)

    return {
        "years_to_retire":  years_to_retire,
        "fire_corpus":      fire_corpus,
        "projected_corpus": fv_savings,
        "gap":              max(fire_corpus - fv_savings, 0),
        "on_track":         fv_savings >= fire_corpus,
        "future_expenses":  future_expenses,
        "years":            years,
        "corpus_journey":   corpus,
    }


# ── Tax Calculator ───────────────────────────────────────────────────────────

def _slab_tax(income: float, slabs: list) -> float:
    tax = 0.0
    prev = 0.0
    for limit, rate in slabs:
        if income <= prev:
            break
        taxable = min(income, limit) - prev
        tax    += taxable * rate
        prev    = limit
    return tax


def compute_taxes(gross_income: float) -> dict:
    # Old regime: standard deduction ₹50k + 80C ₹1.5L
    old_deductions = 50_000 + 150_000
    old_taxable    = max(gross_income - old_deductions, 0)
    old_tax        = _slab_tax(old_taxable, OLD_SLABS)
    old_cess       = old_tax * 0.04
    old_total      = old_tax + old_cess

    # New regime: standard deduction ₹75k only
    new_deductions = 75_000
    new_taxable    = max(gross_income - new_deductions, 0)
    new_tax        = _slab_tax(new_taxable, NEW_SLABS)
    new_cess       = new_tax * 0.04
    new_total      = new_tax + new_cess

    return {
        "old_taxable": old_taxable,
        "old_tax":     old_tax,
        "old_cess":    old_cess,
        "old_total":   old_total,
        "new_taxable": new_taxable,
        "new_tax":     new_tax,
        "new_cess":    new_cess,
        "new_total":   new_total,
        "savings":     old_total - new_total,   # positive → new regime better
        "better":      "New" if new_total < old_total else "Old",
    }


# ── Portfolio X-Ray ──────────────────────────────────────────────────────────

def compute_overlap(selected_funds: list[str]) -> pd.DataFrame:
    """Return a DataFrame: stock × fund with % weight (0 if not held)."""
    if not selected_funds:
        return pd.DataFrame()
    all_stocks = sorted({s for f in selected_funds for s in FUND_HOLDINGS.get(f, [])})
    data = {}
    for fund in selected_funds:
        weights = FUND_WEIGHTS.get(fund, {})
        data[fund] = [weights.get(stock, 0.0) for stock in all_stocks]
    return pd.DataFrame(data, index=all_stocks)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#F0F4FF"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(201,168,76,0.2)", borderwidth=1),
)


def chart_wealth_journey(fire: dict) -> go.Figure:
    years        = fire["years"]
    corpus       = fire["corpus_journey"]
    fire_line    = [fire["fire_corpus"]] * len(years)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=corpus, mode="lines",
        line=dict(color="#C9A84C", width=3),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)",
        name="Projected Corpus",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=fire_line, mode="lines",
        line=dict(color="#DA250A", width=2, dash="dash"),
        name="FIRE Target",
    ))
    # Mark retirement year
    ret_year = years[-1]
    fig.add_vline(x=ret_year, line_width=1, line_dash="dot", line_color="#8A95B0")
    fig.add_annotation(x=ret_year, y=max(corpus)*0.92,
                       text=f"Retire @ {ret_year}", showarrow=False,
                       font=dict(color="#8A95B0", size=11))
    fig.update_layout(**PLOTLY_LAYOUT, title="Wealth Journey to FIRE",
                      xaxis_title="Age", yaxis_title="Corpus (₹)")
    fig.update_yaxes(tickformat=".2s")
    return fig


def chart_tax_comparison(tax: dict) -> go.Figure:
    regimes = ["Old Regime", "New Regime"]
    totals  = [tax["old_total"], tax["new_total"]]
    colors  = ["#8A95B0", "#C9A84C"]
    if tax["better"] == "New":
        colors = ["#8A95B0", "#C9A84C"]
    else:
        colors = ["#C9A84C", "#8A95B0"]

    fig = go.Figure(go.Bar(
        x=regimes, y=totals,
        marker_color=colors,
        text=[fmt_inr(t) for t in totals],
        textposition="outside",
        width=0.45,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Tax Outgo Comparison (FY 2024-25)",
                      yaxis_title="Total Tax + Cess (₹)")
    fig.update_yaxes(tickformat=".2s")
    return fig


def chart_overlap_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=[[0, "#141A27"], [0.5, "#C9A84C"], [1, "#DA250A"]],
        text=[[f"{v:.1f}%" for v in row] for row in df.values],
        texttemplate="%{text}",
        hovertemplate="%{y} in %{x}: %{z:.1f}%<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#F0F4FF")),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Stock Overlap Heatmap (% allocation)")
    return fig


def chart_overlap_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette = ["#C9A84C", "#DA250A", "#3B82F6", "#34D399", "#F472B6"]
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
            name=col, x=df.index, y=df[col],
            marker_color=palette[i % len(palette)],
            text=[f"{v:.1f}%" if v > 0 else "" for v in df[col]],
            textposition="outside",
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                      title="Per-Stock Allocation Across Selected Funds",
                      yaxis_title="% Allocation", xaxis_title="Stock")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  CREWAI MULTI-AGENT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def build_crew(api_key: str, age, income, target_retirement, selected_funds):
    llm = LLM(
        model=_crewai_gemini_model(),
        api_key=api_key,
        temperature=0.3,
    )

    # ── Agents ──────────────────────────────────────────────────────────────
    strategist = Agent(
        role="FIRE Strategist",
        goal="Calculate the exact FIRE corpus needed for early retirement using 6% inflation and 4% withdrawal rate.",
        backstory=(
            "You are a Certified Financial Planner specialising in FIRE (Financial Independence, "
            "Retire Early) planning for Indian investors. You reason step-by-step, show all arithmetic, "
            "and present final figures in INR crores."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    tax_auditor = Agent(
        role="Tax Auditor",
        goal="Compare Old vs New Tax Regimes for FY 2024-25 and recommend the optimal regime.",
        backstory=(
            "You are a senior Chartered Accountant with deep expertise in Indian income tax. "
            "You walk through every slab, deduction (80C, standard deduction), and cess meticulously, "
            "citing the exact Finance Act provisions."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    xray_analyst = Agent(
        role="Portfolio X-Ray Analyst",
        goal="Identify and quantify stock overlap risks across mutual fund holdings.",
        backstory=(
            "You are a quantitative analyst at a top Indian asset management firm. "
            "You audit portfolio concentration, flag hidden single-stock exposure, "
            "and recommend diversification actions with clear risk scores."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ── Tasks ──────────────────────────────────────────────────────────────
    fire_data = compute_fire(age, income, target_retirement)
    tax_data  = compute_taxes(income)
    overlap_df = compute_overlap(selected_funds)

    task_fire = Task(
        description=(
            f"The investor is {age} years old, earns ₹{income:,.0f}/year, "
            f"and wants to retire at {target_retirement}. "
            f"Using 6% inflation and 4% withdrawal rate, calculate:\n"
            f"1. Inflation-adjusted annual expenses at retirement\n"
            f"2. Required FIRE corpus\n"
            f"3. Projected corpus (assuming 30% savings rate, 12% CAGR)\n"
            f"4. Gap analysis and monthly SIP needed to bridge any gap.\n"
            f"Pre-computed numbers for verification: {fire_data}"
        ),
        agent=strategist,
        expected_output="A structured FIRE report with all calculations shown step-by-step in INR.",
    )

    task_tax = Task(
        description=(
            f"Gross annual income: ₹{income:,.0f}.\n"
            f"Compare Old Regime (standard deduction ₹50,000 + Section 80C ₹1,50,000) vs "
            f"New Regime (standard deduction ₹75,000, no 80C) for FY 2024-25.\n"
            f"Show every slab calculation, health & education cess (4%), and final net savings.\n"
            f"Pre-computed summary: {tax_data}"
        ),
        agent=tax_auditor,
        expected_output="A step-by-step tax comparison table and a clear recommendation.",
    )

    fund_str = ", ".join(selected_funds) if selected_funds else "No funds selected"
    overlap_str = overlap_df.to_string() if not overlap_df.empty else "No overlap data."
    task_xray = Task(
        description=(
            f"Selected funds: {fund_str}.\n"
            f"Overlap matrix:\n{overlap_str}\n"
            f"Identify stocks with >10% combined exposure, explain concentration risk, "
            f"and suggest 2–3 alternative funds to improve diversification."
        ),
        agent=xray_analyst,
        expected_output="A risk report highlighting overlapping stocks and actionable recommendations.",
    )

    crew = Crew(
        agents=[strategist, tax_auditor, xray_analyst],
        tasks=[task_fire, task_tax, task_xray],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def run_agents(api_key: str, age, income, target_retirement, selected_funds) -> tuple[str, str]:
    """Run CrewAI and capture verbose reasoning. Returns (result_text, log_text)."""
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        try:
            crew   = build_crew(api_key, age, income, target_retirement, selected_funds)
            result = crew.kickoff()
            output = str(result)
        except Exception as exc:
            output = f"❌ Agent error: {exc}"
    log = log_buffer.getvalue()
    return output, log


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="brand-bar">
        <span class="brand-badge">ET</span>
        <span class="brand-name">Artha<span>AI</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Investor Profile")

    st.session_state["age"] = st.slider(
        "Current Age", 20, 60, st.session_state["age"], 1
    )
    st.session_state["income"] = st.slider(
        "Annual Income (₹)", 300_000, 10_000_000,
        st.session_state["income"], 50_000,
        format="₹%d"
    )
    st.session_state["target_retirement"] = st.slider(
        "Target Retirement Age", st.session_state["age"] + 1, 75,
        max(st.session_state["target_retirement"], st.session_state["age"] + 1), 1
    )

    st.markdown("<div class='et-divider'></div>", unsafe_allow_html=True)

    run_clicked = st.button("🚀 Run Analysis", width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:0.72rem;color:#4B5563;'>"
        "ET ArthaAI v1.0 · Hackathon Edition<br>"
        "Powered by CrewAI + Gemini 1.5 Flash"
        "</span>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE COMPUTED VALUES  (instant on slider change)
# ═══════════════════════════════════════════════════════════════════════════════

age               = st.session_state["age"]
income            = st.session_state["income"]
target_retirement = st.session_state["target_retirement"]
api_key           = (os.getenv("GOOGLE_API_KEY") or "").strip()
selected_funds    = st.session_state["selected_funds"]

fire = compute_fire(age, income, target_retirement)
tax  = compute_taxes(income)

# ── Run agents when button clicked ──────────────────────────────────────────
if run_clicked:
    if not api_key:
        env_path = _PROJECT_ROOT / ".env"
        st.error(
            "⚠️ **GOOGLE_API_KEY** is missing or empty. "
            f"Open `{env_path}` and set a line like `GOOGLE_API_KEY=AIza...` (no quotes), "
            "save the file, then restart Streamlit."
        )
    elif not CREWAI_AVAILABLE:
        st.error("⚠️ `crewai` is not installed. Run: pip install crewai")
    else:
        with st.spinner("🤖 Multi-agent system activating…"):
            output, log = run_agents(
                api_key, age, income, target_retirement, selected_funds
            )
        st.session_state["agent_output"]  = output
        st.session_state["reasoning_log"] = log
        st.session_state["agents_ran"]    = True

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    <h1 style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
               margin:0;color:#F0F4FF;">
        Autonomous Money Mentor
    </h1>
    <span style="background:#DA250A;color:#fff;padding:4px 12px;border-radius:6px;
                 font-family:'Share Tech Mono',monospace;font-size:0.82rem;">LIVE</span>
</div>
<p style="color:#8A95B0;margin-top:2px;margin-bottom:24px;font-size:0.95rem;">
    AI-powered wealth intelligence · Drag sliders for instant previews · Click 🚀 for deep analysis
</p>
""", unsafe_allow_html=True)

# ── KPI Strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, value, sub="", good=True):
    color = "#C9A84C" if good else "#DA250A"
    col.markdown(f"""
    <div class="metric-tile">
        <p class="metric-label">{label}</p>
        <p class="metric-value" style="color:{color};">{value}</p>
        <p style="font-size:0.74rem;color:#8A95B0;margin:4px 0 0 0;">{sub}</p>
    </div>
    """, unsafe_allow_html=True)

kpi(k1, "FIRE Corpus Needed",  fmt_inr(fire["fire_corpus"]))
kpi(k2, "Projected Corpus",    fmt_inr(fire["projected_corpus"]),
    sub="@ 12% CAGR, 30% savings", good=fire["on_track"])
kpi(k3, "Corpus Gap",          fmt_inr(fire["gap"]),
    sub="Bridge with SIP", good=fire["gap"] == 0)
kpi(k4, "Best Tax Regime",     tax["better"] + " Regime",
    sub=f"Saves {fmt_inr(abs(tax['savings']))}", good=True)

st.markdown("<div class='et-divider'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📈 Wealth Journey", "🧾 Tax Wizard", "🔍 Portfolio X-Ray"])

# ── TAB 1 : WEALTH JOURNEY ────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="artha-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_wealth_journey(fire), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="artha-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 FIRE Summary")
        items = [
            ("Years to Retire", str(fire["years_to_retire"])),
            ("Future Annual Expenses", fmt_inr(fire["future_expenses"])),
            ("Required Corpus", fmt_inr(fire["fire_corpus"])),
            ("Projected Corpus", fmt_inr(fire["projected_corpus"])),
            ("Gap", fmt_inr(fire["gap"])),
        ]
        for label, val in items:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"border-bottom:1px solid rgba(201,168,76,0.1);padding:8px 0;'>"
                f"<span style='color:#8A95B0;font-size:0.85rem;'>{label}</span>"
                f"<span style='color:#C9A84C;font-weight:600;'>{val}</span></div>",
                unsafe_allow_html=True,
            )
        status = "✅ ON TRACK" if fire["on_track"] else "⚠️ NEEDS ATTENTION"
        color  = "#C9A84C" if fire["on_track"] else "#DA250A"
        st.markdown(
            f"<br><p style='text-align:center;font-family:Syne,sans-serif;"
            f"font-size:1.1rem;font-weight:800;color:{color};'>{status}</p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Monthly SIP estimate to close gap
        if fire["gap"] > 0 and fire["years_to_retire"] > 0:
            monthly_sip = (fire["gap"] * EXPECTED_RETURNS / 12) / \
                          ((1 + EXPECTED_RETURNS / 12) ** (fire["years_to_retire"] * 12) - 1)
            st.markdown('<div class="artha-card">', unsafe_allow_html=True)
            st.markdown("#### 📌 Gap-Closing SIP")
            st.markdown(
                f"<p style='color:#DA250A;font-family:Syne,sans-serif;"
                f"font-size:1.6rem;font-weight:800;margin:0;'>{fmt_inr(monthly_sip)}<span "
                f"style='font-size:0.9rem;'>/month</span></p>"
                f"<p style='color:#8A95B0;font-size:0.8rem;'>At {EXPECTED_RETURNS*100:.0f}% CAGR</p>",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Agent output for this tab
    if st.session_state["agents_ran"] and st.session_state["agent_output"]:
        st.markdown('<div class="artha-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Strategist Agent Report")
        st.markdown(st.session_state["agent_output"])
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2 : TAX WIZARD ───────────────────────────────────────────────────────
with tab2:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="artha-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_tax_comparison(tax), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="artha-card">', unsafe_allow_html=True)
        st.markdown("#### 🧾 Detailed Tax Breakdown")

        for regime, data in [
            ("Old Regime", {
                "Gross Income":       income,
                "Deductions (80C+SD)": -(150_000 + 50_000),
                "Taxable Income":     tax["old_taxable"],
                "Base Tax":           tax["old_tax"],
                "Cess (4%)":          tax["old_cess"],
                "Total Tax":          tax["old_total"],
            }),
            ("New Regime", {
                "Gross Income":       income,
                "Std Deduction":      -75_000,
                "Taxable Income":     tax["new_taxable"],
                "Base Tax":           tax["new_tax"],
                "Cess (4%)":          tax["new_cess"],
                "Total Tax":          tax["new_total"],
            }),
        ]:
            col_color = "#C9A84C" if regime[0] == tax["better"][0] else "#8A95B0"
            st.markdown(
                f"<p style='font-family:Syne,sans-serif;font-weight:700;"
                f"color:{col_color};margin:14px 0 6px 0;'>{regime}</p>",
                unsafe_allow_html=True,
            )
            for lbl, val in data.items():
                vstr = fmt_inr(abs(val)) if val >= 0 else f"- {fmt_inr(abs(val))}"
                bold = "font-weight:700;" if lbl == "Total Tax" else ""
                bc   = "#DA250A" if lbl == "Total Tax" else "transparent"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"background:{bc};padding:5px 8px;border-radius:6px;"
                    f"border-bottom:1px solid rgba(201,168,76,0.08);{bold}'>"
                    f"<span style='color:#8A95B0;font-size:0.83rem;'>{lbl}</span>"
                    f"<span style='color:#F0F4FF;'>{vstr}</span></div>",
                    unsafe_allow_html=True,
                )

        saving_color = "#C9A84C" if tax["savings"] > 0 else "#DA250A"
        saving_label = "New saves over Old" if tax["savings"] > 0 else "Old saves over New"
        st.markdown(
            f"<br><p style='text-align:center;'>"
            f"<span style='font-family:Syne,sans-serif;font-size:1.1rem;"
            f"font-weight:800;color:{saving_color};'>"
            f"💡 {saving_label}: {fmt_inr(abs(tax['savings']))}</span></p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3 : PORTFOLIO X-RAY ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="artha-card artha-card-active">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Select Mutual Funds to Analyse Overlap")

    chosen = st.multiselect(
        "Choose 2–3 funds:",
        options=list(FUND_HOLDINGS.keys()),
        default=st.session_state["selected_funds"] or [],
        key="fund_multiselect",
    )
    st.session_state["selected_funds"] = chosen
    st.markdown('</div>', unsafe_allow_html=True)

    if len(chosen) < 2:
        st.markdown(
            "<div class='artha-card'>"
            "<p style='color:#8A95B0;text-align:center;padding:24px 0;'>"
            "Select at least 2 funds to detect overlap.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        overlap_df = compute_overlap(chosen)
        # Determine overlapping stocks (appear in >1 fund with weight > 0)
        overlap_stocks = [s for s in overlap_df.index if (overlap_df.loc[s] > 0).sum() > 1]

        if overlap_stocks:
            st.markdown(
                f"<div class='artha-card artha-card-active'>"
                f"<p style='font-family:Syne,sans-serif;font-size:1rem;"
                f"color:#DA250A;font-weight:700;'>⚠️ Overlapping Stocks Detected: "
                f"{', '.join(overlap_stocks)}</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='artha-card'>"
                "<p style='color:#C9A84C;font-weight:700;'>✅ No stock overlap detected across selected funds.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="artha-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_overlap_heatmap(overlap_df), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="artha-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_overlap_bar(overlap_df), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # Risk table
        st.markdown('<div class="artha-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Exposure Table")
        display_df = overlap_df.copy()
        display_df["Combined Exposure %"] = display_df.sum(axis=1)
        display_df = display_df.sort_values("Combined Exposure %", ascending=False)
        display_df = display_df.map(lambda x: f"{x:.1f}%" if x > 0 else "—")
        st.dataframe(display_df, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT AUDIT TRAIL (always visible, populated after run)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='et-divider'></div>", unsafe_allow_html=True)

with st.expander("🕵️ Agent Audit Trail", expanded=st.session_state["agents_ran"]):
    if not st.session_state["agents_ran"]:
        st.markdown(
            "<p style='color:#8A95B0;font-size:0.9rem;'>"
            "Click <strong>🚀 Run Analysis</strong> in the sidebar to activate the multi-agent system. "
            "Full reasoning logs will appear here.</p>",
            unsafe_allow_html=True,
        )
    else:
        agent_col, log_col = st.columns([1, 1], gap="large")

        with agent_col:
            st.markdown("##### 📝 Agent Outputs")
            st.markdown(
                f'<div class="artha-card">{st.session_state["agent_output"]}</div>',
                unsafe_allow_html=True,
            )

        with log_col:
            st.markdown("##### 🖥️ Reasoning Log (Terminal)")
            log_text = st.session_state["reasoning_log"] or "No verbose log captured."
            # Sanitise for HTML display
            escaped = (log_text
                       .replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))
            st.markdown(
                f'<div class="terminal-box">{escaped}</div>',
                unsafe_allow_html=True,
            )

        # Agent status pills
        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        for col, name, icon in [
            (p1, "FIRE Strategist",    "📈"),
            (p2, "Tax Auditor",        "🧾"),
            (p3, "X-Ray Analyst",      "🔍"),
        ]:
            col.markdown(
                f"<div class='artha-card' style='text-align:center;padding:16px;'>"
                f"<span style='font-size:1.5rem;'>{icon}</span><br>"
                f"<span style='font-family:Syne,sans-serif;font-weight:700;'>{name}</span>"
                f"<span class='agent-pill pill-done'>COMPLETED</span></div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;margin-top:48px;padding:20px 0;
            border-top:1px solid rgba(201,168,76,0.12);">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#4B5563;">
        ET ArthaAI · Autonomous Money Mentor · Hackathon Edition 2025
        &nbsp;|&nbsp; Powered by CrewAI + Gemini 1.5 Flash + Streamlit
        &nbsp;|&nbsp; Not financial advice.
    </span>
</div>
""", unsafe_allow_html=True)
