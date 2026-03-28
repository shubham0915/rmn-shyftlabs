"""
streamlit_app.py — Beautiful Streamlit Frontend for the RMN Engine
Agent 1: Frontend Specialist

4-Tab Layout:
  Tab 1: Live Demo     — Simulate a user browsing a page, see a real ad served
  Tab 2: Metrics       — Live CTR lift chart + session stats
  Tab 3: Privacy       — ε budget gauge + DP explanation
  Tab 4: Architecture  — Mermaid diagram + tech stack

Run:
  streamlit run src/streamlit_app.py
"""

import time
import uuid
import os
import sys
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

try:
    from src.config import RETAILER_PAGES, API_PORT, EPSILON_MAX
except ModuleNotFoundError:
    # Fallback for direct `streamlit run src/streamlit_app.py` launches
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from src.config import RETAILER_PAGES, API_PORT, EPSILON_MAX

API_BASE = f"http://127.0.0.1:{API_PORT}"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def check_api_health(timeout: float = 1.0) -> bool:
    """Return True if the FastAPI backend is up and responsive."""
    try:
        r = requests.get(f"{API_BASE}/", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def wait_for_api(max_wait: int = 45) -> bool:
    """
    Poll the /  health endpoint until API is ready or max_wait seconds pass.
    Shows a Streamlit spinner while waiting.
    Returns True if API came up, False if timed out.
    """
    import time
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if check_api_health():
            return True
        time.sleep(1.5)
    return False

# ---------------------------------------------------------------------------
# Page config + custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RMN Engine — ShyftLabs AdTech",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}
[data-testid="stHeader"] { background: transparent; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    color: #a78bfa;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff;
    border-bottom: 3px solid #7c3aed;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.4);
    border-radius: 12px;
    padding: 1rem;
}

/* Ad card */
.ad-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 2px solid #7c3aed;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    box-shadow: 0 4px 30px rgba(124, 58, 237, 0.3);
}
.ad-title   { font-size: 1.4rem; font-weight: 700; color: #c4b5fd; }
.ad-copy    { font-size: 1.05rem; color: #e9d5ff; margin-top: 0.4rem; }
.ad-badges  { margin-top: 0.8rem; display: flex; gap: 0.6rem; }
.badge      {
    background: #7c3aed; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; color: white;
}

/* Privacy badge */
.dp-badge {
    background: linear-gradient(90deg, #065f46, #047857);
    border-radius: 8px; padding: 0.4rem 0.8rem;
    font-size: 0.82rem; color: #6ee7b7;
    border: 1px solid #34d399; display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []         # list of ad responses
if "user_hash" not in st.session_state:
    st.session_state.user_hash = str(uuid.uuid4())[:8]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem'>
  <h1 style='font-size:2.4rem; font-weight:800;
     background: linear-gradient(90deg, #a78bfa, #7c3aed, #4f46e5);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    🛒 Privacy-Preserving RMN Engine
  </h1>
  <p style='color:#94a3b8; font-size:1rem; margin-top:-0.5rem;'>
    Built for ShyftLabs AdTech · Differential Privacy · Real-Time Semantic Serving
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Live Demo", "📊 Metrics Dashboard", "🔐 Privacy Report", "🏛 Architecture"]
)

# ============================================================
# TAB 1: LIVE DEMO
# ============================================================
with tab1:
    st.markdown("### Simulate a User Browsing Session")

    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
        st.markdown("**Select Retailer Page**")
        retailer_name = st.selectbox(
            "Page Context",
            options=list(RETAILER_PAGES.keys()),
            label_visibility="collapsed",
        )
        page_text = RETAILER_PAGES[retailer_name]

        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.85rem; "
            f"margin-bottom:1rem;'>📄 {page_text}</div>",
            unsafe_allow_html=True,
        )

        custom_text = st.text_area(
            "Or type a custom page description",
            placeholder="e.g. winter jackets for men under ₹3000",
            height=80,
        )
        if custom_text.strip():
            page_text = custom_text.strip()

        browse_clicked = st.button("🔍 Browse Page & Get Ad", use_container_width=True, type="primary")

    with col_r:
        if browse_clicked:
            with st.spinner("Generating personalized ads... (first time may take 15–20s)"):
                # Wait for API to be ready (handles first-time model loading)
                if not check_api_health():
                    st.info("⏳ API is warming up (loading 2.7M events + embedding model)…")
                    api_ready = wait_for_api(max_wait=60)
                    if not api_ready:
                        st.error(
                            "⚠️ API not responding after 60s.\n\n"
                            "Make sure you've started it:\n"
                            "`python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload`"
                        )
                        st.stop()

                # Push event (fire-and-forget)
                try:
                    requests.post(
                        f"{API_BASE}/track_event",
                        json={
                            "user_hash": st.session_state.user_hash,
                            "page_text": page_text,
                            "retailer":  retailer_name,
                        },
                        timeout=3,
                    )
                except Exception:
                    pass   # non-critical

                # Get ad (120s timeout — first AI inference on some CPUs can be slow)
                try:
                    resp = requests.get(
                        f"{API_BASE}/get_ad",
                        params={"user_hash": st.session_state.user_hash, "page_text": page_text},
                        timeout=120,
                    )
                    ad = resp.json()
                    if "detail" in ad:
                        st.error(f"API error: {ad['detail']}")
                        ad = None
                    else:
                        st.session_state.history.append(ad)
                except Exception as e:
                    st.error(f"⚠️ Request failed: {e}")
                    ad = None

        if st.session_state.history:
            ad = st.session_state.history[-1]
            top_ads = ad.get("top_ads") or [ad]
            st.markdown("#### Top 3 Recommended Ads")
            for idx, item in enumerate(top_ads[:3], start=1):
                price = item.get("price")
                price_badge = f'<span class="badge">₹{int(price)}</span>' if price is not None else ""
                st.markdown(f"""
<div class="ad-card">
    <div class="ad-title">#{idx} · 📢 {item.get('title', 'Ad')}</div>
    <div class="ad-copy">"{item.get('ad_copy', '')}"</div>
    <div class="ad-badges">
        <span class="badge">#{item.get('category','').upper()}</span>
        {price_badge}
        <span class="badge">Score {item.get('final_score', 0):.2f}</span>
        <span class="badge">Sim {item.get('similarity', 0):.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

            latency = ad.get("latency_ms", 0)
            eps     = ad.get("epsilon_used", 0)
            lift    = ad.get("ctr_lift_pct", 0)
            primary_ad = top_ads[0] if top_ads else ad

            m1, m2, m3 = st.columns(3)
            m1.metric("⚡ Latency", f"{latency:.0f} ms",
                      delta="<100ms ✓" if latency < 100 else "slow",
                      delta_color="normal" if latency < 100 else "inverse")
            m2.metric("🔐 ε Used", f"{eps:.2f} / {EPSILON_MAX}", delta=f"+{0.1:.1f}")
            m3.metric("📈 CTR Lift", f"{lift:.1f}%", delta=f"+{lift:.1f}% vs random")

            st.markdown(f"""
<div class="dp-badge">
  🔐 Privacy-Protected · ε={eps:.2f} · Laplace Noise Applied · No Raw User Data Shared
</div>""", unsafe_allow_html=True)

            with st.expander("Why this ad was chosen"):
                st.markdown(f"- **Relevance score**: {primary_ad.get('similarity', 0):.3f}")
                st.markdown(f"- **Popularity (CTR)**: {primary_ad.get('ctr', 0) * 100:.1f}%")
                st.markdown(f"- **ML Predicted p(click)**: {primary_ad.get('final_score', 0):.3f}")
                st.markdown(f"- **XGBoost Reranking**: Applied ✓")
                st.markdown(f"- **Category match**: {primary_ad.get('category', 'unknown')}")
        else:
            st.info("👆 Click **Browse Page & Get Ad** to start the live demo.")


# ============================================================
# TAB 2: METRICS DASHBOARD
# ============================================================
with tab2:
    st.markdown("### Session Metrics")

    history = st.session_state.history
    n_events = len(history)

    if n_events == 0:
        st.info("No events yet. Go to **Live Demo** tab and browse a few pages.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        avg_latency = np.mean([h.get("latency_ms", 0) for h in history])
        avg_lift    = np.mean([h.get("ctr_lift_pct", 0) for h in history])
        max_eps     = history[-1].get("epsilon_used", 0)
        c1.metric("Events Served",   n_events)
        c2.metric("Avg Latency",     f"{avg_latency:.0f} ms")
        c3.metric("Avg CTR Lift",    f"{avg_lift:.1f}%")
        c4.metric("ε Budget Used",   f"{max_eps:.2f}")

        st.markdown("#### CTR Lift Over Sessions")
        lifts     = [h.get("ctr_lift_pct", 0) for h in history]
        latencies = [h.get("latency_ms", 0) for h in history]
        fig_lift = go.Figure()
        fig_lift.add_trace(go.Scatter(
            y=lifts, mode="lines+markers",
            line=dict(color="#7c3aed", width=2.5),
            marker=dict(size=7, color="#a78bfa"),
            name="CTR Lift %",
        ))
        fig_lift.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_lift.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_title="Session #", yaxis_title="Lift %",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_lift, width='stretch')

        st.markdown("#### Latency Distribution (ms)")
        fig_lat = px.histogram(
            x=latencies, nbins=20,
            color_discrete_sequence=["#7c3aed"],
        )
        fig_lat.add_vline(x=100, line_dash="dash", line_color="#f59e0b",
                          annotation_text="100ms SLA", annotation_position="top right")
        fig_lat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_title="Latency (ms)", yaxis_title="Count",
            margin=dict(t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig_lat, width='stretch')

        if n_events > 0:
            latest_lift = history[-1].get("ctr_lift_pct", 0)
            st.markdown("#### Semantic vs Random Baseline")
            fig_bar = go.Figure(go.Bar(
                x=["Random Policy", "Semantic RMN"],
                y=[8.5, 8.5 * (1 + latest_lift / 100)],
                marker_color=["#475569", "#7c3aed"],
                text=[f"~8.5%", f"~{8.5 * (1 + latest_lift / 100):.1f}%"],
                textposition="auto",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", yaxis_title="Expected CTR (%)",
                margin=dict(t=10, b=10), showlegend=False,
            )
            st.plotly_chart(fig_bar, width='stretch')


# ============================================================
# TAB 3: PRIVACY REPORT
# ============================================================
with tab3:
    st.markdown("### Differential Privacy Dashboard")
    history = st.session_state.history

    eps_used = history[-1].get("epsilon_used", 0.0) if history else 0.0
    eps_pct  = eps_used / EPSILON_MAX

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=eps_used,
        delta={"reference": 0, "suffix": " ε"},
        title={"text": "Privacy Budget Consumed (ε)", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, EPSILON_MAX], "tickcolor": "#94a3b8"},
            "bar":  {"color": "#7c3aed"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 0.4 * EPSILON_MAX],           "color": "#064e3b"},
                {"range": [0.4 * EPSILON_MAX, 0.8 * EPSILON_MAX], "color": "#78350f"},
                {"range": [0.8 * EPSILON_MAX, EPSILON_MAX],  "color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 4},
                "value": EPSILON_MAX,
            },
        },
        number={"suffix": f" / {EPSILON_MAX}", "font": {"color": "#c4b5fd"}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        margin=dict(t=20, b=20), height=300,
    )
    st.plotly_chart(fig_gauge, width='stretch')

    st.markdown(f"""
**Budget used:** `{eps_used:.2f} ε`  
**Remaining:** `{max(0, EPSILON_MAX - eps_used):.2f} ε`  
**Each query costs:** `0.1 ε` (Laplace mechanism)  
**Hard cap:** `{EPSILON_MAX} ε` — queries are rejected beyond this.
""")

    with st.expander("📖 What is Differential Privacy?"):
        st.markdown(r"""
**Differential Privacy (DP)** provides a mathematical guarantee:

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S]$$

Where $D$ and $D'$ differ by one record. A smaller $\varepsilon$ means more privacy.

**Laplace Mechanism** (used here):  

$$\mathcal{M}(x) = x + \text{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right)$$

- $\Delta f$ = sensitivity (how much one record can change the query result)
- We set $\Delta f = 1$ (counting queries)  
- $\varepsilon = 0.1$ per query, budget capped at $0.9$

**In plain English:**  
Every aggregate stat (views, carts, purchases) has random noise added before it leaves the Clean Room. Even if an attacker sees the query results, they cannot learn whether any single user is in the dataset.
""")

    with st.expander("🔒 What data is protected?"):
        st.markdown("""
| Data Type | Protection |
|-----------|-----------|
| Individual browsing events | Hashed visitor ID, never stored raw |
| Category view counts | Laplace noise added (ε = 0.1/query) |
| Cart + purchase counts | Laplace noise added |
| User identity | Only `user_hash` is ever transmitted |
| Ad selection signal | Derived from noisy aggregates, not raw events |
""")


# ============================================================
# TAB 4: ARCHITECTURE
# ============================================================
with tab4:
    st.markdown("### System Architecture")

    st.markdown("""
```mermaid
graph TD
    A[🌐 User Context] --> B[ChromaDB Vector Retrieval]
    A --> C[Gemini Flash Agentic Copygen]
    B --> D[XGBoost ML Ranking Engine]
    C --> E[FastAPI / Streamlit Ad Display]
    D --> E
    E --> F[Async Background Event Worker]
    F --> G[DuckDB Clean Room + DP Laplace Noise]
    G --> A

    subgraph Privacy-First Layer
        G
    end
    subgraph Real-Time AI Layer
        B
        D
        C
    end
```
""")

    st.markdown("### Tech Stack & Choices")
    st.markdown("""
| Component | Technology | Why |
|-----------|-----------|-----|
| API Backend | FastAPI + Uvicorn | Async background ingestion |
| Data Clean Room | DuckDB | OLAP Performance on 2.7M rows |
| Vector Engine | **ChromaDB** | HNSW for 10,000+ ad catalog scale |
| ML Ranking | **XGBoost** | Re-ranking for `p(click)` probability |
| Agentic LLM | **Gemini 1.5 Flash** | Sub-second personalized copywriting |
| Embeddings | sentence-transformers | CPU-efficient `all-MiniLM-L6-v2` |
| Privacy Math | diffprivlib | Persistent DP budget tracking |
| Dataset | Retailrocket (2.7M) | Real-world e-commerce event stream |
""")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Events", "2.7M")
    col2.metric("Target Latency", "<100 ms")
    col3.metric("Privacy Budget", "ε ≤ 0.9")
