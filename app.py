import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go
import time

from failurenet_pipeline import predict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FailureNet",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark theme ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0d14;
    color: #c8d6e5;
}

/* ── Streamlit main container ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Header ── */
.fn-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.25rem;
}
.fn-logo {
    width: 44px; height: 44px;
    border-radius: 10px;
    background: linear-gradient(135deg, #00e5ff 0%, #0072ff 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 18px #00e5ff55;
}
.fn-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #00e5ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.fn-subtitle {
    font-size: 0.95rem;
    color: #5a7a9a;
    letter-spacing: 1px;
    margin-bottom: 1.5rem;
}

/* ── Upload zone ── */
.stFileUploader > div {
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    background: #0d1520 !important;
    transition: border-color 0.3s;
}
.stFileUploader > div:hover {
    border-color: #00e5ff !important;
}

/* ── Metric card ── */
.metric-card {
    background: #0d1929;
    border: 1px solid #1a2e45;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00e5ff, #0072ff);
    opacity: 0.7;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a7a9b;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e0f0ff;
}

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00e5ff, #0072ff) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #1a2e45 !important;
    border-radius: 4px !important;
    height: 8px !important;
}

/* ── Accept / Reject banner ── */
.verdict-accept {
    background: linear-gradient(135deg, #0a2a1f, #0d3526);
    border: 1px solid #00c97a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #00e68a;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 0 20px #00c97a22;
    animation: pulse-green 2s ease-in-out infinite;
}
.verdict-reject {
    background: linear-gradient(135deg, #2a0a0a, #350d0d);
    border: 1px solid #ff3b3b;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #ff6b6b;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 0 20px #ff3b3b22;
    animation: pulse-red 2s ease-in-out infinite;
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px #00c97a22; }
    50%       { box-shadow: 0 0 32px #00c97a55; }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px #ff3b3b22; }
    50%       { box-shadow: 0 0 32px #ff3b3b55; }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #080c12 !important;
    border-right: 1px solid #1a2e45;
}
section[data-testid="stSidebar"] * {
    color: #8aa8c8 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00e5ff !important;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
}

/* ── Divider ── */
hr {
    border-color: #1a2e45 !important;
    margin: 1.25rem 0 !important;
}

/* ── Info box ── */
.stInfo {
    background: #0a1929 !important;
    border-left: 3px solid #0072ff !important;
    color: #8aa8c8 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #00e5ff !important;
}

/* ── History table ── */
.history-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #1a2e45;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #7a9ab5;
}
.history-row:last-child { border-bottom: none; }
.history-row span.accept { color: #00e68a; }
.history-row span.reject { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)


# ── Session state for history ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99, value=0.70, step=0.01,
        help="Minimum confidence required to ACCEPT a prediction."
    )
    entropy_threshold = st.slider(
        "Max entropy",
        min_value=0.1, max_value=3.0, value=1.5, step=0.1,
        help="Predictions above this entropy are flagged as uncertain."
    )
    resize_dim = st.selectbox(
        "Input resize (px)",
        options=[32, 64, 128, 224],
        index=0,
        help="Resize uploaded image before inference."
    )

    st.markdown("---")
    st.markdown("### 📋 Session History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-8:]):
            verdict_cls = "accept" if h["decision"] == "ACCEPT" else "reject"
            st.markdown(
                f'<div class="history-row">'
                f'<span>{h["class"]}</span>'
                f'<span>{h["confidence"]:.2f}</span>'
                f'<span class="{verdict_cls}">{h["decision"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No predictions yet.")

    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fn-header">
    <div class="fn-logo">🧠</div>
    <p class="fn-title">FAILURENET</p>
</div>
<p class="fn-subtitle">RELIABLE AI CLASSIFIER · UNCERTAINTY ESTIMATION</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([2, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop an image to analyse",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

with col_preview:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=200, caption="Uploaded image")


# ── Inference ────────────────────────────────────────────────────────────────
if uploaded_file:
    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)

    with st.spinner("Running inference …"):
        time.sleep(0.4)   # brief pause so spinner is visible
        result = predict(tensor)

    # Append to session history
    st.session_state.history.append(result)

    st.markdown("---")
    st.markdown("#### 🔬 Prediction Results")

    # ── Top metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Class</div>
            <div class="metric-value">{result["class"]}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{result["confidence"]:.3f}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Entropy</div>
            <div class="metric-value">{result["entropy"]:.3f}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Failure Prob.</div>
            <div class="metric-value">{result["failure_probability"]:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Progress bars ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pb1, pb2 = st.columns(2, gap="large")

    with pb1:
        st.caption("CONFIDENCE")
        st.progress(min(result["confidence"], 1.0))

        st.caption("FAILURE PROBABILITY")
        st.progress(min(result["failure_probability"], 1.0))

    with pb2:
        st.caption("ENTROPY  (normalised, max = 3.0)")
        st.progress(min(result["entropy"] / 3.0, 1.0))

        st.caption("RELIABILITY  (1 − failure prob.)")
        reliability = max(0.0, 1.0 - result["failure_probability"])
        st.progress(reliability)

    # ── Gauge chart ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    gauge_col, verdict_col = st.columns([1, 1], gap="large")

    with gauge_col:
        st.markdown("##### Confidence Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(result["confidence"] * 100, 1),
            number={"suffix": "%", "font": {"color": "#e0f0ff", "family": "Share Tech Mono", "size": 28}},
            delta={
                "reference": confidence_threshold * 100,
                "increasing": {"color": "#00e68a"},
                "decreasing": {"color": "#ff6b6b"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#1a2e45",
                    "tickfont": {"color": "#4a7a9b", "size": 10},
                },
                "bar": {"color": "#00e5ff", "thickness": 0.25},
                "bgcolor": "#0d1929",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, confidence_threshold * 100], "color": "#1a1f2e"},
                    {"range": [confidence_threshold * 100, 100], "color": "#0d2a1f"},
                ],
                "threshold": {
                    "line": {"color": "#00e68a", "width": 3},
                    "thickness": 0.85,
                    "value": confidence_threshold * 100,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0a0d14",
            plot_bgcolor="#0a0d14",
            font={"color": "#c8d6e5", "family": "Rajdhani"},
            margin=dict(t=10, b=10, l=20, r=20),
            height=220,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Verdict ───────────────────────────────────────────────────────────────
    with verdict_col:
        st.markdown("##### Decision")
        st.markdown("<br>", unsafe_allow_html=True)

        if result["decision"] == "ACCEPT":
            st.markdown(
                '<div class="verdict-accept">'
                '✅ &nbsp; PREDICTION ACCEPTED<br>'
                '<small>Model is sufficiently confident.</small>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-reject">'
                '❌ &nbsp; PREDICTION REJECTED<br>'
                '<small>High uncertainty detected — do not trust this result.</small>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        threshold_ok   = result["confidence"] >= confidence_threshold
        entropy_ok     = result["entropy"] <= entropy_threshold

        st.caption(f"{'✅' if threshold_ok else '❌'}  Confidence ≥ {confidence_threshold:.2f} threshold")
        st.caption(f"{'✅' if entropy_ok  else '❌'}  Entropy ≤ {entropy_threshold:.1f} threshold")

else:
    st.info("Upload a PNG / JPG image above to run inference.")