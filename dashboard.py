"""
dashboard.py
------------
Network Intrusion Detection System — Real-Time Dashboard
CyberShield IDS · BCT 2315 Group 6 · Multimedia University of Kenya

Run with:
    python -m streamlit run dashboard.py
"""

import time
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter, deque

from detector import RuleEngine
from classifier import MLClassifier
from responder import Responder
from simulator import (
    generate_benign, generate_dos,
    generate_bruteforce, generate_portscan,
    add_ip_addresses,
)

# ── Constants ───────────────────────────────────────────────────────────────────
MAX_EVENTS    = 600          # rolling window
TIMELINE_BINS = 30           # buckets for the timeline spark chart
LABEL_COLORS  = {
    "BENIGN":     "#2d8a4e",
    "DoS":        "#e53e3e",
    "BruteForce": "#d4891a",
    "PortScan":   "#c8960a",
}
GENERATORS   = {
    "BENIGN":     generate_benign,
    "DoS":        generate_dos,
    "BruteForce": generate_bruteforce,
    "PortScan":   generate_portscan,
}
PROTOCOL_MAP = {"TCP": 0, "UDP": 1, "ICMP": 2}

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield IDS",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:      #0b0900;
    --bg-surface:   #131007;
    --bg-card:      #1a1500;
    --bg-card2:     #201a00;
    --gold:         #c8960a;
    --gold-bright:  #f0b90b;
    --gold-dim:     #7a5c05;
    --gold-glow:    rgba(240,185,11,0.12);
    --gold-border:  rgba(240,185,11,0.25);
    --text-primary: #f5eedc;
    --text-muted:   #8a7d5a;
    --text-dim:     #4a4030;
    --danger:       #e53e3e;
    --danger-dim:   rgba(229,62,62,0.15);
    --warning:      #d4891a;
    --success:      #2d8a4e;
    --info:         #2b6cb0;
    --font-display: 'Bebas Neue', sans-serif;
    --font-body:    'Barlow', sans-serif;
    --font-cond:    'Barlow Condensed', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

.stApp { background: var(--bg-base) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--gold-border) !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stMarkdownContainer"] { border: none !important; outline: none !important; box-shadow: none !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
h1, h2, h3 { font-family: var(--font-display) !important; letter-spacing: 2px; }

/* Metric overrides */
[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 2.4rem !important;
    color: var(--gold-bright) !important;
    line-height: 1 !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-cond) !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* Buttons */
.stButton > button {
    background: var(--gold) !important;
    color: #000 !important;
    font-family: var(--font-cond) !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 10px 24px !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--gold-bright) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(240,185,11,0.35) !important;
}

/* Danger button variant */
button[data-testid="clear_btn"] {
    background: rgba(229,62,62,0.15) !important;
    color: #e88888 !important;
    border: 1px solid rgba(229,62,62,0.3) !important;
}
button[data-testid="clear_btn"]:hover {
    background: rgba(229,62,62,0.25) !important;
    box-shadow: 0 4px 14px rgba(229,62,62,0.2) !important;
}

/* Sliders */
[data-testid="stSlider"] label {
    font-family: var(--font-cond) !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--gold-border) !important;
    color: var(--text-primary) !important;
    border-radius: 2px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--gold-border) !important;
    border-radius: 0 !important;
}

hr { border-color: var(--gold-border) !important; margin: 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(240,185,11,0.6); }
    50%       { opacity: 0.6; box-shadow: 0 0 0 6px rgba(240,185,11,0); }
}
@keyframes pulse-red {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(229,62,62,0.7); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(229,62,62,0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "engine":           RuleEngine(),
        "responder":        Responder(),
        "classifier":       None,
        "ml_loaded":        False,
        "events":           [],
        # deque of (timestamp_str, label) for timeline spark chart
        "timeline":         deque(maxlen=TIMELINE_BINS * 10),
        "running":          False,
        "total_processed":  0,
        "threat_counts":    Counter(),
        "response_times":   [],
        "last_threat":      None,
        "peak_threat_rate": 0,   # track max threats seen in a single batch
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["ml_loaded"]:
        try:
            clf = MLClassifier()
            clf.load()
            st.session_state["classifier"] = clf
            st.session_state["ml_loaded"]  = True
        except Exception:
            pass


# ── Core logic ──────────────────────────────────────────────────────────────────
def generate_traffic_batch(n: int = 5) -> list[dict]:
    weights = [0.50, 0.20, 0.15, 0.15]
    labels  = random.choices(list(GENERATORS.keys()), weights=weights, k=n)
    records = []
    for label in labels:
        df  = GENERATORS[label](1)
        df  = add_ip_addresses(df)
        df["protocol_num"] = df["protocol"].map(PROTOCOL_MAP).fillna(0).astype(int)
        row = df.iloc[0].to_dict()
        row["_true_label"] = label
        records.append(row)
    return records


def run_detection(record: dict) -> dict:
    engine      = st.session_state["engine"]
    responder   = st.session_state["responder"]
    clf         = st.session_state["classifier"]

    rule_result = engine.detect(record)
    ml_result   = None
    if clf:
        try:
            ml_result = clf.predict(record)
        except Exception:
            pass

    final = rule_result
    if ml_result and ml_result["confidence"] > rule_result["confidence"]:
        final = {
            "label":       ml_result["label"],
            "confidence":  ml_result["confidence"],
            "rule":        "ML",
            "description": f"ML: {ml_result['label']} ({ml_result['confidence']:.0%})",
        }

    response = responder.respond(record, final)
    return {
        "timestamp":        datetime.now().strftime("%H:%M:%S"),
        "src_ip":           record.get("src_ip", "N/A"),
        "dst_ip":           record.get("dst_ip", "N/A"),
        "dst_port":         int(record.get("dst_port", 0)),
        "protocol":         record.get("protocol", "N/A"),
        "final_label":      final["label"],
        "confidence":       round(final["confidence"], 3),
        "rule_fired":       final.get("rule", ""),
        "response_time_ms": response["response_time_ms"],
        "ip_blocked":       response["actions"]["ip_blocked"],
    }


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _section_header(eyebrow: str, title: str, border_top: bool = True):
    border = "border-top:1px solid var(--gold-border);" if border_top else ""
    st.markdown(f"""
    <div style="padding:36px 40px 20px;{border}">
        <div style="font-family:var(--font-cond);font-size:0.68rem;letter-spacing:4px;
                     text-transform:uppercase;color:var(--gold);margin-bottom:6px;">
            ◈ {eyebrow.upper()}</div>
        <h2 style="font-family:var(--font-display);font-size:1.9rem;letter-spacing:3px;
                    margin:0;color:var(--text-primary);">{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def _base_layout(height: int = 300) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a7d5a", family="Barlow Condensed"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=height,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero():
    running  = st.session_state["running"]
    last     = st.session_state["last_threat"]
    status   = "LIVE MONITORING" if running else "SYSTEM PAUSED"
    dot_anim = "animation: pulse-dot 1.4s ease infinite;" if running else ""
    dot_clr  = "#f0b90b" if running else "#4a4030"
    now      = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # 1. Topbar — no dynamic content inside, safe as a plain string
    st.markdown(f"""
    <div style="background:var(--bg-surface);border-bottom:1px solid var(--gold-border);
        padding:0 40px;display:flex;align-items:center;justify-content:space-between;height:56px;
        position:sticky;top:0;z-index:999;">
        <div style="display:flex;align-items:center;gap:20px;">
            <span style="font-family:var(--font-display);font-size:1.4rem;letter-spacing:4px;
                          color:var(--gold-bright);">CYBERSHIELD</span>
            <span style="font-size:0.7rem;color:var(--text-dim);letter-spacing:2px;
                          text-transform:uppercase;border-left:1px solid var(--gold-border);
                          padding-left:16px;">IDS / REAL-TIME THREAT DETECTION</span>
        </div>
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:8px;height:8px;border-radius:50%;
                             background:{dot_clr};{dot_anim}"></div>
                <span style="font-family:var(--font-mono);font-size:0.7rem;
                              color:var(--text-muted);letter-spacing:1px;">{status}</span>
            </div>
            <span style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-dim);">{now}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Last-threat banner — separate st.markdown so its f-string is isolated
    if last:
        tc      = LABEL_COLORS.get(last["final_label"], "#c8960a")
        label   = last["final_label"]
        src_ip  = last["src_ip"]
        dst_ip  = last["dst_ip"]
        dst_port = last["dst_port"]
        conf    = f"{last['confidence']:.0%}"
        ts      = last["timestamp"]
        st.markdown(f"""
        <div style="background:rgba(229,62,62,0.08);border-bottom:1px solid rgba(229,62,62,0.25);
            padding:8px 40px;display:flex;align-items:center;gap:16px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#e53e3e;
                         animation:pulse-red 1.2s ease infinite;flex-shrink:0;"></div>
            <span style="font-family:var(--font-mono);font-size:0.7rem;color:#e88888;letter-spacing:1px;">
                LAST THREAT &nbsp;&rsaquo;&nbsp;
                <span style="color:{tc};">{label}</span>
                &nbsp;&middot;&nbsp; {src_ip} &rarr; {dst_ip}:{dst_port}
                &nbsp;&middot;&nbsp; conf {conf}
                &nbsp;&middot;&nbsp; {ts}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # 3. Hero banner — static HTML, no interpolated values
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0b0900 0%,#1a1200 50%,#0b0900 100%);
        border-bottom:1px solid var(--gold-border);padding:48px 40px 40px;
        position:relative;overflow:hidden;">
        <div style="position:absolute;inset:0;background-image:
            linear-gradient(rgba(200,150,10,0.04) 1px,transparent 1px),
            linear-gradient(90deg,rgba(200,150,10,0.04) 1px,transparent 1px);
            background-size:40px 40px;pointer-events:none;"></div>
        <div style="position:absolute;left:0;top:0;bottom:0;width:3px;
                     background:linear-gradient(180deg,transparent,var(--gold-bright),transparent);"></div>
        <div style="position:relative;">
            <div style="font-family:var(--font-cond);font-size:0.72rem;letter-spacing:4px;
                         text-transform:uppercase;color:var(--gold);margin-bottom:10px;">
                &#9672; NETWORK INTRUSION DETECTION SYSTEM &mdash; BCT 2315 GROUP 6
            </div>
            <h1 style="font-family:var(--font-display);font-size:3.6rem;letter-spacing:4px;
                        margin:0;line-height:1;color:var(--text-primary);">A NEW ERA OF</h1>
            <h1 style="font-family:var(--font-display);font-size:3.6rem;letter-spacing:4px;
                        margin:0 0 20px;line-height:1;color:var(--gold-bright);">CYBER DEFENSE</h1>
            <p style="font-family:var(--font-body);font-size:0.9rem;font-weight:300;
                       color:var(--text-muted);max-width:520px;line-height:1.7;margin:0;">
                Real-time rule-based threat detection with machine learning classification and
                automated incident response. Detecting DoS, BruteForce, and PortScan attacks
                &mdash; stopping attacks before they strike.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_controls():
    """Control strip: start/pause, batch size, refresh rate, clear."""
    st.markdown("""
    <div style="background:var(--bg-surface);border-bottom:1px solid var(--gold-border);
        padding:16px 40px;">
    """, unsafe_allow_html=True)

    col_btn, col_batch, col_refresh, col_clear, col_spacer = st.columns([2, 2, 2, 2, 6])

    with col_btn:
        label = "⏹  PAUSE" if st.session_state["running"] else "▶  START MONITORING"
        if st.button(label, key="toggle_btn", use_container_width=True):
            st.session_state["running"] = not st.session_state["running"]

    with col_batch:
        batch = st.slider("Events / batch", 1, 15, 5, key="batch_size")

    with col_refresh:
        refresh = st.slider("Refresh (s)", 1, 10, 2, key="refresh_rate")

    with col_clear:
        if st.button("🗑  CLEAR LOG", key="clear_btn", use_container_width=True):
            st.session_state.update({
                "events":          [],
                "timeline":        deque(maxlen=TIMELINE_BINS * 10),
                "total_processed": 0,
                "threat_counts":   Counter(),
                "response_times":  [],
                "last_threat":     None,
                "peak_threat_rate": 0,
            })

    st.markdown("</div>", unsafe_allow_html=True)
    return batch, refresh


def render_kpi_strip():
    events  = st.session_state["events"]
    resp    = st.session_state["responder"]
    stats   = resp.get_stats()
    total   = st.session_state["total_processed"]
    threats = [e for e in events if e["final_label"] != "BENIGN"]
    rt      = st.session_state["response_times"]
    avg_rt  = f"{sum(rt)/len(rt):.2f} ms" if rt else "N/A"
    max_rt  = f"{max(rt):.2f} ms" if rt else "N/A"
    dos_c   = sum(1 for e in events if e["final_label"] == "DoS")
    brute_c = sum(1 for e in events if e["final_label"] == "BruteForce")
    scan_c  = sum(1 for e in events if e["final_label"] == "PortScan")
    det_r   = f"{(len(threats)/total*100):.1f}%" if total > 0 else "0.0%"
    ml_ok   = "ACTIVE" if st.session_state["ml_loaded"] else "INACTIVE"

    st.markdown(
        "<div style='background:var(--bg-surface);border-bottom:1px solid var(--gold-border);"
        "padding:8px 40px 4px;'>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Processed",   f"{total:,}",                   help="Network events analysed")
    c2.metric("Threats Detected",  f"{len(threats):,}",            help=f"DoS {dos_c}  Brute {brute_c}  Scan {scan_c}")
    c3.metric("Detection Rate",    det_r,                          help="Threats as % of total traffic")
    c4.metric("IPs Blocked",       str(stats["total_ips_blocked"]), help="Auto-contained by responder")
    c5.metric("Avg Response",      avg_rt,                         help=f"Peak: {max_rt}")
    c6.metric("ML Classifier",     ml_ok,                          help="Random Forest — target >=85%")
    st.markdown("</div>", unsafe_allow_html=True)


def render_detection_capabilities():
    _section_header("Detection Capabilities",
                     "ALL-IN-ONE PROTECTION DESIGNED TO DETECT, PREVENT",
                     border_top=False)

    cards = [
        ("◈", "Rule-Based Engine",   "DoS",        "#e53e3e",
         "10 handcrafted signature rules for DoS, BruteForce and PortScan with confidence scoring"),
        ("⚡", "ML Classifier",       "BruteForce", "#d4891a",
         "Random Forest trained on CICIDS2017 &amp; UNSW-NB15 &mdash; target &ge;85% accuracy, &lt;5% FPR"),
        ("◎", "Automated Response",  "PortScan",   "#c8960a",
         "Alerts, activity logging, and simulated IP blocking within 500ms of threat confirmation"),
        ("✓", "Live Dashboard",      "BENIGN",     "#2d8a4e",
         "Real-time Streamlit visualization with sub-2-second refresh and live threat feed"),
    ]

    cols = st.columns(4)
    for col, (icon, title, label, color, desc) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div style="padding:28px 24px;background:var(--bg-card);
                border:1px solid var(--gold-border);border-top:3px solid {color};
                border-bottom:1px solid var(--gold-border);
                animation:fadeUp 0.4s ease both;">
                <div style="font-size:1.6rem;color:{color};margin-bottom:14px;">{icon}</div>
                <div style="font-family:var(--font-cond);font-weight:700;font-size:0.9rem;
                             letter-spacing:1px;text-transform:uppercase;
                             color:var(--text-primary);margin-bottom:10px;">{title}</div>
                <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_charts():
    _section_header("Real-Time Analytics",
                     "THREAT DETECTION: STOPPING ATTACKS BEFORE THEY STRIKE")

    events  = st.session_state["events"]
    threats = [e for e in events if e["final_label"] != "BENIGN"]
    layout  = _base_layout(300)

    col_donut, col_bar, col_timeline = st.columns(3)

    # ── 1. Attack type donut ─────────────────────────────────────────────────────
    with col_donut:
        st.markdown('<div style="padding:0 16px 32px 40px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-cond);font-size:0.72rem;letter-spacing:3px;'
                    'text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">'
                    'Attack Type Distribution</div>', unsafe_allow_html=True)
        if events:
            counts = Counter(e["final_label"] for e in events)
            labels = list(counts.keys())
            values = list(counts.values())
            colors = [LABEL_COLORS.get(l, "#4a4030") for l in labels]
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.62,
                marker=dict(colors=colors, line=dict(color="#0b0900", width=3)),
                textinfo="label+percent",
                textfont=dict(size=11, family="Barlow Condensed"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            ))
            fig.update_layout(**layout, showlegend=False)
            fig.add_annotation(
                text=f"<b>{len(events)}</b><br>events",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#f0b90b", family="Bebas Neue"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(_empty_chart_msg("No data yet — start monitoring"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 2. Confidence bar chart (last 40 threats) ───────────────────────────────
    with col_bar:
        st.markdown('<div style="padding:0 16px 32px 16px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-cond);font-size:0.72rem;letter-spacing:3px;'
                    'text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">'
                    'Detection Confidence — Last 40 Threats</div>', unsafe_allow_html=True)
        last40 = threats[-40:]
        if last40:
            df_t = pd.DataFrame(last40)
            fig2 = go.Figure()
            for label, color in LABEL_COLORS.items():
                if label == "BENIGN":
                    continue
                sub = df_t[df_t["final_label"] == label]
                if not sub.empty:
                    fig2.add_trace(go.Bar(
                        x=sub["timestamp"], y=sub["confidence"],
                        name=label, marker_color=color, marker_line_width=0,
                        hovertemplate=(
                            f"<b>{label}</b><br>Time: %{{x}}<br>"
                            "Confidence: %{y:.0%}<extra></extra>"
                        ),
                    ))
            fig2.update_layout(
                **layout, barmode="stack",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=10, family="Barlow Condensed"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(
                    range=[0, 1],
                    gridcolor="rgba(200,150,10,0.08)",
                    tickformat=".0%",
                    tickfont=dict(size=10),
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(_empty_chart_msg("No threats detected yet"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. Threat timeline (area spark) ─────────────────────────────────────────
    with col_timeline:
        st.markdown('<div style="padding:0 40px 32px 16px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-cond);font-size:0.72rem;letter-spacing:3px;'
                    'text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">'
                    'Threat Rate Over Time</div>', unsafe_allow_html=True)

        tl = list(st.session_state["timeline"])
        if tl:
            # bucket by index into TIMELINE_BINS windows
            bucket_size = max(1, len(tl) // TIMELINE_BINS)
            buckets = [tl[i:i+bucket_size] for i in range(0, len(tl), bucket_size)]
            threat_rates = [
                sum(1 for lbl in bucket if lbl != "BENIGN") / len(bucket)
                for bucket in buckets
            ]
            x_vals = list(range(len(threat_rates)))
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=x_vals, y=threat_rates,
                mode="lines",
                line=dict(color="#e53e3e", width=2),
                fill="tozeroy",
                fillcolor="rgba(229,62,62,0.10)",
                hovertemplate="Threat rate: %{y:.0%}<extra></extra>",
            ))
            fig3.update_layout(
                **layout,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(
                    range=[0, 1],
                    gridcolor="rgba(200,150,10,0.08)",
                    tickformat=".0%",
                    tickfont=dict(size=10),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.markdown(_empty_chart_msg("Threat rate will appear here"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_live_log():
    _section_header("Real-Time Threat Feed", "LIVE EVENT LOG")

    events = st.session_state["events"]
    if not events:
        st.markdown(
            '<div style="padding:40px 40px;color:var(--text-dim);font-size:0.85rem;'
            'background:var(--bg-surface);">No events yet — click START MONITORING above.</div>',
            unsafe_allow_html=True,
        )
        return

    # Show last 60 events, newest first
    df = pd.DataFrame(events[-60:]).iloc[::-1].reset_index(drop=True)
    display_cols = [
        "timestamp", "src_ip", "dst_ip", "dst_port", "protocol",
        "final_label", "confidence", "rule_fired", "response_time_ms", "ip_blocked",
    ]
    df = df[[c for c in display_cols if c in df.columns]]
    df.columns = ["Time", "Src IP", "Dst IP", "Port", "Proto",
                  "Label", "Conf", "Rule", "RT (ms)", "Blocked"]

    st.markdown('<div style="padding:0 40px 32px;background:var(--bg-surface);">', unsafe_allow_html=True)

    _row_bg = {"DoS": "#3d0a0a", "BruteForce": "#2d1800", "PortScan": "#1a1500", "BENIGN": ""}

    def highlight_row(row):
        bg     = _row_bg.get(row["Label"], "")
        styles = [f"background-color: {bg}" if bg else ""] * len(row)
        try:
            idx  = list(row.index).index("Label")
            tc   = LABEL_COLORS.get(row["Label"], "#8a7d5a")
            styles[idx] = f"background-color: {bg}; color: {tc}; font-weight: 700;"
        except ValueError:
            pass
        return styles

    styled = (
        df.style
        .apply(highlight_row, axis=1)
        .format({"Conf": "{:.0%}", "RT (ms)": "{:.2f}"})
    )
    st.dataframe(styled, use_container_width=True, height=360)
    st.markdown('</div>', unsafe_allow_html=True)


def render_blocked_and_rules():
    _section_header("Detection Rules & Blocked Hosts", "SYSTEM INTELLIGENCE")

    col_rules, col_blocked = st.columns([3, 2])

    with col_rules:
        st.markdown('<div style="padding:0 20px 32px 40px;">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--font-cond);font-size:0.72rem;letter-spacing:3px;'
            'text-transform:uppercase;color:var(--text-muted);margin-bottom:16px;">'
            'Active Signature Rules</div>',
            unsafe_allow_html=True,
        )
        rules     = st.session_state["engine"].get_rule_summary()
        rule_cols = st.columns(2)
        for i, r in enumerate(rules):
            color = LABEL_COLORS.get(r["label"], "#c8960a")
            with rule_cols[i % 2]:
                st.markdown(f"""
                <div style="padding:14px 16px;margin-bottom:8px;background:var(--bg-card);
                    border:1px solid var(--gold-border);border-left:3px solid {color};">
                    <div style="font-family:var(--font-mono);font-size:0.65rem;
                                 color:{color};margin-bottom:4px;letter-spacing:1px;">
                        {r['label']} · {r['confidence']:.0%}</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);line-height:1.5;">
                        {r['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_blocked:
        st.markdown('<div style="padding:0 40px 32px 20px;">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--font-cond);font-size:0.72rem;letter-spacing:3px;'
            'text-transform:uppercase;color:var(--text-muted);margin-bottom:16px;">'
            'Blocked IP Addresses</div>',
            unsafe_allow_html=True,
        )
        blocked = sorted(st.session_state["responder"].get_blocked_ips())
        if blocked:
            for ip in blocked[:20]:
                st.markdown(f"""
                <div style="padding:10px 16px;margin-bottom:6px;
                    background:rgba(229,62,62,0.06);border:1px solid rgba(229,62,62,0.2);
                    border-left:3px solid #e53e3e;display:flex;align-items:center;gap:10px;">
                    <span style="color:#e53e3e;font-size:0.8rem;">✕</span>
                    <span style="font-family:var(--font-mono);font-size:0.78rem;
                                  color:#e88888;">{ip}</span>
                </div>
                """, unsafe_allow_html=True)
            if len(blocked) > 20:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:var(--text-dim);padding-top:8px;">'
                    f'+{len(blocked) - 20} more blocked</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="padding:32px 0;color:var(--text-dim);font-size:0.85rem;'
                'text-align:center;">No IPs blocked yet</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div style="background:var(--bg-surface);border-top:1px solid var(--gold-border);
        padding:40px 40px 32px;display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:40px;">
        <div>
            <div style="font-family:var(--font-display);font-size:1.2rem;
                         letter-spacing:4px;color:var(--gold-bright);margin-bottom:10px;">CYBERSHIELD</div>
            <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.7;max-width:280px;">
                Real-time network intrusion detection and response system.
                BCT 2315 Computer Systems Project · Group 6 · Multimedia University of Kenya.
            </div>
        </div>
        <div>
            <div style="font-family:var(--font-cond);font-size:0.68rem;letter-spacing:3px;
                         text-transform:uppercase;color:var(--gold);margin-bottom:12px;">Detection</div>
            <div style="font-size:0.8rem;color:var(--text-muted);line-height:2;">
                Rule Engine<br>ML Classifier<br>Threat Hunting<br>Port Scan Detection</div>
        </div>
        <div>
            <div style="font-family:var(--font-cond);font-size:0.68rem;letter-spacing:3px;
                         text-transform:uppercase;color:var(--gold);margin-bottom:12px;">Response</div>
            <div style="font-size:0.8rem;color:var(--text-muted);line-height:2;">
                Auto Alerts<br>IP Blocking<br>Event Logging<br>Response &lt;500ms</div>
        </div>
        <div>
            <div style="font-family:var(--font-cond);font-size:0.68rem;letter-spacing:3px;
                         text-transform:uppercase;color:var(--gold);margin-bottom:12px;">Stack</div>
            <div style="font-size:0.8rem;color:var(--text-muted);line-height:2;">
                Python 3.11<br>Streamlit<br>scikit-learn<br>CICIDS2017 / UNSW-NB15</div>
        </div>
    </div>
    <div style="background:var(--bg-base);border-top:1px solid var(--gold-border);
        padding:16px 40px;display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.72rem;color:var(--text-dim);">
            © 2026 Group 6 · BCT 2315 · Multimedia University of Kenya</span>
        <span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-dim);">
            CYBERSHIELD IDS v1.1</span>
    </div>
    """, unsafe_allow_html=True)


# ── Private helpers ──────────────────────────────────────────────────────────────
def _empty_chart_msg(text: str) -> str:
    return (
        f'<div style="height:300px;display:flex;align-items:center;justify-content:center;'
        f'color:var(--text-dim);font-size:0.85rem;">{text}</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()

    render_hero()
    batch, refresh = render_controls()
    render_kpi_strip()
    render_detection_capabilities()
    render_charts()
    render_live_log()
    render_blocked_and_rules()
    render_footer()

    if st.session_state["running"]:
        records     = generate_traffic_batch(batch)
        batch_threats = 0

        for record in records:
            result = run_detection(record)
            st.session_state["events"].append(result)
            st.session_state["total_processed"] += 1
            st.session_state["response_times"].append(result["response_time_ms"])
            # feed timeline deque
            st.session_state["timeline"].append(result["final_label"])

            if result["final_label"] != "BENIGN":
                st.session_state["threat_counts"][result["final_label"]] += 1
                st.session_state["last_threat"] = result
                batch_threats += 1

        # update peak threat rate
        if batch_threats > st.session_state["peak_threat_rate"]:
            st.session_state["peak_threat_rate"] = batch_threats

        # trim event log to rolling window
        if len(st.session_state["events"]) > MAX_EVENTS:
            st.session_state["events"] = st.session_state["events"][-MAX_EVENTS:]

        time.sleep(refresh)
        st.rerun()


if __name__ == "__main__":
    main()