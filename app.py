import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_audit import load_and_clean_data, find_col
from scoring_engine import (compute_dimension_scores, get_gap_analysis,
                             compute_momentum, get_silent_winner_flag,
                             get_customer_persona)
from report_generator import generate_pdf_report

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligence Engine v1.3",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LIGHT-MODE CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container,
[class*="css"] {
  background-color: #F0F4F8 !important;
  color: #0F172A !important;
  font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

[data-testid="stSidebar"] > div:first-child {
  background: #0F172A !important;
  border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
  background: #1E293B !important; border-color: #334155 !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span { color: white !important; }
[data-testid="stSidebar"] label {
  color: #94A3B8 !important; font-size:11px !important;
  text-transform:uppercase !important; letter-spacing:0.08em !important;
}

.kpi-card {
  background: #FFFFFF; border-radius: 12px; padding: 18px 22px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.05);
  border: 1px solid #E2E8F0; position: relative; overflow: hidden; margin-bottom: 0;
}
.kpi-card::before {
  content: ''; position: absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg,#0EA5E9,#14B8A6);
  border-radius: 12px 12px 0 0;
}
.kpi-label { font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#64748B; margin-bottom:6px; }
.kpi-value { font-size:32px; font-weight:800; color:#0F172A; line-height:1; font-family:'Space Mono',monospace; }
.kpi-sub   { font-size:11px; color:#64748B; margin-top:4px; }
.delta-pos { color:#22C55E; font-weight:700; }
.delta-neg { color:#EF4444; font-weight:700; }
.delta-warn{ color:#F59E0B; font-weight:700; }

.section-card {
  background: #FFFFFF; border-radius: 12px; padding: 22px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07); border: 1px solid #E2E8F0; margin-bottom: 18px;
}
.card-header {
  font-size:13px; font-weight:700; color:#0F172A;
  display:flex; align-items:center; gap:8px;
  padding-bottom:12px; margin-bottom:14px; border-bottom:1px solid #E2E8F0;
}

.engine-badge {
  background: linear-gradient(135deg,#0EA5E9,#14B8A6);
  color:white; padding:4px 12px; border-radius:20px;
  font-size:10px; font-weight:700; letter-spacing:0.08em; display:inline-block; margin-bottom:14px;
}
.page-header {
  background:#FFFFFF; border-radius:12px; padding:18px 26px;
  margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.07); border:1px solid #E2E8F0;
}
.rest-name { font-size:26px; font-weight:800; color:#0F172A; margin:0; }
.rest-meta { font-size:12px; color:#64748B; margin-top:4px; }

.badge-high   { background:#FEF3C7; color:#92400E; padding:2px 9px; border-radius:20px; font-size:10px; font-weight:700; }
.badge-med    { background:#DBEAFE; color:#1E40AF; padding:2px 9px; border-radius:20px; font-size:10px; font-weight:700; }
.badge-low    { background:#DCFCE7; color:#166534; padding:2px 9px; border-radius:20px; font-size:10px; font-weight:700; }
.badge-silent { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }

.action-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 14px; border-radius:8px; border:1px solid #E2E8F0;
  background:#F8FAFC; margin-bottom:8px; transition: all 0.15s;
}
.action-row:hover { background:#F0F9FF; border-color:#0EA5E9; }
.act-title { font-size:13px; font-weight:600; color:#0F172A; }
.act-sub   { font-size:11px; color:#64748B; margin-top:2px; }
.act-imp   { font-size:11px; color:#0EA5E9; font-weight:600; margin-top:3px; }

.pitch-wrap {
  background: linear-gradient(135deg,#0F172A 0%,#1E3A5F 100%);
  border-radius:10px; padding:18px; margin-bottom:10px;
}
.pitch-lang { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px; }
.pitch-body { font-size:12.5px; line-height:1.65; }
.pitch-de   { font-size:11.5px; border-left:3px solid #14B8A6; padding-left:10px; font-style:italic; color:#94A3B8; }

.resp-zero  { color:#EF4444; font-weight:800; }
.resp-low   { color:#F59E0B; font-weight:800; }
.resp-good  { color:#22C55E; font-weight:800; }

.stDownloadButton > button {
  background: linear-gradient(135deg,#0EA5E9,#14B8A6) !important;
  color:white !important; border:none !important; border-radius:8px !important;
  font-weight:600 !important; padding:9px 20px !important; width:100% !important; font-size:13px !important;
}
.stButton > button {
  background: linear-gradient(135deg,#0EA5E9,#14B8A6) !important;
  color:white !important; border:none !important; border-radius:8px !important;
  font-weight:600 !important; width:100% !important;
}
[data-testid="stVerticalBlock"] { background: transparent !important; }
div[data-testid="metric-container"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD DATA ────────────────────────────────────────────────────────────────────
# IMPORTANT: if you update your CSV files you must clear the Streamlit cache.
# Either restart the app or call load_data.clear() programmatically.
@st.cache_data(show_spinner=False)
def load_data():
    return load_and_clean_data()   # data_audit._resolve_path() finds the CSVs automatically

with st.spinner("Loading intelligence engine..."):
    df_rest, df_rev, benchmarks = load_data()

restaurant_names = sorted(df_rest["name"].dropna().unique().tolist())


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 20px;border-bottom:1px solid #1E293B;margin-bottom:18px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <span style="font-size:20px">⚡</span>
        <span style="font-size:17px;font-weight:800;color:white">Intelligence Engine</span>
      </div>
      <div style="font-size:11px;color:#475569">Revenue Audit Platform v1.3</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#94A3B8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Select Establishment</p>', unsafe_allow_html=True)
    selected_restaurant = st.selectbox("", restaurant_names, label_visibility="collapsed")

    st.markdown('<p style="color:#94A3B8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-top:16px;margin-bottom:6px">District Filter</p>', unsafe_allow_html=True)
    district = st.selectbox("", ["FRA-Standard", "FRA-Premium", "FRA-All"],
                             label_visibility="collapsed", key="dist")



    st.markdown("---")
    st.markdown(f'<p style="color:#475569;font-size:11px;line-height:1.6">Analyzing <b style="color:#94A3B8">{len(restaurant_names)} establishments</b> across Frankfurt City.</p>', unsafe_allow_html=True)

    # Silent winners list
    st.markdown('<p style="color:#94A3B8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-top:18px;margin-bottom:8px">🔴 Silent Winners</p>', unsafe_allow_html=True)
    sw_list = df_rest[
        (df_rest["rating_n"] >= 4.5) & (df_rest["res_rate"].fillna(0) < 0.3)
    ]["name"].head(4).tolist()
    for sw in sw_list:
        st.markdown(
            f'<div style="background:#1E293B;border-left:3px solid #EF4444;padding:7px 11px;'
            f'border-radius:4px;margin-bottom:5px;font-size:11px;color:#FCA5A5">{sw}</div>',
            unsafe_allow_html=True
        )


# ─── COMPUTE ──────────────────────────────────────────────────────────────────────
scores      = compute_dimension_scores(selected_restaurant, df_rest, df_rev)
gaps        = get_gap_analysis(scores, benchmarks)
momentum    = compute_momentum(selected_restaurant, df_rev, df_rest)
persona     = get_customer_persona(selected_restaurant, df_rest, df_rev)
silent_flag = get_silent_winner_flag(selected_restaurant, df_rest)
res_data    = df_rest[df_rest["name"] == selected_restaurant].iloc[0]

# District ranking
@st.cache_data(show_spinner=False)
def compute_all_ranks(_df_rest, _df_rev):
    out = []
    for _, r in _df_rest.iterrows():
        try:
            s = compute_dimension_scores(r["name"], _df_rest, _df_rev)["Composite"]
        except Exception:
            s = 0
        out.append({"name": r["name"], "score": s})
    return (
        pd.DataFrame(out)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

df_ranks = compute_all_ranks(df_rest, df_rev)
df_ranks["rank"] = df_ranks.index + 1
cur_rank = int(df_ranks[df_ranks["name"] == selected_restaurant]["rank"].values[0])
total    = len(df_ranks)


# ─── PAGE HEADER ──────────────────────────────────────────────────────────────────
silent_badge = (
    '&nbsp;&nbsp;<span class="badge-silent">🔴 SILENT WINNER DETECTED</span>'
    if silent_flag else ""
)
st.markdown(f"""
<div class="page-header">
  <div class="engine-badge">⚡ Intelligence Engine Active v1.3</div>
  <h1 class="rest-name">{selected_restaurant}</h1>
  <div class="rest-meta">
    📍 District Context: Ranked <strong>#{cur_rank}</strong> of {total} establishments
    in Frankfurt City {silent_badge}
  </div>
</div>
""", unsafe_allow_html=True)


# ─── KPI TILES ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
health    = scores["Composite"]
resp_pct  = scores["Responsiveness"]
sent_pct  = scores["Intelligence"]
vis_pct   = scores["Visibility"]

# Responsiveness colour coding
if resp_pct == 0:
    resp_color_cls = "delta-neg"
    resp_delta     = "⚠ No replies found"
elif resp_pct < 30:
    resp_color_cls = "delta-warn"
    resp_delta     = f"↘ Low ({resp_pct:.0f}%)"
else:
    resp_color_cls = "delta-pos"
    resp_delta     = f"↗ +{max(resp_pct - 70, 0):.0f}% vs avg"

tiles = [
    (c1, "Overall Score",    f"{health:.1f}",         "out of 100",         f"↗ Score", True),
    (c2, "Rank",             f"#{cur_rank}",           "Frankfurt City",     "↗ +2",    True),
    (c3, "Responsiveness",   f"{resp_pct:.0f}%",       "Owner reply rate",   resp_delta, resp_pct >= 50),
    (c4, "Sentiment",        f"{sent_pct:.0f}%",       "Review sentiment",
     ("↗ Strong" if sent_pct >= 75 else "↘ Needs work"), sent_pct >= 75),
    (c5, "Freshness",        f"{vis_pct:.0f}%",        "Review velocity",
     ("↗ Active" if vis_pct >= 50 else "↘ Slow"),        vis_pct >= 50),
]
for col, label, val, sub, delta, pos in tiles:
    dc = "delta-pos" if pos else "delta-neg"
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub} &nbsp;<span class="{dc}">{delta}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# ─── RESPONSIVENESS EXPLAINER (shown when 0%) ─────────────────────────────────────
if resp_pct == 0:
    has_reviews = "_slug" in df_rest.columns and (
        df_rev["_slug"].eq(res_data.get("_slug", "")).sum() > 0
        if "_slug" in df_rev.columns else False
    )
    if has_reviews:
        msg = (
            "📋 **Responsiveness is 0%** — this restaurant has reviews in the dataset "
            "but the owner has not replied to any of them. "
            "This is a major sales opportunity for Praxiotech!"
        )
    else:
        msg = (
            "📋 **Responsiveness shows 0%** — no reviews were found in reviews.csv for this "
            "restaurant, so the response rate cannot be calculated. "
            "This is normal for restaurants not yet scraped."
        )
    st.info(msg)


# ─── ROW 2: RADAR + GAP ───────────────────────────────────────────────────────────
col_radar, col_gap = st.columns(2)

with col_radar:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⚙️ Dimension Radar</div>', unsafe_allow_html=True)

    dim_labels = ["Reputation", "Responsiveness", "Digital\nPresence", "Intelligence", "Visibility"]
    sv = [scores["Reputation"], scores["Responsiveness"], scores["Digital Presence"],
          scores["Intelligence"], scores["Visibility"]]
    bv = [benchmarks.get("rating", 4.4) * 20, 90, 85, 75, 70]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sv + [sv[0]], theta=dim_labels + [dim_labels[0]],
        fill="toself", name="Score",
        line=dict(color="#0EA5E9", width=2.5),
        fillcolor="rgba(14,165,233,0.13)",
        marker=dict(size=5, color="#0EA5E9"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=bv + [bv[0]], theta=dim_labels + [dim_labels[0]],
        fill="toself", name="Benchmark",
        line=dict(color="#A855F7", width=1.5, dash="dot"),
        fillcolor="rgba(168,85,247,0.05)",
        marker=dict(size=4, color="#A855F7"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9), gridcolor="#F0E2E2"),
            angularaxis=dict(tickfont=dict(color="#0F172A", size=11, family="Inter")),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=11)),
        height=340, margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_gap:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 Competitive Gap Analysis</div>', unsafe_allow_html=True)

    gap_items = [
        ("Customer Responsiveness", scores["Responsiveness"], 90),
        ("Market Sentiment",        scores["Intelligence"],   75),
        ("Review Freshness",        scores["Visibility"],     70),
        ("Brand Visibility",        scores["Digital Presence"], 80),
        ("Reputation Score",        scores["Reputation"],     benchmarks.get("rating", 4.4) * 20),
    ]
    for label, current, target in gap_items:
        diff  = round(current - target, 1)
        color = "#22C55E" if diff >= 0 else "#EF4444"
        sign  = "+" if diff >= 0 else ""
        bar_w = min(current, 100)
        tgt_w = min(target, 100)
        st.markdown(f"""
        <div style="margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px">
            <span style="font-size:12px;font-weight:600;color:#0F172A">{label}</span>
            <span style="font-size:12px;font-weight:700;color:{color}">{sign}{diff}%</span>
          </div>
          <div style="position:relative;height:8px;background:#F1F5F9;border-radius:4px">
            <div style="position:absolute;left:0;top:0;height:100%;width:{bar_w}%;
                        background:linear-gradient(90deg,#0EA5E9,#14B8A6);border-radius:4px"></div>
            <div style="position:absolute;top:-3px;left:{tgt_w}%;width:2px;height:14px;
                        background:#0F172A;border-radius:2px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:3px">
            <span style="font-size:10px;color:#94A3B8">Current: {current:.0f}%</span>
            <span style="font-size:10px;color:#94A3B8">Target: {target:.0f}%</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── ROW 3: PERSONA + ACTIONS ─────────────────────────────────────────────────────
col_ai, col_act = st.columns(2)

with col_ai:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">👤 AI Customer Insights</div>', unsafe_allow_html=True)
    p = persona
    st.markdown(f"""
    <div style="margin-bottom:12px">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#94A3B8;margin-bottom:3px">PRIMARY PERSONA</div>
      <div style="font-size:15px;font-weight:700;color:#0F172A">{p["primary"]}</div>
    </div>
    <div style="margin-bottom:12px">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#94A3B8;margin-bottom:3px">SEGMENT</div>
      <div style="font-size:13px;font-weight:600;color:#334155">{p["segment"]}</div>
    </div>
    <div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:8px;padding:10px;margin-bottom:14px">
      <span style="font-size:11px;font-weight:700;color:#0369A1">Core Motivation:</span>
      <span style="font-size:11px;color:#0F172A;margin-left:4px">{p["motivation"]}</span>
    </div>
    <div class="pitch-wrap">
      <div class="pitch-lang" style="color:#0EA5E9">✦ SALES PITCH — ENGLISH</div>
      <div class="pitch-body" style="color:#E2E8F0">{p["pitch_en"]}</div>
      <div style="height:10px"></div>
      <div class="pitch-lang" style="color:#14B8A6">✦ VERKAUFSPITCH — DEUTSCH</div>
      <div class="pitch-de">{p["pitch_de"]}</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        pdf_bytes = generate_pdf_report(
            selected_restaurant, res_data, scores, gaps, momentum,
            persona, benchmarks, df_rest, df_rev, cur_rank, total,
        )
        st.download_button(
            label="📄 Export Full Intelligence Brief (PDF)",
            data=pdf_bytes,
            file_name=f"Revenue_Intelligence_{selected_restaurant.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_act:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">💡 Actionable Solutions</div>', unsafe_allow_html=True)

    resp_lift = max(round((90 - scores["Responsiveness"]) * 0.8), 5)
    actions = [
        ("⚡", "Optimize Response Time",    "Reduce avg reply to under 2 hours — top revenue lever",
         f"Est. +{resp_lift:.0f} pts score lift",  "HIGH",   "badge-high"),
        ("⭐", "Launch Review Campaign",     "Target 15 new reviews this quarter via post-visit SMS",
         "Est. +12% visibility",             "MEDIUM", "badge-med"),
        ("🔗", "Update Google Profile",      "Refresh photos, menu links & booking CTA",
         "Est. +8% CTR",                     "LOW",    "badge-low"),
        ("🤖", "AI Review Management",       "Automate personalized responses at scale — 120 EUR/mo",
         "Est. 3x response rate",            "HIGH",   "badge-high"),
        ("📡", "Sentiment Monitoring",       "Real-time alerts for negative reviews across platforms",
         f"Protect {float(res_data.get('rating_n', 4)):.1f}★ rating", "MEDIUM", "badge-med"),
    ]
    for icon, title, sub, impact, priority, badge in actions:
        st.markdown(f"""
        <div class="action-row">
          <div>
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:3px">
              <span>{icon}</span>
              <span class="act-title">{title}</span>
              <span class="{badge}">{priority}</span>
            </div>
            <div class="act-sub">{sub}</div>
            <div class="act-imp">{impact}</div>
          </div>
          <span style="color:#CBD5E1;font-size:16px">→</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── ROW 4: MOMENTUM + DONUT + GAUGE ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">📈 Momentum Tracker</div>', unsafe_allow_html=True)

cm1, cm2, cm3 = st.columns([3, 1, 1])

with cm1:
    mom = momentum
    if mom is not None and len(mom) > 0:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=mom["month"], y=mom["count"],
            mode="lines+markers", name="Reviews/Month",
            line=dict(color="#0EA5E9", width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
            marker=dict(size=6, color="#0EA5E9", line=dict(color="white", width=1.5)),
        ))
        fig_m.update_layout(
            xaxis=dict(tickfont=dict(color="#1E293B", size=10)),
            yaxis=dict(tickfont=dict(color="#1E293B", size=10)),
            height=210, margin=dict(l=40, r=20, t=10, b=30),
            paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No momentum data available.")

with cm2:
    if "_slug" in df_rest.columns and "_slug" in df_rev.columns:
        try:
            rest_slug = df_rest[df_rest["name"] == selected_restaurant].iloc[0]["_slug"]
            sub = df_rev[df_rev["_slug"] == rest_slug]
        except (IndexError, KeyError):
            sub = pd.DataFrame()
    else:
        sub = pd.DataFrame()

    if len(sub) > 0 and "review_rating" in sub.columns:
        rc = sub["review_rating"].value_counts().sort_index(ascending=False)
    else:
        rc = pd.Series([40, 30, 15, 10, 5], index=[5, 4, 3, 2, 1])

    fig_d = go.Figure(go.Pie(
        values=rc.values,
        labels=[f"{i}★" for i in rc.index],
        hole=0.6,
        marker=dict(colors=["#22C55E", "#86EFAC", "#FCD34D", "#FCA5A5", "#EF4444"]),
        textfont=dict(size=9), showlegend=True,
    ))
    fig_d.update_layout(
        title=dict(text="Rating Split", font=dict(size=11), x=0.5),
        height=210, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(font=dict(size=8), orientation="v"),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

with cm3:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        number={"font": {"size": 26, "family": "Space Mono", "color": "#0F172A"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 8}},
            "bar":  {"color": "#0EA5E9", "thickness": 0.28},
            "bgcolor": "#F1F5F9", "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 50],  "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 100],"color": "#DCFCE7"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 2}, "thickness": 0.75, "value": health},
        },
    ))
    fig_g.update_layout(
        title=dict(text="Health Score", font=dict(size=11), x=0.5),
        height=210, margin=dict(l=20, r=20, t=30, b=0),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

st.markdown("</div>", unsafe_allow_html=True)


# ─── ROW 5: LEADERBOARD ───────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">🏆 District Leaderboard — Top 10</div>', unsafe_allow_html=True)

top10  = df_ranks.head(10).copy()
colors = ["#0EA5E9" if n == selected_restaurant else "#D266A5" for n in top10["name"]]

fig_lb = go.Figure()
fig_lb.add_trace(go.Bar(
    x=top10["score"], y=top10["name"],
    orientation="h",
    marker=dict(color=colors, line=dict(width=0)),
    text=[f"{s:.1f}" for s in top10["score"]],
    textposition="inside",
    textfont=dict(size=10, color="white", family="Space Mono"),
    hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
))
fig_lb.update_layout(
    xaxis=dict(tickfont=dict(color='#1E293B', size=10)),
    yaxis=dict(tickfont=dict(color='#1E293B', size=10)),
    height=290, margin=dict(l=10, r=20, t=10, b=30),
    paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
)
st.plotly_chart(fig_lb, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:16px;color:#94A3B8;font-size:11px;
            border-top:1px solid #E2E8F0;margin-top:6px">
  ⚡ Intelligence Engine v1.3 &nbsp;·&nbsp; Praxiotech GmbH &nbsp;·&nbsp;
  Frankfurt City · {total} Establishments &nbsp;·&nbsp; Data refreshed daily
</div>
""", unsafe_allow_html=True)