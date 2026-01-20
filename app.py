import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(
    page_title="UIDAI Enrolment Intelligence System",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.metric-box {
    background-color:#161b22;
    padding:20px;
    border-radius:12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🆔 UIDAI Aadhaar Enrolment Intelligence System")
st.caption("Data Cleaning • Baseline Analytics • Forecasting • Decision Support")

# ---------------- FILE UPLOAD ----------------
files = st.file_uploader(
    "📤 Upload UIDAI Enrolment CSV files",
    type="csv",
    accept_multiple_files=True
)

if not files:
    st.info("Upload one or more UIDAI enrolment CSV files to begin")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

st.subheader("📄 Raw Data Preview")
st.dataframe(df.head())

# ---------------- BASIC COLUMN CHECK ----------------
required_cols = ["state", "date", "age_0_5", "age_5_17", "age_18_greater"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------------- DATE & ENROLMENTS ----------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_enrolments"] = df[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
df = df.dropna(subset=["date"])

# ---------------- STATE STANDARDIZATION ----------------
state_fix = {
    "west bengal": "West Bengal",
    "west bangal": "West Bengal",
    "westbengal": "West Bengal",
    "orissa": "Odisha",
    "odissa": "Odisha",
    "andaman & nicobar islands": "Andaman & Nicobar Islands",
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "dadra & nagar haveli": "Dadra & Nagar Haveli",
    "dadra and nagar haveli": "Dadra & Nagar Haveli",
    "daman & diu": "Daman & Diu",
    "pondicherry": "Puducherry",
    "jammu & kashmir": "Jammu And Kashmir"
}

df["state_clean"] = (
    df["state"]
    .astype(str)
    .str.lower()
    .str.strip()
    .replace(state_fix)
)

df["state_display"] = df["state_clean"].str.title()

# ---------------- DAILY AGGREGATION ----------------
daily = df.groupby(["state_clean", "date"]).agg(
    total_enrolments=("total_enrolments", "sum"),
    age_0_5=("age_0_5", "sum"),
    age_5_17=("age_5_17", "sum"),
    age_18_greater=("age_18_greater", "sum")
).reset_index()

# ---------------- BASELINE SERIES ----------------
daily["baseline_7d"] = (
    daily.groupby("state_clean")["total_enrolments"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

daily["deviation"] = (
    abs(daily["total_enrolments"] - daily["baseline_7d"]) /
    daily["baseline_7d"].replace(0, np.nan)
).fillna(0)

# ---------------- COVERAGE SCORE ----------------
coverage = daily.groupby("state_clean").agg(
    reported_days=("date", "nunique")
).reset_index()

coverage["expected_days"] = daily["date"].nunique()
coverage["coverage_score"] = (coverage["reported_days"] / coverage["expected_days"]).round(2)

# ---------------- VOLATILITY ----------------
volatility = daily.groupby("state_clean").agg(
    mean_enrol=("total_enrolments", "mean"),
    std_enrol=("total_enrolments", "std")
).reset_index()

volatility["volatility_index"] = (volatility["std_enrol"] / volatility["mean_enrol"]).fillna(0).round(2)

# ---------------- STABILITY ----------------
stability = daily.groupby("state_clean")["deviation"].mean().reset_index()
stability["stability_score"] = (1 - stability["deviation"]).clip(0, 1).round(2)

# ---------------- FORECAST (NO ML) ----------------
recent = daily.sort_values("date").groupby("state_clean").tail(7)
growth = recent.groupby("state_clean")["baseline_7d"].pct_change().mean().fillna(0).reset_index()
last_base = recent.groupby("state_clean")["baseline_7d"].last().reset_index()

forecast = pd.merge(last_base, growth, on="state_clean")
forecast["7_day_forecast"] = (forecast["baseline_7d"] * (1 + forecast["baseline_7d_y"])).round(0)

# ---------------- FINAL MERGE ----------------
final = (
    stability
    .merge(coverage, on="state_clean")
    .merge(volatility, on="state_clean")
    .merge(forecast[["state_clean", "7_day_forecast"]], on="state_clean", how="left")
)

# ---------------- UIDAI ACTION ENGINE ----------------
def uidai_action(row):
    if row["coverage_score"] < 0.5:
        return "⚠️ Audit reporting pipeline"
    if row["volatility_index"] > 0.6:
        return "⚠️ Investigate spikes"
    if row["stability_score"] > 0.8:
        return "✅ Expand enrolment centers"
    if row["7_day_forecast"] > row["mean_enrol"] * 1.2:
        return "⚡ Scale staff & kits"
    return "Monitor"

final["UIDAI_Recommendation"] = final.apply(uidai_action, axis=1)

# ---------------- DASHBOARD ----------------
st.markdown("## 📊 State-wise Intelligence Summary")
st.dataframe(final, use_container_width=True)

# ---------------- VISUALS ----------------
c1, c2 = st.columns(2)

with c1:
    fig1 = px.bar(
        final.sort_values("stability_score"),
        x="stability_score",
        y="state_clean",
        orientation="h",
        title="Stability Score by State"
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.bar(
        final.sort_values("volatility_index"),
        x="volatility_index",
        y="state_clean",
        orientation="h",
        title="Volatility Index by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- AGE DISTRIBUTION ----------------
st.markdown("## 👶🧑‍🎓👴 Age-wise Enrolment Contribution")

age_df = daily.groupby("state_clean")[["age_0_5","age_5_17","age_18_greater"]].sum()
age_pct = age_df.div(age_df.sum(axis=1), axis=0).reset_index()

fig3 = px.bar(
    age_pct,
    x="state_clean",
    y=["age_0_5","age_5_17","age_18_greater"],
    title="Age Group Contribution (%)"
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------- INSIGHTS ----------------
st.markdown("## 🧠 Strategic Insights for UIDAI")
st.markdown("""
• Standardization prevents fragmented national statistics  
• Coverage score identifies unreliable reporting regions  
• Baseline forecasting supports planning without ML  
• Volatility flags operational anomalies  
• Recommendations guide audits & center expansion  
""")

st.success("✅ Premium UIDAI Enrolment Intelligence Prototype Ready")
