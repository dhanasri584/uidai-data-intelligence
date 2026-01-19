import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="UIDAI Enrolment Intelligence System",
    layout="wide"
)

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
    "Upload UIDAI Enrolment CSV files",
    type="csv",
    accept_multiple_files=True
)

if not files:
    st.stop()

@st.cache_data
def load(files):
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = load(files)

# ---------------- STATE STANDARDIZATION ----------------
state_fix = {
    "west bengal": "West Bengal",
    "west bangal": "West Bengal",
    "westbengal": "West Bengal",
    "orissa": "Odisha",
    "odissa": "Odisha"
}

df["state_original"] = df["state"]
df["state_clean"] = (
    df["state"].astype(str).str.lower().str.strip()
    .replace(state_fix)
    .str.title()
)

# ---------------- DATE & ENROLMENTS ----------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["total_enrolments"] = (
    df["age_0_5"] +
    df["age_5_17"] +
    df["age_18_greater"]
)

df = df.dropna(subset=["date"])

# ---------------- DAILY AGGREGATION ----------------
daily = (
    df.groupby(["state_clean", "date"])["total_enrolments"]
    .sum()
    .reset_index()
)

# ---------------- BASELINE SERIES ----------------
daily["baseline"] = (
    daily.groupby("state_clean")["total_enrolments"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Avoid division by zero
daily["baseline"] = daily["baseline"].replace(0, 1e-6)
daily["deviation"] = abs(daily["total_enrolments"] - daily["baseline"]) / daily["baseline"]

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

volatility["volatility_index"] = (volatility["std_enrol"] / volatility["mean_enrol"]).round(2)

# ---------------- STABILITY ----------------
stability = daily.groupby("state_clean")["deviation"].mean().reset_index()
stability["stability_score"] = (1 - stability["deviation"]).round(2)

# ---------------- FORECAST ----------------
# last 7 days per state
forecast_window = (
    daily.sort_values("date")
    .groupby("state_clean")
    .tail(7)
    .copy()
)

# percent growth in baseline
forecast_window["baseline_growth"] = (
    forecast_window
    .groupby("state_clean")["baseline"]
    .pct_change()
)

# average growth rate per state
growth_rate = (
    forecast_window
    .groupby("state_clean")["baseline_growth"]
    .mean()
    .fillna(0)
    .reset_index()
)

# last baseline value per state
last_baseline = (
    forecast_window
    .groupby("state_clean")["baseline"]
    .last()
    .reset_index()
)

# merge forecast inputs
forecast_df = pd.merge(
    last_baseline,
    growth_rate,
    on="state_clean",
    how="left"
)

# compute 7-day forecast
forecast_df["7_day_forecast"] = (
    forecast_df["baseline"] * (1 + forecast_df["baseline_growth"])
).round(0)

# ---------------- MERGE ALL ----------------
final = stability.merge(coverage, on="state_clean")
final = final.merge(volatility, on="state_clean")
final = final.merge(forecast_df[["state_clean", "7_day_forecast"]], on="state_clean", how="left")

# ---------------- RECOMMENDATIONS ----------------
def uidai_action(row):
    if row["coverage_score"] < 0.7:
        return "Data incomplete – audit reporting pipeline"
    if row["volatility_index"] > 0.6:
        return "High volatility – investigate enrolment spikes"
    if row["stability_score"] > 0.8:
        return "Stable – increase enrolment centers"
    return "Monitor closely"

final["UIDAI_Recommendation"] = final.apply(uidai_action, axis=1)

# ---------------- DASHBOARD ----------------
st.markdown("## 📊 State-wise Intelligence Summary")
st.dataframe(final, use_container_width=True)

# ---------------- VISUALS ----------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        final.sort_values("stability_score"),
        x="stability_score",
        y="state_clean",
        orientation="h",
        title="Stability Score by State"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        final.sort_values("volatility_index"),
        x="volatility_index",
        y="state_clean",
        orientation="h",
        title="Volatility Index by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- INSIGHTS ----------------
st.markdown("## 📝 Strategic Insights for UIDAI")
st.markdown("""
• Standardizing state names prevents analytical fragmentation  
• Coverage score highlights unreliable reporting regions  
• Baseline forecasting enables demand planning without ML  
• Combined metrics enable targeted audits and resource allocation  
""")

st.success("Premium UIDAI Enrolment Intelligence Prototype Ready")
