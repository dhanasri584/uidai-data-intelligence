import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="UIDAI Data Intelligence Platform",
    layout="wide"
)

st.title("🆔 UIDAI Aadhaar Enrolment Data Intelligence Platform")
st.caption("State Standardization • Baseline Analysis • Anomaly Detection • Geo-Visualisation")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    files = [
        "api_data_aadhar_enrolment_0_500000.csv",
        "api_data_aadhar_enrolment_500000_1000000.csv",
        "api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df

@st.cache_data
def load_india_geojson():
    url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    return requests.get(url).json()

df = load_data()
india_geojson = load_india_geojson()

# ---------------- STATE STANDARDIZATION ----------------
state_mapping = {
    "west bengal": "West Bengal",
    "west  bengal": "West Bengal",
    "west bangal": "West Bengal",
    "westbengal": "West Bengal",
    "andhra pradesh": "Andhra Pradesh",
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "dadra and nagar haveli": "Dadra & Nagar Haveli",
    "daman and diu": "Daman & Diu",
}

df["state_original"] = df["state"]

df["state_clean"] = (
    df["state"]
    .astype(str)
    .str.lower()
    .str.strip()
    .replace(state_mapping)
    .str.title()
)

# ---------------- DATE & ENROLMENT ----------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["total_enrolments"] = (
    df["age_0_5"] +
    df["age_5_17"] +
    df["age_18_greater"]
)

df = df.dropna(subset=["date"])

# ---------------- BASELINE SERIES ----------------
daily = (
    df.groupby(["state_clean", "date"])["total_enrolments"]
    .sum()
    .reset_index()
)

daily["baseline"] = (
    daily.groupby("state_clean")["total_enrolments"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

daily["deviation_pct"] = (
    (daily["total_enrolments"] - daily["baseline"]) / daily["baseline"]
).abs()

daily["anomaly"] = daily["deviation_pct"] > 0.5

# ---------------- LATEST SNAPSHOT ----------------
latest = daily.sort_values("date").groupby("state_clean").tail(1)

# ---------------- INDIA MAP ----------------
st.markdown("## 🗺️ India Aadhaar Enrolment Heatmap")

latest["state_match"] = latest["state_clean"].str.upper().str.strip()

fig_map = px.choropleth(
    latest,
    geojson=india_geojson,
    featureidkey="properties.ST_NM",
    locations="state_match",
    color="total_enrolments",
    color_continuous_scale="YlOrRd",
    title="Latest Aadhaar Enrolments by State"
)

fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(height=650)

st.plotly_chart(fig_map, use_container_width=True)

# ---------------- STANDARDIZATION IMPACT ----------------
st.markdown("## 🧹 State Name Standardization Impact")

variants = (
    df.groupby("state_clean")["state_original"]
    .nunique()
    .reset_index(name="name_variants")
)

st.dataframe(variants[variants["name_variants"] > 1], use_container_width=True)

# ---------------- UNDER-ENROLLED STATES ----------------
st.markdown("## 🚨 Under-Enrolled States vs Baseline")

under = latest[latest["total_enrolments"] < latest["baseline"]]

st.dataframe(
    under[["state_clean", "total_enrolments", "baseline"]],
    use_container_width=True
)

# ---------------- VOLATILITY ----------------
st.markdown("## ⚠️ High Volatility States")

volatility = (
    daily.groupby("state_clean")["deviation_pct"]
    .std()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_vol = px.bar(
    volatility,
    x="deviation_pct",
    y="state_clean",
    orientation="h",
    title="State-wise Reporting Volatility"
)

st.plotly_chart(fig_vol, use_container_width=True)

# ---------------- DATA INTEGRITY SCORE ----------------
st.markdown("## 🧬 State Data Integrity Score")

integrity = daily.groupby("state_clean").agg(
    anomaly_rate=("anomaly", "mean"),
    avg_deviation=("deviation_pct", "mean")
).reset_index()

integrity["integrity_score"] = (
    100 - (integrity["anomaly_rate"] * 100)
).clip(0, 100)

fig_integrity = px.bar(
    integrity.sort_values("integrity_score"),
    x="integrity_score",
    y="state_clean",
    orientation="h",
    title="UIDAI State Data Integrity Index"
)

st.plotly_chart(fig_integrity, use_container_width=True)

# ---------------- EXECUTIVE INSIGHTS ----------------
st.markdown("## 📝 Executive Insights for UIDAI")

st.markdown(f"""
• {df['state_original'].nunique()} raw state naming formats detected  
• {variants[variants['name_variants'] > 1].shape[0]} states required normalization  
• {daily['anomaly'].sum()} anomalous enrolment days identified  
• Highest volatility observed in **{volatility.iloc[0]['state_clean']}**

### Recommendations:
✔ Enforce state naming standards at source  
✔ Deploy mobile enrolment units in underperforming states  
✔ Flag high-volatility regions for audit  
✔ Use baseline deviation alerts for real-time monitoring  
""")

st.success("✅ UIDAI Data Intelligence Platform Loaded Successfully")
