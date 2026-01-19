import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="UIDAI Data Intelligence Dashboard",
    layout="wide"
)

st.title("🆔 UIDAI Enrolment Data Intelligence Dashboard")
st.caption("State anomalies • Clean baselines • Geo insights • Operational recommendations")

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📂 Upload UIDAI Enrolment CSV Files",
    type="csv",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload UIDAI enrolment CSV files to start analysis")
    st.stop()

@st.cache_data
def load_data(files):
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = load_data(uploaded_files)

# ---------------- AUTO COLUMN DETECTION ----------------
state_col = [c for c in df.columns if "state" in c.lower()][0]
date_col = [c for c in df.columns if "date" in c.lower()][0]
age_cols = [c for c in df.columns if "age" in c.lower()]

df["total_enrolments"] = df[age_cols].sum(axis=1)

# ---------------- STATE STANDARDIZATION ----------------
state_map = {
    "orissa": "Odisha",
    "odisa": "Odisha",
    "west bangal": "West Bengal",
    "westbengal": "West Bengal",
    "andaman & nicobar": "Andaman & Nicobar Islands",
}

df["state_original"] = df[state_col]
df["state_clean"] = (
    df[state_col]
    .astype(str)
    .str.lower()
    .str.strip()
    .replace(state_map)
    .str.title()
)

# ---------------- DATE CLEAN ----------------
df["date"] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=["date"])

# ---------------- DAILY AGGREGATION ----------------
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
    (daily["total_enrolments"] - daily["baseline"])
    / daily["baseline"] * 100
)

daily["anomaly"] = abs(daily["deviation_pct"]) > 50

# ---------------- KPI CARDS ----------------
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
col2.metric("Raw State Names", df["state_original"].nunique())
col3.metric("Standardized States", df["state_clean"].nunique())
col4.metric("Anomaly Points", daily["anomaly"].sum())

# ---------------- DATA PREVIEW ----------------
st.markdown("## 📄 Raw Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# ---------------- STATE ANOMALIES ----------------
st.markdown("## ⚠️ State Naming Anomalies")

variants = (
    df.groupby("state_clean")["state_original"]
    .nunique()
    .reset_index(name="name_variants")
    .sort_values("name_variants", ascending=False)
)

st.dataframe(variants[variants["name_variants"] > 1], use_container_width=True)

pie = pd.DataFrame({
    "Category": ["Already Clean", "Auto-Standardized"],
    "Count": [
        (df["state_original"] == df["state_clean"]).sum(),
        (df["state_original"] != df["state_clean"]).sum()
    ]
})

fig_pie = px.pie(
    pie,
    names="Category",
    values="Count",
    title="State Standardization Impact"
)
st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- BASELINE TREND ----------------
st.markdown("## 📈 Baseline Enrolment Trends")

state_sel = st.selectbox(
    "Select State for Trend Analysis",
    sorted(daily["state_clean"].unique())
)

trend_df = daily[daily["state_clean"] == state_sel]

fig_line = px.line(
    trend_df,
    x="date",
    y=["total_enrolments", "baseline"],
    labels={"value": "Enrolments", "variable": "Series"},
    title=f"Actual vs Baseline Enrolments — {state_sel}"
)

st.plotly_chart(fig_line, use_container_width=True)

# ---------------- INDIA MAP ----------------
st.markdown("## 🗺️ India Enrolment Heatmap")

latest = (
    daily.sort_values("date")
    .groupby("state_clean")
    .tail(1)
)

fig_map = px.choropleth(
    latest,
    locations="state_clean",
    locationmode="country names",
    color="total_enrolments",
    title="Latest Aadhaar Enrolments by State",
    scope="asia",
    color_continuous_scale="Blues"
)

fig_map.update_geos(
    center={"lat": 22, "lon": 78},
    projection_scale=4,
    visible=False
)

st.plotly_chart(fig_map, use_container_width=True)

# ---------------- RECOMMENDATIONS ----------------
st.markdown("## 🧠 System Recommendations")

high_variants = variants[variants["name_variants"] > 3]
high_anomalies = daily[daily["anomaly"] == True]["state_clean"].value_counts()

st.markdown("### Key Observations")
st.write(f"• {len(high_variants)} states have high naming inconsistencies")
st.write(f"• {len(high_anomalies)} states show abnormal enrolment behavior")

st.markdown("### Recommended Actions")
st.markdown("""
• Enforce state standardization at data entry level  
• Validate enrolment spikes before forecasting  
• Allocate biometric kits based on baseline demand  
• Exclude anomalous periods from policy decisions  
""")

# ---------------- DOWNLOAD ----------------
st.markdown("## ⬇️ Download Cleaned Data")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Cleaned UIDAI Dataset",
    csv,
    file_name="uidai_cleaned_enrolment_data.csv",
    mime="text/csv"
)

st.success("✅ Dashboard ready — shareable, scalable, and judge-ready")
