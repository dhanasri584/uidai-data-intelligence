import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(
    page_title="UIDAI Data Intelligence Platform",
    layout="wide"
)

st.title("🆔 UIDAI Aadhaar Enrolment Data Intelligence Platform")
st.caption("State Standardization • Baseline Analysis • Anomaly Detection • Geo-Visualisation")

# ===================== FILE UPLOAD =====================
uploaded_files = st.file_uploader(
    "📤 Upload UIDAI Aadhaar Enrolment CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("Please upload all Aadhaar enrolment CSV files to proceed.")
    st.stop()

@st.cache_data
def load_data(files):
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = load_data(uploaded_files)

# ===================== GEOJSON =====================
@st.cache_data
def load_india_geojson():
    url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    return requests.get(url).json()

india_geojson = load_india_geojson()

# ===================== STATE STANDARDIZATION =====================
state_mapping = {
    "west bengal": "West Bengal",
    "west  bengal": "West Bengal",
    "west bangal": "West Bengal",
    "westbengal": "West Bengal",
    "andhra pradesh": "Andhra Pradesh",
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "dadra and nagar haveli": "Dadra & Nagar Haveli",
    "daman and diu": "Daman & Diu"
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

# ===================== DATE & ENROLMENTS =====================
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["total_enrolments"] = (
    df["age_0_5"] +
    df["age_5_17"] +
    df["age_18_greater"]
)

df = df.dropna(subset=["date"])

# ===================== BASELINE SERIES =====================
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

daily["deviation"] = abs(daily["total_enrolments"] - daily["baseline"]) / daily["baseline"]
daily["anomaly"] = daily["deviation"] > 0.5

latest = daily.sort_values("date").groupby("state_clean").tail(1)

# ===================== INDIA MAP =====================
st.markdown("## 🗺️ India Aadhaar Enrolment Heatmap")

latest["state_match"] = latest["state_clean"].str.upper()

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

# ===================== STANDARDIZATION INSIGHT =====================
st.markdown("## 🧹 State Name Anomalies")

variants = (
    df.groupby("state_clean")["state_original"]
    .nunique()
    .reset_index(name="variant_count")
)

st.dataframe(variants[variants["variant_count"] > 1], use_container_width=True)

# ===================== VOLATILITY =====================
st.markdown("## ⚠️ Reporting Volatility")

volatility = (
    daily.groupby("state_clean")["deviation"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_vol = px.bar(
    volatility,
    x="deviation",
    y="state_clean",
    orientation="h",
    title="High Volatility States"
)

st.plotly_chart(fig_vol, use_container_width=True)

# ===================== UIDAI RECOMMENDATIONS =====================
st.markdown("## 📝 UIDAI Actionable Insights")

st.markdown("""
✔ Auto-standardize state names at data entry  
✔ Flag enrolment drops beyond baseline deviation  
✔ Prioritize audits in high volatility states  
✔ Use baseline trends for staff & kit allocation  
""")

st.success("✅ UIDAI Data Intelligence Platform Ready")
