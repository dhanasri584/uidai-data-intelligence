import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="UIDAI Data Intelligence Studio", layout="wide")

st.title("🆔 UIDAI Enrolment Data Intelligence Studio")
st.caption("Standardization • Anomalies • Baseline Trends • Geo Insights")

@st.cache_data
def load_data():
    files = [
        "api_data_aadhar_enrolment_0_500000.csv",
        "api_data_aadhar_enrolment_500000_1000000.csv",
        "api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = load_data()

# Auto detect columns
state_col = [c for c in df.columns if "state" in c.lower()][0]
date_col = [c for c in df.columns if "date" in c.lower()][0]
age_cols = [c for c in df.columns if "age" in c.lower()]

df["total_enrolments"] = df[age_cols].sum(axis=1)

# Standardization
state_map = {
    "orissa": "Odisha",
    "odisa": "Odisha",
    "west bangal": "West Bengal",
    "westbengal": "West Bengal"
}

df["state_original"] = df[state_col]
df["state_clean"] = (
    df[state_col].astype(str)
    .str.lower().str.strip()
    .replace(state_map).str.title()
)

df["date"] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=["date"])

# Daily aggregation
daily = df.groupby(["state_clean", "date"])["total_enrolments"].sum().reset_index()

daily["baseline"] = (
    daily.groupby("state_clean")["total_enrolments"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Latest snapshot for map
latest = (
    daily.sort_values("date")
    .groupby("state_clean")
    .tail(1)
)

# UI Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Overview",
    "⚠️ Standardization",
    "📈 Baseline Trends",
    "🗺️ India Map"
])

with tab1:
    st.metric("Total Records", len(df))
    st.metric("States (Raw)", df["state_original"].nunique())
    st.metric("States (Clean)", df["state_clean"].nunique())
    st.dataframe(df.head())

with tab2:
    variants = (
        df.groupby("state_clean")["state_original"]
        .nunique().reset_index(name="variants")
    )
    st.dataframe(variants[variants["variants"] > 1])

    pie = pd.DataFrame({
        "Type": ["Clean", "Standardized"],
        "Count": [
            (df["state_original"] == df["state_clean"]).sum(),
            (df["state_original"] != df["state_clean"]).sum()
        ]
    })

    fig = px.pie(pie, names="Type", values="Count",
                 title="State Standardization Impact")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    state = st.selectbox("Select State", daily["state_clean"].unique())
    fig = px.line(
        daily[daily["state_clean"] == state],
        x="date", y=["total_enrolments", "baseline"],
        title=f"Actual vs Baseline — {state}"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.choropleth(
        latest,
        locations="state_clean",
        locationmode="geojson-id",
        color="total_enrolments",
        title="Latest Aadhaar Enrolment by State (Baseline Cleaned)",
        scope="asia"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.success("✅ Live Prototype Ready — Shareable Link Generated")
