import streamlit as st
from src.data_loader import load_chs_data
import plotly.express as px

st.set_page_config(
    page_title="Housing Affordability Stress – Canada (CHS 2022)",
    layout="wide"
)

st.title("🏠 Housing Affordability Stress in Canada")
st.caption("Source: Statistics Canada – Canadian Housing Survey 2022 (PUMF)")

df = load_chs_data()

st.metric("Total Households (Unweighted)", f"{df.shape[0]:,}")

st.write("Preview of CHS 2022 Data")
st.dataframe(df.head())

st.sidebar.header("Filter Households")

tenure = st.sidebar.selectbox(
    "Housing Tenure",
    options=["All"] + sorted(df["TENURE"].dropna().unique().tolist())
)

if tenure != "All":
    df = df[df["TENURE"] == tenure]


pchn_dist = (
    df["PCHN"]
    .value_counts(dropna=False)
    .reset_index()
    .rename(columns={"index": "PCHN", "PCHN": "Count"})
)

fig = px.bar(
    pchn_dist,
    x="PCHN",
    y="Count",
    title="Core Housing Need Status (Unweighted)"
)

st.plotly_chart(fig, use_container_width=True)

