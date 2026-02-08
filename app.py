import streamlit as st
#import plotly.express as px
from src.data_loader import load_chs_data

st.set_page_config(
    page_title="Housing Affordability Stress – Canada (CHS 2022)",
    layout="wide"
)

st.title("🏠 Housing Affordability Stress in Canada")
st.caption("Source: Statistics Canada – Canadian Housing Survey 2022 (PUMF)")

# Load data
df = load_chs_data()

st.metric("Total Households (Unweighted)", f"{df.shape[0]:,}")

st.write("Preview of CHS 2022 Data")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filter Households")

if "TENURE" in df.columns:
    tenure_options = ["All"] + sorted(df["TENURE"].dropna().unique().tolist())
    tenure = st.sidebar.selectbox("Housing Tenure", tenure_options)

    if tenure != "All":
        df = df[df["TENURE"] == tenure]
else:
    st.sidebar.warning("TENURE variable not found in dataset.")

# ---------------- PCHN Distribution ----------------
st.subheader("Core Housing Need Status")

if "PCHN" in df.columns and not df["PCHN"].dropna().empty:
    pchn_dist = (
        df["PCHN"]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "PCHN", "PCHN": "Count"})
    )

    # fig = px.bar(
    #     pchn_dist,
    #     x="PCHN",
    #     y="Count",
    #     title="Core Housing Need Status (Unweighted)",
    #     labels={"PCHN": "PCHN Category", "Count": "Number of Households"}
    # )

    # st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("No PCHN data available for the selected filter.")

# ---------------- Disclaimer ----------------
st.info(
    "All results shown are unweighted. Survey weights (PFWEIGHT) "
    "will be applied in later stages of the analysis."
)
