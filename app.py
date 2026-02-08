import streamlit as st
#import plotly.express as px
from src.data_loader import load_chs_data

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Housing Affordability Stress – Canada (CHS 2022)",
    layout="wide"
)

st.title("🏠 Housing Affordability Stress in Canada")
st.caption("Source: Statistics Canada – Canadian Housing Survey 2022 (PUMF)")

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = load_chs_data()

st.metric("Total Households (Unweighted)", f"{df.shape[0]:,}")

st.write("Preview of CHS 2022 Data")
st.dataframe(df.head())

# -------------------------------------------------
# Tenure setup (PDCT_05)
# -------------------------------------------------
TENURE_COL = "PDCT_05"

TENURE_LABELS = {
    1: "Owner",
    2: "Renter"
}

# -------------------------------------------------
# Sidebar filter (Step 1 & 3)
# -------------------------------------------------
st.sidebar.header("Filter Households")

if TENURE_COL in df.columns:
    tenure_options = ["All"] + list(TENURE_LABELS.values())
    tenure = st.sidebar.selectbox("Housing Tenure", tenure_options)

    if tenure != "All":
        tenure_code = [k for k, v in TENURE_LABELS.items() if v == tenure][0]
        df = df[df[TENURE_COL] == tenure_code]

    # Step 2: label mapping (display only)
    df["Tenure_Label"] = df[TENURE_COL].map(TENURE_LABELS)

else:
    st.sidebar.warning("Tenure variable (PDCT_05) not found in dataset.")

# -------------------------------------------------
# Step 4: Sanity check (temporary but useful)
# -------------------------------------------------
st.subheader("Tenure Distribution (Sanity Check)")
st.write(df[TENURE_COL].value_counts(dropna=False))

# -------------------------------------------------
# PCHN Distribution
# -------------------------------------------------
st.subheader("Core Housing Need Status")

if "PCHN" in df.columns and not df["PCHN"].dropna().empty:
    pchn_dist = (
        df["PCHN"]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "PCHN", "PCHN": "Count"})
    )

#     fig = px.bar(
#         pchn_dist,
#         x="PCHN",
#         y="Count",
#         title="Core Housing Need Status (Unweighted)",
#         labels={"PCHN": "PCHN Category", "Count": "Number of Households"}
#     )

#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("No PCHN data available for the selected filter.")

# -------------------------------------------------
# Disclaimer
# -------------------------------------------------
st.info(
    "All results shown are unweighted. Survey weights (PFWEIGHT) "
    "will be applied in later stages of the analysis."
)
