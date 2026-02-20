import streamlit as st
from src.data_loader import load_scored_data
import pandas as pd

st.title("Prediction Results")

df = load_scored_data()

st.success("Scored dataset loaded successfully!")

st.write("Preview:")
st.dataframe(df.head())

# SIDEBAR FILTERS
st.sidebar.header("Filter Households")

province_filter = st.sidebar.multiselect(
    "Province",
    options=sorted(df["Province_Name"].dropna().unique()),
    default=None
)

tenure_filter = st.sidebar.multiselect(
    "Tenure Group",
    options=sorted(df["Tenure_Group"].dropna().unique()),
    default=None
)

risk_filter = st.sidebar.multiselect(
    "Risk Band",
    options=sorted(df["Risk_Band"].dropna().unique()),
    default=None
)

# Apply filters
filtered_df = df.copy()

if province_filter:
    filtered_df = filtered_df[filtered_df["Province_Name"].isin(province_filter)]

if tenure_filter:
    filtered_df = filtered_df[filtered_df["Tenure_Group"].isin(tenure_filter)]

if risk_filter:
    filtered_df = filtered_df[filtered_df["Risk_Band"].isin(risk_filter)]

st.caption(f"Filtered observations: {len(filtered_df):,}")


# METRICS

if len(filtered_df) > 0:

    actual_rate = filtered_df["PCHN_Clean"].mean()
    predicted_rate = filtered_df["Predicted_Housing_Stress"].mean()

    col1, col2 = st.columns(2)

    col1.metric(
        "Actual Housing Stress Prevalence",
        f"{actual_rate:.2%}"
    )

    col2.metric(
        "Predicted Flag Rate (Threshold = 0.30)",
        f"{predicted_rate:.2%}"
    )

    st.divider()

# Confusion Matrix Components

TP = ((df["PCHN_Clean"] == 1) & (df["Predicted_Housing_Stress"] == 1)).sum()
FP = ((df["PCHN_Clean"] == 0) & (df["Predicted_Housing_Stress"] == 1)).sum()
TN = ((df["PCHN_Clean"] == 0) & (df["Predicted_Housing_Stress"] == 0)).sum()
FN = ((df["PCHN_Clean"] == 1) & (df["Predicted_Housing_Stress"] == 0)).sum()

st.subheader("Confusion Matrix (Counts)")

cm_df = pd.DataFrame(
    {
        "Predicted Stress (1)": [TP, FP],
        "Predicted No Stress (0)": [FN, TN]
    },
    index=["Actual Stress (1)", "Actual No Stress (0)"]
)

st.dataframe(cm_df)

st.caption(
    "TP = True Positive | FP = False Positive | "
    "TN = True Negative | FN = False Negative"
)