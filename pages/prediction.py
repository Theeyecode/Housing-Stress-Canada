import streamlit as st
from src.data_loader import load_scored_data
import pandas as pd
import plotly.express as px

st.title("Prediction Results")

df = load_scored_data()

st.success("Scored dataset loaded successfully!")

st.write("Preview:")
st.dataframe(df.head())

# MODEL INFORMATION PANEL

with st.expander("Model Information", expanded=False):

    st.markdown("### Model Details")

    col1, col2 = st.columns(2)

    col1.markdown("**Model Type:** Logistic Regression")
    col1.markdown("**Decision Threshold:** 0.30")

    col2.markdown("**Training Dataset:** CHS 2022 PUMF")
    col2.markdown("**Prediction Target:** Core Housing Need (PCHN)")

    st.markdown("---")
    st.markdown("### Performance Metrics (Validation Set)")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric("ROC-AUC", "0.90")
    metric_col2.metric("PR-AUC", "0.58")
    metric_col3.metric("Recall", "0.75")
    metric_col4.metric("Precision", "0.53")

    st.caption(
        "Metrics are computed during model validation. "
        "No model training or tuning occurs within this dashboard."
    )

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

# Confusion Matrix Components (USE FILTERED DATA)

TP = ((filtered_df["PCHN_Clean"] == 1) &
      (filtered_df["Predicted_Housing_Stress"] == 1)).sum()

FP = ((filtered_df["PCHN_Clean"] == 0) &
      (filtered_df["Predicted_Housing_Stress"] == 1)).sum()

TN = ((filtered_df["PCHN_Clean"] == 0) &
      (filtered_df["Predicted_Housing_Stress"] == 0)).sum()

FN = ((filtered_df["PCHN_Clean"] == 1) &
      (filtered_df["Predicted_Housing_Stress"] == 0)).sum()

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


# GROUP BREAKDOWN VIEWS

st.divider()
st.header("Group Breakdown Views")

def flagged_rate_by_group(df, group_col):
    result = (
        df
        .groupby(group_col)["Predicted_Housing_Stress"]
        .mean()
        .reset_index()
    )
    result["Percent_Flagged"] = result["Predicted_Housing_Stress"] * 100
    return result.sort_values("Percent_Flagged", ascending=False)


def create_interactive_bar(df, group_col, title):
    fig = px.bar(
        df,
        x="Percent_Flagged",
        y=group_col,
        orientation="h",
        text="Percent_Flagged",
        color="Percent_Flagged",
        color_continuous_scale="Blues"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        title=title,
        xaxis_title="% Flagged (Threshold = 0.30)",
        yaxis_title="",
        coloraxis_showscale=False,
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_xaxes(range=[0, 100])

    return fig

# % Flagged by Province

if "Province_Name" in filtered_df.columns:
    province_breakdown = flagged_rate_by_group(filtered_df, "Province_Name")
    fig_province = create_interactive_bar(
        province_breakdown,
        "Province_Name",
        "% Flagged by Province"
    )
    st.plotly_chart(fig_province, width="stretch")

# % Flagged by Tenure

if "Tenure_Group" in filtered_df.columns:
    tenure_breakdown = flagged_rate_by_group(filtered_df, "Tenure_Group")
    fig_tenure = create_interactive_bar(
        tenure_breakdown,
        "Tenure_Group",
        "% Flagged by Tenure Group"
    )
    st.plotly_chart(fig_tenure, width="stretch")

# LABEL MAPPINGS
PSTIR_LABELS = {
    1: "< 30% Income on Shelter",
    2: "30%–49% Income on Shelter",
    3: "50%+ Income on Shelter"
}

PHTYPE_LABELS = {
    1: "One-person Household",
    2: "Couple without Children",
    3: "Couple with Children",
    4: "Lone-parent Family",
    5: "Other Family Household",
    6: "Multiple-family Household"
}

def flagged_rate_by_group(df, group_col, label_map=None, order=None):
    result = (
        df
        .groupby(group_col)["Predicted_Housing_Stress"]
        .mean()
        .reset_index()
    )

    result["Percent_Flagged"] = result["Predicted_Housing_Stress"] * 100

    # Apply label mapping if provided
    if label_map:
        result[group_col] = result[group_col].map(label_map)

    # Enforce logical ordering if provided
    if order:
        result[group_col] = pd.Categorical(
            result[group_col],
            categories=order,
            ordered=True
        )
        result = result.sort_values(group_col)
    else:
        result = result.sort_values("Percent_Flagged", ascending=False)

    return result

# % Flagged by Income Group (PSTIR_GR_Clean)

if "PSTIR_GR_Clean" in filtered_df.columns:

    stir_order = [
        "< 30% Income on Shelter",
        "30%–49% Income on Shelter",
        "50%+ Income on Shelter"
    ]

    stir_breakdown = flagged_rate_by_group(
        filtered_df,
        "PSTIR_GR_Clean",
        label_map=PSTIR_LABELS,
        order=stir_order
    )

    fig_stir = create_interactive_bar(
        stir_breakdown,
        "PSTIR_GR_Clean",
        "% Flagged by Shelter Cost-to-Income Group"
    )

    st.plotly_chart(fig_stir, width="stretch")


# % Flagged by Household Type (PHTYPE_Clean)

if "PHTYPE_Clean" in filtered_df.columns:

    hh_order = [
        "One-person Household",
        "Lone-parent Family",
        "Couple without Children",
        "Couple with Children",
        "Other Family Household",
        "Multiple-family Household"
    ]

    hh_breakdown = flagged_rate_by_group(
        filtered_df,
        "PHTYPE_Clean",
        label_map=PHTYPE_LABELS,
        order=hh_order
    )

    fig_hh = create_interactive_bar(
        hh_breakdown,
        "PHTYPE_Clean",
        "% Flagged by Household Composition"
    )

    st.plotly_chart(fig_hh, width="stretch")