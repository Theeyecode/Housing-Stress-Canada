import streamlit as st
from src.data_loader import load_scored_data
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

st.set_page_config(page_title="Prediction Results", layout="wide")
st.title("Prediction Results")
st.caption("Scored outputs from the finalized logistic regression model (no training inside Streamlit).")

# artifact paths 
BASE_DIR = Path(__file__).resolve().parent.parent  # /pages -> project root

def resolve_artifact(filename: str) -> Path:
    candidates = [
        BASE_DIR / "artifacts" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / filename
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # for error messaging

MODEL_INFO_PATH = resolve_artifact("model_info.json")
TOP10_COEF_PATH = resolve_artifact("top10_coefficients.csv")
ROBUSTNESS_PATH = resolve_artifact("pstir_robustness.csv")

# Load scored dataset
df = load_scored_data()
st.success("Scored dataset loaded successfully!")

with st.expander("Preview (first 50 rows)", expanded=False):
    st.dataframe(df.head(50), width="stretch")

# Sidebar filters
st.sidebar.header("Filter Households")

def safe_multiselect(label, col):
    if col not in df.columns:
        st.sidebar.warning(f"Missing column: {col}")
        return []
    options = sorted(df[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, options=options, default=[])

province_filter = safe_multiselect("Province", "Province_Name")
tenure_filter = safe_multiselect("Tenure Group", "Tenure_Group")
# risk_filter = safe_multiselect("Risk Band", "Risk_Band")

filtered_df = df.copy()
if province_filter and "Province_Name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Province_Name"].isin(province_filter)]
if tenure_filter and "Tenure_Group" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Tenure_Group"].isin(tenure_filter)]
# if risk_filter and "Risk_Band" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["Risk_Band"].isin(risk_filter)]

st.caption(f"Filtered observations: {len(filtered_df):,} / {len(df):,}")

# Top KPI metrics
st.subheader("Prediction Summary (Filtered)")

required_for_metrics = ["PCHN_Clean", "Predicted_Housing_Stress"]
missing_metrics_cols = [c for c in required_for_metrics if c not in filtered_df.columns]

if missing_metrics_cols:
    st.warning(f"Cannot compute KPIs — missing columns: {missing_metrics_cols}")
else:
    if len(filtered_df) > 0:
        # Note: this assumes binary 0/1 coding for both columns
        actual_rate = filtered_df["PCHN_Clean"].mean()
        predicted_rate = filtered_df["Predicted_Housing_Stress"].mean()

        c1, c2 = st.columns(2)
        c1.metric("Actual Housing Stress Prevalence", f"{actual_rate:.2%}")
        c2.metric("Predicted Flag Rate (Threshold = 0.30)", f"{predicted_rate:.2%}")
    else:
        st.info("No rows after filtering. Adjust filters to see metrics.")

st.divider()

# Model Info Panel (JSON)
with st.expander("Model Information", expanded=False):
    if MODEL_INFO_PATH.exists():
        try:
            model_info = json.loads(MODEL_INFO_PATH.read_text())
            st.markdown("### Model Details")

            c1, c2 = st.columns(2)
            c1.markdown(f"**Model Type:** {model_info.get('model_type', '—')}")
            c1.markdown(f"**Decision Threshold:** {model_info.get('threshold', '—')}")

            st.markdown("---")
            st.markdown("### Performance Metrics (Validation Set)")

            m1, m2, m3, m4 = st.columns(4)
            roc = model_info.get("roc_auc", None)
            pr = model_info.get("pr_auc", None)
            rec = model_info.get("recall_at_threshold", None)
            prec = model_info.get("precision_at_threshold", None)

            m1.metric("ROC-AUC", f"{roc:.3f}" if isinstance(roc, (int, float)) else "—")
            m2.metric("PR-AUC", f"{pr:.3f}" if isinstance(pr, (int, float)) else "—")
            m3.metric("Recall", f"{rec:.3f}" if isinstance(rec, (int, float)) else "—")
            m4.metric("Precision", f"{prec:.3f}" if isinstance(prec, (int, float)) else "—")

            st.caption("Metrics are loaded from stored training artifacts. No retraining occurs in Streamlit.")
        except Exception:
            st.error("model_info.json exists but could not be parsed.")
    else:
        st.warning(f"Model info file not found: {MODEL_INFO_PATH}")

st.divider()

# Top Drivers (CSV)
with st.expander("Top Model Drivers (Associations Only)", expanded=False):
    if TOP10_COEF_PATH.exists():
        try:
            coef_df = pd.read_csv(TOP10_COEF_PATH)

            # Expected: feature, coefficient, abs_coef (you have these)
            needed = ["feature", "coefficient"]
            if any(c not in coef_df.columns for c in needed):
                st.warning(f"top10_coefficients.csv missing expected columns: {needed}")
            else:
                if "abs_coef" not in coef_df.columns:
                    coef_df["abs_coef"] = coef_df["coefficient"].abs()

                coef_df = coef_df.sort_values("abs_coef", ascending=False).head(10).copy()

                coef_df["direction"] = coef_df["coefficient"].apply(
                    lambda x: "+ Higher likelihood of flag" if x > 0 else "- Lower likelihood of flag"
                )

                st.dataframe(
                    coef_df[["feature", "coefficient", "direction", "abs_coef"]],
                    width="stretch"
                )

                st.caption(
                    "Coefficients are log-odds from logistic regression and represent statistical associations only "
                    "(no causal claims)."
                )
        except Exception:
            st.error("Could not read top10_coefficients.csv.")
    else:
        st.warning(f"Top coefficients file not found: {TOP10_COEF_PATH}")

st.divider()

# Robustness display (PSTIR)
st.header("Validation View: Shelter Cost Burden Gradient (Robustness)")

if ROBUSTNESS_PATH.exists():
    try:
        robustness_df = pd.read_csv(ROBUSTNESS_PATH)

        # Expect columns: PSTIR_GR_Clean, flagged_rate, true_stress_rate
        required = ["PSTIR_GR_Clean", "flagged_rate", "true_stress_rate"]
        missing_rb = [c for c in required if c not in robustness_df.columns]

        if missing_rb:
            st.warning(f"pstir_robustness.csv missing columns: {missing_rb}")
        else:
            robustness_df = robustness_df.sort_values("PSTIR_GR_Clean")

            fig_validation = px.line(
                robustness_df,
                x="PSTIR_GR_Clean",
                y=["flagged_rate", "true_stress_rate"],
                markers=True
            )
            fig_validation.update_layout(
                template="plotly_white",
                height=420,
                xaxis_title="Shelter Cost-to-Income Group (PSTIR_GR_Clean)",
                yaxis_title="Rate"
            )
            fig_validation.update_yaxes(range=[0, 1])  # rates 0-1
            st.plotly_chart(fig_validation, width="stretch")

            st.caption(
                "Robustness artifact compares predicted flag rate vs true stress rate across PSTIR groups. "
                "A monotonic pattern supports economic consistency."
            )
    except Exception:
        st.error("pstir_robustness.csv exists but could not be read.")
else:
    st.warning(f"Robustness file not found: {ROBUSTNESS_PATH}")

st.divider()

# Confusion matrix (UNWEIGHTED counts, filtered)
st.subheader("Confusion Matrix (Counts, Filtered)")

if all(c in filtered_df.columns for c in ["PCHN_Clean", "Predicted_Housing_Stress"]):
    if len(filtered_df) == 0:
        st.info("No rows after filtering to compute confusion matrix.")
    else:
        TP = ((filtered_df["PCHN_Clean"] == 1) & (filtered_df["Predicted_Housing_Stress"] == 1)).sum()
        FP = ((filtered_df["PCHN_Clean"] == 0) & (filtered_df["Predicted_Housing_Stress"] == 1)).sum()
        TN = ((filtered_df["PCHN_Clean"] == 0) & (filtered_df["Predicted_Housing_Stress"] == 0)).sum()
        FN = ((filtered_df["PCHN_Clean"] == 1) & (filtered_df["Predicted_Housing_Stress"] == 0)).sum()

        cm_df = pd.DataFrame(
            {
                "Predicted Stress (1)": [TP, FP],
                "Predicted No Stress (0)": [FN, TN]
            },
            index=["Actual Stress (1)", "Actual No Stress (0)"]
        )

        st.dataframe(cm_df, width="stretch")
        st.caption("Counts are unweighted. If you want a weighted confusion matrix, we can add PFWEIGHT support.")
else:
    st.warning("Missing PCHN_Clean or Predicted_Housing_Stress, cannot compute confusion matrix.")

st.divider()

# Group Breakdown Views (interactive bars)
st.header("Group Breakdown Views (% Flagged, Filtered)")

def flagged_rate_by_group(df_in: pd.DataFrame, group_col: str, label_map=None, order=None) -> pd.DataFrame:
    if group_col not in df_in.columns or "Predicted_Housing_Stress" not in df_in.columns:
        return pd.DataFrame()
    tmp = df_in[[group_col, "Predicted_Housing_Stress"]].dropna()
    if tmp.empty:
        return pd.DataFrame()
    out = tmp.groupby(group_col)["Predicted_Housing_Stress"].mean().reset_index()
    out["Percent_Flagged"] = out["Predicted_Housing_Stress"] * 100

    if label_map is not None:
        out[group_col] = out[group_col].map(label_map).fillna(out[group_col].astype(str))

    if order is not None:
        out[group_col] = pd.Categorical(out[group_col], categories=order, ordered=True)
        out = out.sort_values(group_col)
    else:
        out = out.sort_values("Percent_Flagged", ascending=False)

    return out

def create_interactive_bar(df_plot: pd.DataFrame, group_col: str, title: str):
    fig = px.bar(
        df_plot,
        x="Percent_Flagged",
        y=group_col,
        orientation="h",
        text="Percent_Flagged",
        color="Percent_Flagged",
        color_continuous_scale="Blues"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        title=title,
        xaxis_title="% Flagged (Threshold = 0.30)",
        yaxis_title="",
        coloraxis_showscale=False,
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=55, b=30)
    )
    fig.update_xaxes(range=[0, 100])
    return fig

# Province
if "Province_Name" in filtered_df.columns:
    prov_tbl = flagged_rate_by_group(filtered_df, "Province_Name")
    if not prov_tbl.empty:
        st.plotly_chart(create_interactive_bar(prov_tbl, "Province_Name", "% Flagged by Province"), width="stretch")

# Tenure
if "Tenure_Group" in filtered_df.columns:
    ten_tbl = flagged_rate_by_group(filtered_df, "Tenure_Group")
    if not ten_tbl.empty:
        st.plotly_chart(create_interactive_bar(ten_tbl, "Tenure_Group", "% Flagged by Tenure Group"), width="stretch")

# PSTIR labels + order
PSTIR_LABELS = {
    1: "< 30% Income on Shelter",
    2: "30%–49% Income on Shelter",
    3: "50%+ Income on Shelter"
}
PSTIR_ORDER = [
    "< 30% Income on Shelter",
    "30%–49% Income on Shelter",
    "50%+ Income on Shelter"
]

if "PSTIR_GR_Clean" in filtered_df.columns:
    stir_tbl = flagged_rate_by_group(filtered_df, "PSTIR_GR_Clean", label_map=PSTIR_LABELS, order=PSTIR_ORDER)
    if not stir_tbl.empty:
        st.plotly_chart(create_interactive_bar(stir_tbl, "PSTIR_GR_Clean", "% Flagged by Shelter Cost-to-Income Group"), width="stretch")

# Household type labels + order
PHTYPE_LABELS = {
    1: "One-person Household",
    2: "Couple without Children",
    3: "Couple with Children",
    4: "Lone-parent Family",
    5: "Other Family Household",
    6: "Multiple-family Household"
}
PHTYPE_ORDER = [
    "One-person Household",
    "Lone-parent Family",
    "Couple without Children",
    "Couple with Children",
    "Other Family Household",
    "Multiple-family Household"
]

if "PHTYPE_Clean" in filtered_df.columns:
    hh_tbl = flagged_rate_by_group(filtered_df, "PHTYPE_Clean", label_map=PHTYPE_LABELS, order=PHTYPE_ORDER)
    if not hh_tbl.empty:
        st.plotly_chart(create_interactive_bar(hh_tbl, "PHTYPE_Clean", "% Flagged by Household Composition"), width="stretch")

