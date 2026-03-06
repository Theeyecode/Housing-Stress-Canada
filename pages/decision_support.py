import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page setup
st.set_page_config(page_title="Decision Support", layout="wide")
st.title("Decision Support")
st.caption(
    "Policy-oriented targeting and planning using precomputed dashboard artifacts "
    "(no training or threshold tuning inside Streamlit)."
)

BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_path(filename: str) -> Path:
    """
    Try common locations in a typical repo structure.
    """
    candidates = [
        BASE_DIR / "artifacts" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  

# Artifact paths 
HOUSEHOLDS_PATH = resolve_path("dashboard_households.csv")
POLICY_MATRIX_PATH = resolve_path("dashboard_policy_matrix.csv")
TRADEOFF_PATH = resolve_path("dashboard_threshold_tradeoff.csv")
STRATEGY_CURVES_PATH = resolve_path("dashboard_strategy_curves.csv")
PSTIR_ROBUST_PATH = resolve_path("pstir_robustness.csv")  # optional

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# Load required artifacts
missing = []
for p in [HOUSEHOLDS_PATH, POLICY_MATRIX_PATH, TRADEOFF_PATH, STRATEGY_CURVES_PATH]:
    if not p.exists():
        missing.append(str(p))

if missing:
    st.error(
        "Missing decision-support artifact files. Expected one of these paths to exist:\n\n- "
        + "\n- ".join(missing)
    )
    st.stop()

households = load_csv(HOUSEHOLDS_PATH)
policy_matrix = load_csv(POLICY_MATRIX_PATH)
tradeoff = load_csv(TRADEOFF_PATH)
strategy_curves = load_csv(STRATEGY_CURVES_PATH)

pstir_robust = load_csv(PSTIR_ROBUST_PATH) if PSTIR_ROBUST_PATH.exists() else None

# households expected columns (from your updated file)
REQ_HH = [
    "Province_Name",
    "Tenure_Group",
    "Risk_Band",
    "Risk_Segment",
    "Predicted_Risk",
    "Predicted_Stress",
    "Actual_Stress",
    "Household_Weight",
    "Income_Quintile",
]
missing_cols = [c for c in REQ_HH if c not in households.columns]
if missing_cols:
    st.warning(f"`dashboard_households.csv` is missing expected columns: {missing_cols}")

# Threshold, Weighted_Caseload, Weighted_Flagged_Rate, Weighted_Precision, Weighted_Recall, Weighted_Coverage_of_Need
TRADEOFF_CANON = {
    "Threshold": ["Threshold", "threshold"],
    "Weighted_Caseload": ["Weighted_Caseload", "WeightedCaseload", "Caseload_Volume", "Caseload"],
    "Weighted_Flagged_Rate": ["Weighted_Flagged_Rate", "Flagged_Rate", "WeightedFlaggedRate"],
    "Weighted_Precision": ["Weighted_Precision", "Precision", "Precision_Pct", "WeightedPrecision"],
    "Weighted_Recall": ["Weighted_Recall", "Recall", "Recall_Pct", "WeightedRecall"],
    "Weighted_Coverage_of_Need": ["Weighted_Coverage_of_Need", "Coverage_of_Need", "Coverage", "WeightedCoverage"],
}

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

thr_col = pick_col(tradeoff, TRADEOFF_CANON["Threshold"])
caseload_col = pick_col(tradeoff, TRADEOFF_CANON["Weighted_Caseload"])
flagged_rate_col = pick_col(tradeoff, TRADEOFF_CANON["Weighted_Flagged_Rate"])
precision_col = pick_col(tradeoff, TRADEOFF_CANON["Weighted_Precision"])
recall_col = pick_col(tradeoff, TRADEOFF_CANON["Weighted_Recall"])
coverage_col = pick_col(tradeoff, TRADEOFF_CANON["Weighted_Coverage_of_Need"])

# Helpers
def safe_weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if value_col not in df.columns or weight_col not in df.columns:
        return float("nan")
    tmp = df[[value_col, weight_col]].dropna()
    if tmp.empty:
        return float("nan")
    wsum = tmp[weight_col].sum()
    if wsum == 0:
        return float("nan")
    return float((tmp[value_col] * tmp[weight_col]).sum() / wsum)

def weighted_flagged_rate(df: pd.DataFrame) -> float:
    # mean of 0/1 with weights
    return safe_weighted_mean(df, "Predicted_Stress", "Household_Weight")

def weighted_actual_rate(df: pd.DataFrame) -> float:
    return safe_weighted_mean(df, "Actual_Stress", "Household_Weight")

def weighted_population(df: pd.DataFrame) -> float:
    if "Household_Weight" not in df.columns:
        return float("nan")
    return float(df["Household_Weight"].dropna().sum())

def flagged_rate_by_group_weighted(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Weighted flagged rate by group: sum(w*y)/sum(w)
    """
    if group_col not in df.columns:
        return pd.DataFrame()
    needed = [group_col, "Predicted_Stress", "Household_Weight"]
    tmp = df[needed].dropna()
    if tmp.empty:
        return pd.DataFrame(columns=[group_col, "Flagged_Rate_Pct", "Weighted_Households"])
    grouped = tmp.groupby(group_col, dropna=False).apply(
        lambda g: pd.Series({
            "Flagged_Rate": (g["Predicted_Stress"] * g["Household_Weight"]).sum() / g["Household_Weight"].sum()
                            if g["Household_Weight"].sum() != 0 else 0.0,
            "Weighted_Households": g["Household_Weight"].sum()
        })
    ).reset_index()
    grouped["Flagged_Rate_Pct"] = grouped["Flagged_Rate"] * 100
    return grouped.sort_values("Flagged_Rate_Pct", ascending=False)

def create_interactive_bar(df_plot: pd.DataFrame, y_col: str, x_col: str, title: str, x_title: str) -> "px.Figure":
    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,
        orientation="h",
        text=x_col,
        color=x_col,
        color_continuous_scale="Blues"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="",
        coloraxis_showscale=False,
        template="plotly_white",
        height=450,
        margin=dict(l=30, r=30, t=55, b=30)
    )
    fig.update_xaxes(range=[0, 100])
    return fig

# Optional label mapping if PHTYPE codes exist (sometimes you have PHTYPE numeric)
PHTYPE_LABELS = {
    1: "One-person Household",
    2: "Couple without Children",
    3: "Couple with Children",
    4: "Lone-parent Family",
    5: "Other Family Household",
    6: "Multiple-family Household",
}

# Sidebar filters
with st.sidebar:
    st.header("Decision Filters")

    provinces = sorted(households["Province_Name"].dropna().unique()) if "Province_Name" in households.columns else []
    tenures = sorted(households["Tenure_Group"].dropna().unique()) if "Tenure_Group" in households.columns else []
    risks = sorted(households["Risk_Band"].dropna().unique()) if "Risk_Band" in households.columns else []
    segments = sorted(households["Risk_Segment"].dropna().unique()) if "Risk_Segment" in households.columns else []
    incomes = sorted(households["Income_Quintile"].dropna().unique()) if "Income_Quintile" in households.columns else []

    province_sel = st.multiselect("Province", options=provinces, default=[])
    tenure_sel = st.multiselect("Tenure Group", options=tenures, default=[])
    risk_sel = st.multiselect("Risk Band", options=risks, default=[])
    segment_sel = st.multiselect("Risk Segment", options=segments, default=[])
    income_sel = st.multiselect("Income Quintile", options=incomes, default=[])

filtered = households.copy()
if province_sel and "Province_Name" in filtered.columns:
    filtered = filtered[filtered["Province_Name"].isin(province_sel)]
if tenure_sel and "Tenure_Group" in filtered.columns:
    filtered = filtered[filtered["Tenure_Group"].isin(tenure_sel)]
if risk_sel and "Risk_Band" in filtered.columns:
    filtered = filtered[filtered["Risk_Band"].isin(risk_sel)]
if segment_sel and "Risk_Segment" in filtered.columns:
    filtered = filtered[filtered["Risk_Segment"].isin(segment_sel)]
if income_sel and "Income_Quintile" in filtered.columns:
    filtered = filtered[filtered["Income_Quintile"].isin(income_sel)]

# SECTION — Executive Snapshot (weighted)
st.subheader("Executive Snapshot (Weighted)")

total_rows = len(filtered)
weighted_pop = weighted_population(filtered)
flag_rate_w = weighted_flagged_rate(filtered)
actual_rate_w = weighted_actual_rate(filtered)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{total_rows:,}")
c2.metric("Estimated households (weighted)", f"{weighted_pop:,.0f}" if pd.notna(weighted_pop) else "—")
c3.metric("Flagged rate (weighted)", f"{flag_rate_w:.2%}" if pd.notna(flag_rate_w) else "—")
c4.metric("Actual stress rate (weighted)", f"{actual_rate_w:.2%}" if pd.notna(actual_rate_w) else "—")

st.caption(
    "All rates are computed using Household_Weight. "
    "This dashboard reads precomputed artifacts only; no training or tuning occurs here."
)

with st.expander("Preview (filtered)", expanded=False):
    st.dataframe(filtered.head(50), width="stretch")

st.divider()

# SECTION — Risk concentration (weighted flagged rate by segment)
st.subheader("Risk Concentration by Segment (Weighted % Flagged)")

segment_options = [c for c in ["Province_Name", "Tenure_Group", "Risk_Band", "Income_Quintile", "Risk_Segment"] if c in filtered.columns]
seg_col = st.selectbox("Choose a segmentation variable", options=segment_options, index=0)

seg_table = flagged_rate_by_group_weighted(filtered, seg_col)

if seg_table.empty:
    st.info("Not enough data to compute segment rates for the selected grouping.")
else:
    # If Risk_Band exists, enforce logical ordering if present
    if seg_col == "Risk_Band":
        order = ["Low", "Medium", "High"]
        if set(order).issubset(set(seg_table["Risk_Band"].astype(str).unique())):
            seg_table["Risk_Band"] = pd.Categorical(seg_table["Risk_Band"], categories=order, ordered=True)
            seg_table = seg_table.sort_values("Risk_Band")

    # If grouping is numeric-coded household type and you have that column, map to labels
    if seg_col == "PHTYPE":
        seg_table["PHTYPE"] = seg_table["PHTYPE"].map(PHTYPE_LABELS).fillna(seg_table["PHTYPE"].astype(str))

    fig_seg = create_interactive_bar(
        seg_table,
        y_col=seg_col,
        x_col="Flagged_Rate_Pct",
        title=f"Weighted % Flagged by {seg_col}",
        x_title="% Flagged (Weighted)"
    )
    st.plotly_chart(fig_seg, width="stretch")

    with st.expander("Segment table (weighted)", expanded=False):
        show_cols = [seg_col, "Flagged_Rate_Pct", "Weighted_Households"]
        st.dataframe(seg_table[show_cols], width="stretch")

st.divider()

# SECTION — Policy Matrix
st.subheader("Policy Matrix (Action Tiers)")

st.dataframe(policy_matrix, width="stretch")

st.caption(
    "This matrix is a precomputed executive artifact mapping tiers to recommended interventions. "
    "It is guidance for decision support and does not imply causality."
)

st.divider()

# SECTION  — Threshold trade-off (UPDATED columns)
st.subheader("Caseload Planning via Threshold Trade-off")

if not thr_col or not caseload_col:
    st.error("Threshold tradeoff file is missing required columns. Expected at least Threshold and Weighted_Caseload.")
else:
    threshold_values = sorted(tradeoff[thr_col].dropna().unique())
    default_thr = 0.30 if 0.30 in threshold_values else threshold_values[len(threshold_values) // 2]

    selected_thr = st.slider(
        "Select decision threshold for caseload planning",
        min_value=float(min(threshold_values)),
        max_value=float(max(threshold_values)),
        value=float(default_thr),
        step=float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.05
    )

    nearest_row = tradeoff.iloc[(tradeoff[thr_col] - selected_thr).abs().argsort()[:1]]

    def get_val(colname: str | None):
        if colname and colname in tradeoff.columns:
            return float(nearest_row[colname].iloc[0])
        return None

    caseload_val = get_val(caseload_col)
    precision_val = get_val(precision_col)
    recall_val = get_val(recall_col)
    coverage_val = get_val(coverage_col)
    flagged_rate_val = get_val(flagged_rate_col)

    m1, m2, m3, m4, m5 = st.columns(5)
    if caseload_val is not None:
        m1.metric("Weighted Caseload", f"{caseload_val:,.0f}")
    if flagged_rate_val is not None:
        m2.metric("Flagged Rate", f"{flagged_rate_val:.2%}")
    if precision_val is not None:
        m3.metric("Precision", f"{precision_val:.2%}")
    if recall_val is not None:
        m4.metric("Recall", f"{recall_val:.2%}")
    if coverage_val is not None:
        m5.metric("Coverage of Need", f"{coverage_val:.2%}")

    # Curves
    fig1 = px.line(tradeoff, x=thr_col, y=caseload_col, markers=True, title="Weighted Caseload by Threshold")
    fig1.update_layout(template="plotly_white", height=320, xaxis_title="Threshold", yaxis_title="Weighted Caseload")
    st.plotly_chart(fig1, width="stretch")

    metric_cols = [c for c in [flagged_rate_col, precision_col, recall_col, coverage_col] if c and c in tradeoff.columns]
    if metric_cols:
        fig2 = px.line(tradeoff, x=thr_col, y=metric_cols, markers=True, title="Quality & Coverage Metrics by Threshold")
        fig2.update_layout(template="plotly_white", height=360, xaxis_title="Threshold", yaxis_title="Rate")
        st.plotly_chart(fig2, width="stretch")

    st.caption(
        "All trade-off values are precomputed (weighted). "
        "Use this section to reason about capacity planning under different thresholds."
    )

st.divider()

# SECTION — Strategy curves (efficiency of targeting)
st.subheader("Strategy Curves (Targeting Efficiency)")

cols = strategy_curves.columns.tolist()
pop_col = "Cum_Pop" if "Cum_Pop" in cols else ("cum_pop" if "cum_pop" in cols else cols[0])
stress_col = "Cum_Stress" if "Cum_Stress" in cols else ("cum_stress" if "cum_stress" in cols else cols[1])
strat_col = "Strategy" if "Strategy" in cols else ("strategy" if "strategy" in cols else (cols[2] if len(cols) > 2 else None))

if strat_col:
    fig_curve = px.line(strategy_curves, x=pop_col, y=stress_col, color=strat_col, title="Cumulative Stress Captured vs Population Targeted")
else:
    fig_curve = px.line(strategy_curves, x=pop_col, y=stress_col, title="Cumulative Stress Captured vs Population Targeted")

fig_curve.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Cumulative Population Targeted",
    yaxis_title="Cumulative Housing Stress Captured"
)
st.plotly_chart(fig_curve, width="stretch")

st.caption("These curves compare targeting efficiency across precomputed strategies (no modeling inside Streamlit).")

# Optional: Robustness display (PSTIR)
if pstir_robust is not None:
    # Expect columns like PSTIR_GR_Clean, flagged_rate, true_stress_rate (based on your earlier artifacts)
    if "PSTIR_GR_Clean" in pstir_robust.columns:
        y_candidates = [c for c in ["flagged_rate", "true_stress_rate", "predicted_risk_mean"] if c in pstir_robust.columns]
        if y_candidates:
            st.divider()
            st.subheader("Robustness Display: PSTIR vs Predicted/Actual Rates")

            fig_pstir = px.line(
                pstir_robust.sort_values("PSTIR_GR_Clean"),
                x="PSTIR_GR_Clean",
                y=y_candidates,
                markers=True
            )
            fig_pstir.update_layout(
                template="plotly_white",
                height=350,
                xaxis_title="Shelter Cost-to-Income Group (PSTIR_GR_Clean)",
                yaxis_title="Rate"
            )
            st.plotly_chart(fig_pstir, width="stretch")
            st.caption("A monotonic gradient supports economic consistency. Displayed values are precomputed artifacts.")

# Export filtered dataset
st.divider()
st.subheader("Export")

st.download_button(
    "Download filtered households (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_dashboard_households.csv",
    mime="text/csv"
)

