import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Decision Support", layout="wide")
st.title("Decision Support")
st.caption("Policy-oriented targeting and planning using precomputed dashboard artifacts (no training in Streamlit).")

# Paths (prediction/decision pages live under /pages)
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
    return candidates[0]  # default for error messaging

HOUSEHOLDS_PATH = resolve_path("dashboard_households.csv")
POLICY_MATRIX_PATH = resolve_path("dashboard_policy_matrix.csv")
TRADEOFF_PATH = resolve_path("dashboard_threshold_tradeoff.csv")
STRATEGY_CURVES_PATH = resolve_path("dashboard_strategy_curves.csv")
PSTIR_ROBUST_PATH = resolve_path("pstir_robustness.csv") 

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

# robustness file
pstir_robust = None
if PSTIR_ROBUST_PATH.exists():
    pstir_robust = load_csv(PSTIR_ROBUST_PATH)

# Basic checks (fail softly with guidance)
required_household_cols = [
    "Province_Name", "Tenure_Group", "PHTYPE",
    "prob_housing_stress", "Predicted_Housing_Stress", "Risk_Band"
]
missing_cols = [c for c in required_household_cols if c not in households.columns]
if missing_cols:
    st.warning(f"dashboard_households.csv is missing expected columns: {missing_cols}")

# SECTION 0 — Executive snapshot
st.subheader("Executive Snapshot")

total_n = len(households)
flagged_n = int(households["Predicted_Housing_Stress"].sum()) if "Predicted_Housing_Stress" in households.columns else 0
flagged_rate = (flagged_n / total_n) if total_n else 0

col1, col2, col3 = st.columns(3)
col1.metric("Households (rows)", f"{total_n:,}")
col2.metric("Flagged (Predicted_Housing_Stress = 1)", f"{flagged_n:,}")
col3.metric("Flagged Rate", f"{flagged_rate:.2%}")

st.caption("These values come from the scored decision dataset (dashboard_households.csv). No model retraining occurs here.")

st.divider()

# SECTION 1 — Targeting filters (policy segmentation)
st.subheader("Targeting Filters")

with st.sidebar:
    st.header("Decision Filters")

    provinces = sorted(households["Province_Name"].dropna().unique()) if "Province_Name" in households.columns else []
    tenures = sorted(households["Tenure_Group"].dropna().unique()) if "Tenure_Group" in households.columns else []
    risks = sorted(households["Risk_Band"].dropna().unique()) if "Risk_Band" in households.columns else []

    province_sel = st.multiselect("Province", options=provinces, default=[])
    tenure_sel = st.multiselect("Tenure Group", options=tenures, default=[])
    risk_sel = st.multiselect("Risk_Band", options=risks, default=[])

filtered = households.copy()
if province_sel and "Province_Name" in filtered.columns:
    filtered = filtered[filtered["Province_Name"].isin(province_sel)]
if tenure_sel and "Tenure_Group" in filtered.columns:
    filtered = filtered[filtered["Tenure_Group"].isin(tenure_sel)]
if risk_sel and "Risk_Band" in filtered.columns:
    filtered = filtered[filtered["Risk_Band"].isin(risk_sel)]

st.caption(f"Filtered households: {len(filtered):,}")

# SECTION 2 — Where is risk concentrated? (top segments)
st.subheader("Risk Concentration by Segment")

def flagged_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns or "Predicted_Housing_Stress" not in df.columns:
        return pd.DataFrame()
    out = df.groupby(group_col)["Predicted_Housing_Stress"].mean().reset_index()
    out["Flagged_Rate"] = out["Predicted_Housing_Stress"]
    out["Flagged_Rate_Pct"] = out["Flagged_Rate"] * 100
    out = out.sort_values("Flagged_Rate_Pct", ascending=False)
    return out[[group_col, "Flagged_Rate_Pct"]]

seg_col = st.selectbox(
    "Choose segmentation variable",
    options=[c for c in ["Province_Name", "Tenure_Group", "Risk_Band", "PHTYPE"] if c in filtered.columns],
    index=0
)

seg_table = flagged_rate_table(filtered, seg_col)
if seg_table.empty:
    st.info("Selected segmentation not available in the filtered dataset.")
else:
    fig_seg = px.bar(
        seg_table,
        x="Flagged_Rate_Pct",
        y=seg_col,
        orientation="h",
        text="Flagged_Rate_Pct"
    )
    fig_seg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_seg.update_layout(
        template="plotly_white",
        height=450,
        xaxis_title="% Flagged",
        yaxis_title="",
        margin=dict(l=30, r=30, t=40, b=30)
    )
    fig_seg.update_xaxes(range=[0, 100])
    st.plotly_chart(fig_seg, width="stretch")

st.divider()

# SECTION 3 — Policy matrix (what to do)
st.subheader("Policy Matrix (Action Tiers)")

st.dataframe(policy_matrix, width="stretch")

st.caption(
    "This matrix is a precomputed decision artifact mapping risk tiers to recommended interventions. "
    "Use it as guidance; it does not imply causality."
)

st.divider()

# SECTION 4 — Threshold caseload planning
st.subheader("Caseload Planning via Threshold Trade-off")

# Ensure expected cols exist
tradeoff_cols = tradeoff.columns.tolist()

# Try common naming variants safely
thr_col = "Threshold" if "Threshold" in tradeoff_cols else ("threshold" if "threshold" in tradeoff_cols else tradeoff_cols[0])
vol_col = "Caseload_Volume" if "Caseload_Volume" in tradeoff_cols else ("caseload_volume" if "caseload_volume" in tradeoff_cols else tradeoff_cols[1])
prec_col = "Precision_Pct" if "Precision_Pct" in tradeoff_cols else ("Precision_Pct" if "precision_pct" in tradeoff_cols else (tradeoff_cols[2] if len(tradeoff_cols) > 2 else None))

# Slider based on available thresholds
threshold_values = sorted(tradeoff[thr_col].dropna().unique())
default_thr = 0.30 if 0.30 in threshold_values else threshold_values[len(threshold_values)//2]

selected_thr = st.slider(
    "Select decision threshold for caseload planning",
    min_value=float(min(threshold_values)),
    max_value=float(max(threshold_values)),
    value=float(default_thr),
    step=float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.05
)

nearest_row = tradeoff.iloc[(tradeoff[thr_col] - selected_thr).abs().argsort()[:1]]
caseload_val = float(nearest_row[vol_col].iloc[0]) if vol_col in tradeoff.columns else None
precision_val = float(nearest_row[prec_col].iloc[0]) if prec_col and prec_col in tradeoff.columns else None

c1, c2 = st.columns(2)
if caseload_val is not None:
    c1.metric("Estimated Caseload Volume", f"{caseload_val:,.0f}")
if precision_val is not None:
    c2.metric("Precision (at threshold)", f"{precision_val:.1f}%")

fig_trade = px.line(tradeoff, x=thr_col, y=vol_col, markers=True)
fig_trade.update_layout(template="plotly_white", height=350, xaxis_title="Threshold", yaxis_title="Caseload Volume")
st.plotly_chart(fig_trade, width="stretch")

if prec_col and prec_col in tradeoff.columns:
    fig_prec = px.line(tradeoff, x=thr_col, y=prec_col, markers=True)
    fig_prec.update_layout(template="plotly_white", height=350, xaxis_title="Threshold", yaxis_title="Precision (%)")
    st.plotly_chart(fig_prec, width="stretch")

st.caption("Use this to plan program capacity. Lower thresholds increase inclusiveness but raise caseload.")

st.divider()

# SECTION 5 — Strategy curves (efficiency of targeting)
st.subheader("Strategy Curves (Targeting Efficiency)")

# Expect columns like Cum_Pop, Cum_Stress, Strategy
cols = strategy_curves.columns.tolist()
pop_col = "Cum_Pop" if "Cum_Pop" in cols else ("cum_pop" if "cum_pop" in cols else cols[0])
stress_col = "Cum_Stress" if "Cum_Stress" in cols else ("cum_stress" if "cum_stress" in cols else cols[1])
strat_col = "Strategy" if "Strategy" in cols else ("strategy" if "strategy" in cols else (cols[2] if len(cols) > 2 else None))

if strat_col:
    fig_curve = px.line(strategy_curves, x=pop_col, y=stress_col, color=strat_col)
else:
    fig_curve = px.line(strategy_curves, x=pop_col, y=stress_col)

fig_curve.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Cumulative Population Targeted",
    yaxis_title="Cumulative Housing Stress Captured"
)
st.plotly_chart(fig_curve, width="stretch")

st.caption("These curves illustrate how efficiently different targeting strategies capture housing stress under resource constraints.")

# Optional: Robustness display (PSTIR)
if pstir_robust is not None and "PSTIR_GR_Clean" in pstir_robust.columns and "flagged_rate" in pstir_robust.columns:
    st.divider()
    st.subheader("Robustness Display: Flagged Rate by Shelter Cost Burden (PSTIR_GR_Clean)")

    fig_pstir = px.line(pstir_robust, x="PSTIR_GR_Clean", y=["flagged_rate","true_stress_rate"], markers=True)
    fig_pstir.update_layout(template="plotly_white", height=350, xaxis_title="PSTIR_GR_Clean", yaxis_title="Flagged Rate")
    st.plotly_chart(fig_pstir, width="stretch")

# Export filtered dataset (helpful for policy review)
st.divider()
st.subheader("Export")

st.download_button(
    "Download filtered households (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_dashboard_households.csv",
    mime="text/csv"
)