import pandas as pd
import streamlit as st
from src.utils import require_columns


# RAW CHS DATA LOADER (EDA phase)

@st.cache_data
def load_chs_data():
    df = pd.read_csv("data/Chs2022ecl_pumf.csv")
    reserved_codes = [9, 99, 96, 99999999]
    df = df.replace(reserved_codes, pd.NA)
    return df


# SCORED DATA LOADER (MODEL INTEGRATION)

@st.cache_data
def load_scored_data():
    """
    Load finalized model output dataset.
    No training or thresholding happens here.
    """

    df = pd.read_csv("artifacts/df_scored.csv")

    required_columns = [
        "PCHN_Clean",   # actual outcome
        "Detailed_Tenure", # tenure type (e.g. owned with mortgage)
        "Tenure_Group",   # tenure group (e.g. owned vs rented)
        "prob_housing_stress",  # predicted probability of housing stress
        "Predicted_Housing_Stress",  # binary prediction of housing stress (threshold = 0.30)
        "Risk_Band",    # risk band (Low, Medium, High)
        "Province_Name"            # province
    ]

    require_columns(df, required_columns)

    return df

# Data Loader for Decision Support Page
@st.cache_data
def load_decision_support_data():
    df = pd.read_csv("artifacts/dashboard_households.csv")
    return df
