import pandas as pd
import streamlit as st

@st.cache_data
def load_chs_data():
    df = pd.read_csv("data/Chs2022ecl_pumf.csv")

    # Reserved codes → NA (EDA phase only)
    reserved_codes = [9, 99, 96, 99999999]

    df = df.replace(reserved_codes, pd.NA)

    return df
