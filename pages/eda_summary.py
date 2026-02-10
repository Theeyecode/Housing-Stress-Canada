import streamlit as st
import pandas as pd

st.title("📊 Exploratory Data Analysis (EDA)")

st.write("""
This section presents descriptive summaries of housing
affordability indicators.
""")

st.subheader("Placeholder Chart")
st.bar_chart(pd.DataFrame({"Example": [1, 2, 3]}))

st.subheader("Placeholder Table")
st.dataframe(pd.DataFrame({"Coming Soon": ["PCHN", "PSTIR_GR"]}))
