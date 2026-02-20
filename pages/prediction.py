import streamlit as st
from src.data_loader import load_scored_data

st.title("Prediction Results")

df = load_scored_data()

st.success("Scored dataset loaded successfully!")

st.write("Preview:")
st.dataframe(df.head())