import streamlit as st

st.title("Exploratory Data Analysis (EDA) Summary")

st.markdown("""
This page presents key descriptive findings from the 
Canadian Housing Survey (CHS) 2022 dataset.

All statistics and figures were generated during the EDA phase
in `descriptive_stat.ipynb`. No transformations or modeling
are performed on this page.
""")

st.divider()

# 1. Core Housing Need by Income Quintile

st.header("Core Housing Need by Income Quintile")

st.markdown("""
Lower-income households experience significantly higher
rates of core housing need.

This confirms the economic vulnerability gradient.
""")

st.image(
    "core_housing_need_by_province.png",
    caption="Percentage of Households in Core Housing Need by Income Quintile",
    width="stretch"
)

st.divider()

# 2. Shelter Cost-to-Income Ratio by Tenure

st.header("Shelter Cost-to-Income Ratio (STIR) by Tenure")

st.markdown("""
Renters and owners with mortgages experience higher
shelter-cost burdens compared to owners without mortgages.
""")

st.image(
    "stir_distribution_by_tenure_stacked.png",
    caption="Distribution of STIR by Tenure Group",
    width="stretch"
)

st.divider()

# 3. Core Housing Need by Province

st.header("Core Housing Need by Province")

st.markdown("""
There is substantial provincial variation in housing stress,
with British Columbia and Ontario showing higher prevalence.
""")

st.image(
    "core_housing_need_by_province.png",
    caption="Percentage of Households in Core Housing Need by Province",
    width="stretch"
)

st.divider()

st.caption(
    "All figures are derived from CHS 2022 PUMF. "
    "Survey weights applied during descriptive analysis."
)