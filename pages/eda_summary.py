import streamlit as st

st.title("Hypothesis Testing Findings")

st.markdown("""
This page presents the results of the hypothesis testing conducted during
the Exploratory Data Analysis phase using the CHS 2022 dataset.
""")

st.divider()


# H1
st.header("H1: Renters Experience Higher Housing Stress")

st.image(
    "housing_need_by_tenure.png",
    caption="Percentage of Households in Core Housing Need by Tenure",
    use_container_width=True
)

st.image(
    "stir_distribution_by_tenure_stacked.png",
    caption="Distribution of Shelter Cost-to-Income Ratio by Tenure",
    use_container_width=True
)

st.divider()

# H2
st.header("H2: Lower Income Households Experience Higher Housing Stress")

st.image(
    "housing_need_by_income_quintile.png",
    caption="Percentage of Households in Core Housing Need by Income Quintile",
    use_container_width=True
)

st.divider()


# H3
st.header("H3: Housing Stress Varies Across Provinces")

st.image(
    "core_housing_need_by_province.png",
    caption="Percentage of Households in Core Housing Need by Province",
    use_container_width=True
)

st.divider()

st.caption(
    "All figures are derived from CHS 2022 PUMF. "
    "Survey weights applied during descriptive analysis."
)