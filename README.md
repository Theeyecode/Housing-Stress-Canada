# Housing Affordability Stress in Canada: A Household-Level, Data-Driven Decision Support

## Project Overview
This capstone project examines housing affordability stress among Canadian households
using household-level microdata from Statistics Canada’s **Canadian Housing Survey (CHS) 2022 Public Use Microdata File (PUMF).

Housing affordability challenges in Canada are often discussed using aggregate indicators such as average housing prices or provincial shelter cost burdens. While useful, these measures can obscure important differences across household types, housing tenure, income levels, and regions. This project addresses that gap by applying a **micro-level, household-based analytical approach**.

The project focuses on two key indicators of housing affordability stress:
**Core Housing Need (PCHN)** and **Shelter Cost-to-Income Ratio Groups (PSTIR_GR)**.
By adopting a **decision-support orientation**, the analysis aims to help policymakers and housing stakeholders better identify households most at risk and design more targeted interventions under limited resource conditions.

## Project Objectives
- Analyze housing affordability stress among Canadian households
- Identify household, economic, and housing characteristics associated with housing stress
- Compare affordability outcomes between renters and homeowners
- Examine variations in housing stress across provinces and regions
- Develop an interactive dashboard to support data-driven housing policy decisions

---

## Data Source
- **Dataset:** Canadian Housing Survey (CHS) 2022 – Public Use Microdata File  
- **Provider:** Statistics Canada  
- **Unit of Analysis:** Household  
- **Catalogue:** 46-25-0001  

Public documentation and access information are available at:  
https://www150.statcan.gc.ca/n1/en/catalogue/46250001

## Key Variables

### Outcome Variables
- `PCHN` – Primary Household in Core Housing Need  
- `PSTIR_GR` – Shelter Cost-to-Income Ratio Group  

### Selected Explanatory Variables
- Housing tenure
- Household income
- Province and region
- Household and dwelling characteristics

Variable definitions, universes, and reserved codes are interpreted strictly using the Statistics Canada CHS 2022 data dictionary.

---

## Methods & Tools
- **Python** for data analysis and processing  
- **Pandas & NumPy** for data manipulation  
- **Matplotlib, Seaborn, and Plotly** for visualization  
- **Streamlit** for interactive dashboard development  

During exploratory data analysis (EDA), reserved or invalid survey codes are converted to missing values according to the official codebook. No recoding, collapsing, or modeling is performed at this stage.

---

## Repository Structure

HOUSING-STRESS-CANADA/
│── app.py                  # Streamlit dashboard application
│── data/
│   └── Chs2022ecl_pumf.csv  # CHS 2022 PUMF dataset
│── notebooks/
│   └── descriptive_stat.ipynb
│── src/
│   ├── data_loader.py
│   └── utils.py
│── requirements.txt
│── README.md


# How to Run the Project Locally

1. # Clone the repository
git clone <your-github-repository-url>
cd HOUSING-STRESS-CANADA

2. # Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

3. # Install project dependencies
pip install -r requirements.txt

4. # Run the Streamlit dashboard
streamlit run app.py



# Ethical & Privacy Considerations

- The analysis uses anonymized, publicly available microdata from Statistics Canada

- No personally identifiable information is included in the dataset

- All analysis complies with Statistics Canada PUMF usage and disclosure guidelines

- Results are reported only at aggregate levels

- No attempt is made to identify individuals, households, or specific dwellings

- This project follows the ethical principles outlined in the Tri-Council Policy Statement
- (TCPS 2) for research involving human data, as applicable to public-use secondary data.