import pandas as pd

RESERVED_CODES = [9, 96, 99, 99999999]

def apply_reserved_codes_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace Statistics Canada reserved codes with NA.
    """
    return df.replace(RESERVED_CODES, pd.NA)

def require_columns(df, columns):
    """
    Ensure required columns exist in the dataframe.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def weighted_count(df, group_col, weight_col):
    """
    Compute weighted counts for categorical variables.
    """
    return (
        df
        .dropna(subset=[group_col, weight_col])
        .groupby(group_col)[weight_col]
        .sum()
        .reset_index(name="Weighted_Count")
    )

def weighted_proportion(df, group_col, weight_col):
    """
    Compute weighted proportions.
    """
    wc = weighted_count(df, group_col, weight_col)
    total = wc["Weighted_Count"].sum()
    wc["Proportion"] = wc["Weighted_Count"] / total
    return wc

def map_labels(df, column, label_map):
    """
    Map coded values to readable labels (display only).
    """
    df = df.copy()
    df[f"{column}_label"] = df[column].map(label_map)
    return df
