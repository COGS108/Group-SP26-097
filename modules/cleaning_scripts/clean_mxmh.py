import pandas as pd
from pathlib import Path

def clean_mxmh(data_path = "data/00-raw/MxMH_Survey/mxmh_survey_results.csv", output_path = "data/01-interim/MxMH_Survey_cleaned.csv"):

    # Tidy Dataset
    df = pd.read_csv(data_path)

    # Clean Data
    # Age has one missing value, so we drop that row instead of imputing a potentially incorrect age.
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    rows_before_drop = len(df)
    df = df.dropna(subset=["Age"]).copy()
    print("Dropped", rows_before_drop - len(df), "rows with missing Age values.")

    # Fill categorical missing values with the mode when there is a clear most common response.
    fill_values = {
        "Primary streaming service": df["Primary streaming service"].mode(dropna=True).iloc[0],
        "While working": df["While working"].mode(dropna=True).iloc[0],
        "Instrumentalist": df["Instrumentalist"].mode(dropna=True).iloc[0],
        "Composer": df["Composer"].mode(dropna=True).iloc[0],
        "Foreign languages": "Unknown",
    }

    for column, fill_value in fill_values.items():
        missing_before = df[column].isna().sum()
        df[column] = df[column].fillna(fill_value)
        print(f"Filled {missing_before} missing values in {column} with {fill_value!r}.")

    # Save cleaned data for later use.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    clean_mxmh("../data/00-raw/MxMH_Survey/mxmh_survey_results.csv", "../data/01-interim/MxMH_Survey_cleaned.csv")