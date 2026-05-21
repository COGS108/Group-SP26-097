import pandas as pd
from pathlib import Path

# Tidy Dataset
df = pd.read_csv("../data/00-raw/MxMH_Survey/mxmh_survey_results.csv")
print(df.head())
print("!! This dataset is tidy as each case study is in its own separate row, and each variable is in its own column.")

# Dataset Shape
print("!! The dataset has " + str(df.shape[0]) + " rows, and " + str(len(df.columns)) + " columns.")

# NaNs in Dataset
print("NaNs per colum:\n", df.isnull().sum())
print("\nTotal NaNs in dataset:", df.isnull().sum().sum())
print("!! We can see that there is a lot of null values in the BPM variable. This is especially important as BPM might be one of our explored variables relating to the mental health of a subject.")

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

print("\nRemaining NaNs after cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Save cleaned data for later use.
output_path = Path("../data/01-interim/MxMH_Survey_cleaned.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")