"""
Encode categorical variables in the cleaned MxMH Survey dataset.

This script performs one-hot encoding on nominal categorical variables 
and ordinal encoding on frequency-based variables.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pathlib import Path
import json


def ordinal_encode_frequencies(df):
    """
    Ordinal encode frequency columns: Never, Rarely, Sometimes, Very frequently.
    Maps to 0, 1, 2, 3 respectively.
    """
    frequency_map = {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Very frequently": 3,
    }
    
    frequency_cols = [col for col in df.columns if col.startswith("Frequency [")]
    
    for col in frequency_cols:
        df[col] = df[col].map(frequency_map)
    
    print(f"Ordinal encoded {len(frequency_cols)} frequency columns.")
    return df, frequency_map


def one_hot_encode_categorical(df, categorical_cols, drop_first=True):
    """
    One-hot encode nominal categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe.
    categorical_cols : list
        List of column names to one-hot encode.
    drop_first : bool, default=True
        Whether to drop the first category to avoid multicollinearity.
    
    Returns:
    --------
    df_encoded : pd.DataFrame
        DataFrame with one-hot encoded columns.
    encoder : OneHotEncoder
        The fitted encoder for future use.
    """
    encoder = OneHotEncoder(sparse_output=False, drop="first" if drop_first else None)
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )
    
    df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    print(f"One-hot encoded {len(categorical_cols)} categorical columns.")
    return df_encoded, encoder


def label_encode_binary(df, binary_cols):
    """
    Label encode binary categorical variables (Yes/No).
    Maps to 0 (No), 1 (Yes).
    """
    binary_map = {"No": 0, "Yes": 1}
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    
    print(f"Label encoded {len([c for c in binary_cols if c in df.columns])} binary columns.")
    return df, binary_map


def encode_mxmh_survey(input_path, output_path, output_mapping_path=None):
    """
    Load cleaned survey data and apply categorical encoding.
    
    Parameters:
    -----------
    input_path : str
        Path to cleaned CSV file.
    output_path : str
        Path to save encoded CSV file.
    output_mapping_path : str, optional
        Path to save encoding mappings as JSON.
    
    Returns:
    --------
    df_encoded : pd.DataFrame
        The encoded dataframe.
    mappings : dict
        Dictionary containing all encoding mappings.
    """
    df = pd.read_csv(input_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Binary columns (Yes/No)
    binary_cols = ["While working", "Instrumentalist", "Composer", "Exploratory"]
    df, binary_map = label_encode_binary(df, binary_cols)
    
    # Ordinal frequency columns
    df, frequency_map = ordinal_encode_frequencies(df)
    
    # Nominal categorical columns for one-hot encoding
    # Note: Fav genre, Primary streaming service, Music effects
    nominal_cols = ["Primary streaming service", "Fav genre", "Music effects"]
    nominal_cols = [col for col in nominal_cols if col in df.columns]
    
    df_encoded, encoder = one_hot_encode_categorical(df, nominal_cols, drop_first=True)
    
    # Drop unnecessary columns
    cols_to_drop = ["Timestamp", "Permissions", "Foreign languages"]
    cols_to_drop = [col for col in cols_to_drop if col in df_encoded.columns]
    df_encoded = df_encoded.drop(columns=cols_to_drop)
    
    print(f"Final encoded data: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
    
    # Save encoded data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(output_path, index=False)
    print(f"Saved encoded data to {output_path}")
    
    # Save mappings for reference
    mappings = {
        "binary_encoding": binary_map,
        "ordinal_frequency_encoding": frequency_map,
        "one_hot_encoded_columns": encoder.get_feature_names_out(nominal_cols).tolist(),
    }
    
    if output_mapping_path:
        output_mapping_path = Path(output_mapping_path)
        output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_mapping_path, "w") as f:
            json.dump(mappings, f, indent=2)
        print(f"Saved encoding mappings to {output_mapping_path}")
    
    return df_encoded, mappings


if __name__ == "__main__":
    input_file = "data/01-interim/MxMH_Survey_cleaned.csv"
    output_file = "data/02-processed/MxMH_Survey_encoded.csv"
    mapping_file = "data/02-processed/encoding_mappings.json"
    
    df_encoded, mappings = encode_mxmh_survey(input_file, output_file, mapping_file)
    print("\nEncoding complete!")
