"""
data_preprocessing.py

Loads the raw IMDb dataset, performs cleaning and feature engineering,
then saves a cleaned CSV for model training.
"""

import pandas as pd
import numpy as np
import os

def load_and_clean_data(input_path: str) -> pd.DataFrame:
    """
    Loads and cleans the raw CSV file:
      1. Fixes Year, Duration, Votes columns.
      2. Drops rows without a target (Rating).
      3. Imputes missing numeric/categorical values.
    Returns a cleaned DataFrame.
    """
    # Read CSV (adjust encoding or path if needed)
    df = pd.read_csv(input_path, encoding='cp1252')

    # 1. Clean Year (remove parentheses)
    df['Year'] = df['Year'].astype(str).str.replace(r'[()]', '', regex=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # 2. Clean Duration (remove 'min', convert to numeric)
    df['Duration'] = (
        df['Duration'].astype(str)
        .str.replace('min', '', regex=False)
        .str.strip()
    )
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

    # 3. Clean Votes (remove commas, convert to numeric)
    df['Votes'] = (
        df['Votes'].astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

    # 4. Drop rows with missing target
    df.dropna(subset=['Rating'], inplace=True)

    # 5. Impute numeric columns (Duration) with median
    df['Duration'].fillna(df['Duration'].median(), inplace=True)

    # 6. Impute categorical columns with "Unknown"
    for cat_col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        if cat_col in df.columns:
            df[cat_col].fillna('Unknown', inplace=True)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds or transforms features, such as:
      - Log_Votes for skew
      - Decade from Year
    Returns the transformed DataFrame.
    """
    # 1. Log-transform Votes
    df['Log_Votes'] = np.log1p(df['Votes'].fillna(0))

    # 2. Create Decade
    df['Decade'] = (df['Year'] // 10) * 10

    # 3. (Optional) Additional transformations here

    return df

if __name__ == "__main__":
    # Define paths
    input_csv = "data/IMDb Movies India.csv"    # Raw data
    output_csv = "data/IMDb_Movies_Cleaned.csv" # Cleaned output

    # Load & clean
    df_clean = load_and_clean_data(input_csv)

    # Feature engineering
    df_final = engineer_features(df_clean)

    # Save final CSV
    os.makedirs("data", exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    print(f"[INFO] Saved cleaned data to {output_csv} with shape {df_final.shape}.")
