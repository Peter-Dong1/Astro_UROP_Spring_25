import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import FEATURES_FILE

# --- Configuration ---

CHECK_COLUMNS = None  # Optionally provide a list of columns to check; or None for all

def check_for_nans(df, columns=None):
    """
    Print which features contain NaNs and how many.
    """
    if columns is None:
        columns = df.columns

    print(f"🔍 Checking {len(columns)} columns for NaNs...\n")

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column {col} not in DataFrame.")
            continue

        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"❌ Column '{col}' has {nan_count} NaN values.")
        else:
            print(f"✅ Column '{col}' has no NaNs.")

    print("\n📌 Total rows with any NaNs across checked columns:", df[columns].isna().any(axis=1).sum())

if __name__ == "__main__":
    print(f"📂 Loading DataFrame from: {FEATURES_FILE}")
    df = pd.read_pickle(FEATURES_FILE)

    # If you want to only check numeric features for modeling:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    check_for_nans(df, columns=numeric_cols)
