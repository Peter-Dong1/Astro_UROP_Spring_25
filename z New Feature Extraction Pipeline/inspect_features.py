import pandas as pd
import numpy as np
import os
from config import FEATURES_FILE

def inspect_columns(file_path, preview_rows=3):
    _, ext = os.path.splitext(file_path)

    if ext == '.pkl':
        df = pd.read_pickle(file_path)
    elif ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print("\n📋 DataFrame Column Tree:")
    print("=" * 40)
    for i, col in enumerate(df.columns):
        print(f"\n{i+1:2d}. {col}")
        col_data = df[col].dropna().head(preview_rows)

        for j, val in enumerate(col_data):
            prefix = f"   ├─ Row {j+1}: "
            if isinstance(val, dict):
                keys = list(val.keys())
                print(prefix + f"dict with keys: {keys}")
            elif isinstance(val, (list, tuple)):
                preview = val[:3] if len(val) > 3 else val
                print(prefix + f"{type(val).__name__} of length {len(val)}: {preview}")
            elif isinstance(val, np.ndarray):
                print(prefix + f"np.ndarray with shape {val.shape}, dtype={val.dtype}")
            elif isinstance(val, pd.DataFrame):
                print(prefix + f"DataFrame with shape {val.shape} and columns: {list(val.columns)}")
            elif isinstance(val, pd.Series):
                print(prefix + f"Series with shape {val.shape}")
            else:
                print(prefix + f"{type(val).__name__}: {str(val)[:80]}")

    print(f"\n🔢 Total columns: {len(df.columns)}")

if __name__ == "__main__":
    print("Inspecting columns in:", FEATURES_FILE)
    inspect_columns(FEATURES_FILE)
