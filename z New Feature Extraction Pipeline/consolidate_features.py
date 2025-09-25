import os
import glob
import pandas as pd
import pickle
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_DIR = "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/processedbatch"  # UPDATE THIS!
CONSOLIDATED_OUTPUT_PATH = "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/bexvar_features/features.pkl"  # UPDATE THIS!

def consolidate_pickles(input_dir, output_path):
    """
    Load all .pkl files from input_dir, concatenate their DataFrames, and save to output_path
    with manual print progress and sample preview.
    """
    pattern = os.path.join(input_dir, "features_*.pkl")
    files = sorted(glob.glob(pattern))
    total_files = len(files)

    print(f"\n🔍 Found {total_files} pickle files in {input_dir}")

    if total_files == 0:
        print("🚫 No .pkl files found. Exiting.")
        return

    all_dfs = []

    # Preview first file and its features
    try:
        first_sample = pd.read_pickle(files[0])
        print(f"\n📌 Sample from first light curve: {files[0]}")
        sample = first_sample.iloc[0]
        print(f"→ File path: {sample['file_path']}")
        print("→ Features:")
        for name, value in zip(sample["feature_names"], sample["feature_values"]):
            print(f"   - {name}: {value:.6f}")
    except Exception as e:
        print(f"⚠️ Could not preview sample from first file: {e}")

    # Load and track progress manually
    print("\n📥 Starting full consolidation...")
    checkpoints = {int(total_files * frac): f"{int(frac*100)}%" for frac in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]}
    for i, file in enumerate(files):
        try:
            df = pd.read_pickle(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

        if i in checkpoints:
            print(f"  → Loaded {checkpoints[i]} of files ({i}/{total_files})")

    if not all_dfs:
        print("🚫 No valid DataFrames were loaded. Exiting.")
        return

    print("\n🧩 Concatenating DataFrames...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_pickle(output_path)

    print(f"\n✅ Consolidated DataFrame saved to: {output_path}")
    print(f"📊 Total records: {len(combined_df)}")

if __name__ == "__main__":
    consolidate_pickles(PROCESSED_DATA_DIR, CONSOLIDATED_OUTPUT_PATH)
