"""
consolidate_features.py — PATH B step 3: merge per-curve feature pickles

Scans the extracted_features/ directory for all features_*.pkl files produced
by batch_feature_extraction.py, concatenates them into a single DataFrame,
and saves the result as the final FEATURES_FILE.

Usage:
    python consolidate_features.py
    sbatch slurm/path_b_3_consolidate.slurm

Input:
    extracted_features/features_*.pkl  (one file per light curve from PATH B step 2)

Output:
    data/all/amp_max_features/features.pkl  (= FEATURES_FILE in config.py)
"""

import os
import glob
import pandas as pd

from config import EXTRACTED_FEATURES_DIR, FEATURES_FILE


def consolidate_pickles(input_dir, output_path):
    """
    Load all features_*.pkl files from input_dir, concatenate, and save.

    Parameters:
        input_dir (str): Directory containing per-curve feature pickles.
        output_path (str): Path to write the merged DataFrame.
    """
    pattern = os.path.join(input_dir, "features_*.pkl")
    files = sorted(glob.glob(pattern))
    total_files = len(files)

    print(f"Found {total_files} pickle files in {input_dir}")
    if total_files == 0:
        print("No .pkl files found. Exiting.")
        return

    # Preview the first file
    try:
        first_sample = pd.read_pickle(files[0]).iloc[0]
        print(f"\nSample from: {files[0]}")
        print(f"  file_path: {first_sample['file_path']}")
        for name, value in zip(first_sample["feature_names"], first_sample["feature_values"]):
            print(f"  {name}: {value:.6f}")
        print()
    except Exception as e:
        print(f"Could not preview first file: {e}")

    # Load all pickles with progress reporting
    all_dfs = []
    checkpoints = {int(total_files * frac): f"{int(frac*100)}%"
                   for frac in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]}
    for i, file in enumerate(files):
        try:
            all_dfs.append(pd.read_pickle(file))
        except Exception as e:
            print(f"Failed to load {file}: {e}")
        if i in checkpoints:
            print(f"  Loaded {checkpoints[i]} ({i}/{total_files} files)")

    if not all_dfs:
        print("No valid DataFrames were loaded. Exiting.")
        return

    print("Concatenating DataFrames...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_pickle(output_path)

    print(f"Consolidated {len(combined_df)} records → {output_path}")


if __name__ == "__main__":
    consolidate_pickles(EXTRACTED_FEATURES_DIR, FEATURES_FILE)
