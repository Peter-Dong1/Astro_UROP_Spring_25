"""
split_curves.py — PATH B step 1: partition FITS file paths for distributed extraction

Splits the full list of accessible FITS file paths into N equal partitions and
saves each as a small pickle file (list of strings). This avoids loading any
light curve data into memory — the old approach required 350 GB RAM; this
requires ~4 GB.

Each partition pickle is later consumed by one SLURM array task running
batch_feature_extraction.py.

Usage:
    python split_curves.py
    sbatch slurm/path_b_1_split.slurm

Output:
    data/split_light_curves/light_curves_partition_00.pkl
    data/split_light_curves/light_curves_partition_01.pkl
    ...  (28 files total by default)
"""

import os
import pickle
from pathlib import Path

from helper import load_all_fits_files, read_inaccessible_lightcurves
from config import BASE_DIR


def split_file_paths_into_partitions(
        num_partitions=28,
        output_dir=os.path.join(BASE_DIR, "data", "split_light_curves")):
    """
    Load all accessible FITS file paths and split them into num_partitions
    pickle files, each containing a list of path strings.

    Parameters:
        num_partitions (int): Number of partitions to create (default 28,
                              matching the run_chunks.slurm array size 0-28).
        output_dir (str): Directory to write partition pickles into.
    """
    print("Loading FITS file paths...")
    fits_files = load_all_fits_files()
    if not fits_files:
        raise ValueError("No FITS files found.")

    # Filter out known inaccessible files
    inaccessible = set(read_inaccessible_lightcurves())
    fits_files = [f for f in fits_files if f not in inaccessible]
    total = len(fits_files)
    print(f"Found {total} accessible FITS files (after filtering inaccessible).")

    partition_size = total // num_partitions
    remainder = total % num_partitions

    os.makedirs(output_dir, exist_ok=True)

    start_idx = 0
    for i in range(num_partitions):
        end_idx = start_idx + partition_size + (1 if i < remainder else 0)
        partition = fits_files[start_idx:end_idx]  # list of str (file paths)
        out_file = Path(output_dir) / f"light_curves_partition_{i:02d}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(partition, f)
        print(f"  Partition {i+1:02d}/{num_partitions}: {len(partition)} paths → {out_file}")
        start_idx = end_idx

    print(f"\nAll {num_partitions} partitions saved to {output_dir}")


if __name__ == "__main__":
    split_file_paths_into_partitions()
