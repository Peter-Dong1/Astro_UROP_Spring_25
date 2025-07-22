import os
import pickle
from pathlib import Path

# Update these imports to your actual environment
from helper import load_all_fits_files, load_n_light_curves
from config import DEFAULT_BAND, BASE_DIR

def split_light_curves_into_partitions(num_partitions=28, output_dir=os.path.join(BASE_DIR, "data/split_light_curves")):
    # Step 1: Load all available FITS files
    print("Loading FITS files...")
    fits_files = load_all_fits_files()
    if not fits_files:
        raise ValueError("No FITS files found.")

    print(f"Found {len(fits_files)} FITS files. Loading light curves...")
    light_curves = load_n_light_curves("all", fits_files, band=DEFAULT_BAND)
    total_curves = len(light_curves)
    print(f"Loaded {total_curves} light curves.")

    # Step 2: Calculate partition sizes
    partition_size = total_curves // num_partitions
    remainder = total_curves % num_partitions

    # Step 3: Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Split and save each partition
    start_idx = 0
    for i in range(num_partitions):
        end_idx = start_idx + partition_size + (1 if i < remainder else 0)
        partition = light_curves[start_idx:end_idx]
        partition_file = Path(output_dir) / f"light_curves_partition_{i:02d}.pkl"
        with open(partition_file, "wb") as f:
            pickle.dump(partition, f)
        print(f"Saved partition {i+1}/{num_partitions} with {len(partition)} light curves to {partition_file}")
        start_idx = end_idx

    print("All partitions saved successfully.")

if __name__ == "__main__":
    split_light_curves_into_partitions()
