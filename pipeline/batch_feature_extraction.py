"""
batch_feature_extraction.py — PATH B step 2: per-partition feature extraction

Receives a pre-split partition file (a pickled list of FITS file paths produced
by split_curves.py), loads the light curves, extracts 11 statistical features
per curve using ProcessPoolExecutor, and saves one pickle per curve.

Shared feature logic lives in lib/feature_functions.py.

Usage (typically via SLURM array job):
    python batch_feature_extraction.py \
        --chunk-file data/split_light_curves/light_curves_partition_07.pkl \
        --chunk-id 7 \
        --output-dir extracted_features/

    sbatch slurm/path_b_2_extract.slurm  (submits array 0-28)

Output:
    <output-dir>/features_<fits_basename>.pkl  (one file per light curve)
"""

import os
import pickle
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from lib.feature_functions import df_extract_statistical_features_error, chunked
from config import EXTRACTED_FEATURES_DIR, DEFAULT_BAND
from helper import load_light_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--chunk-file', type=str, required=True,
                   help="Path to pickled partition file (list of FITS file path strings)")
    p.add_argument('--chunk-id',   type=int, default=0,
                   help="Array task index (used for logging)")
    p.add_argument('--output-dir', type=str, default=EXTRACTED_FEATURES_DIR,
                   help="Directory to write per-curve feature pickles")
    return p.parse_args()


def process_chunk(chunk_light_curves, band, output_dir):
    """
    PATH B chunk worker: extract features from a list of light curve DataFrames
    and save one pickle per curve named features_<fits_basename>.pkl.
    """
    pid = os.getpid()
    print(f"[PID {pid}] Starting chunk with {len(chunk_light_curves)} light curves...")

    for i, lc in enumerate(chunk_light_curves):
        try:
            file_name = lc.attrs.get("FILE_NAME", f"unknown_{i}")
            print(f"[{i}/{len(chunk_light_curves)}] Processing {file_name}...")

            result, _sig_nev = df_extract_statistical_features_error(lc)
            if result is None:
                print(f"[Warning] Skipping {file_name} due to processing error.")
                continue

            base_name = Path(file_name).name.replace(".fits", "").replace("/", "_")
            out_path = os.path.join(output_dir, f"features_{base_name}.pkl")
            result.to_pickle(out_path)
            print(f"  → Saved features to {out_path}")

        except Exception as e:
            print(f"[Error] Failed to process curve {i}: {e}")

    print(f"[PID {pid}] Finished chunk")


def main():
    """Load a partition of FITS paths, extract features in parallel, save per-curve pickles."""
    args = parse_args()
    chunk_file = args.chunk_file
    chunk_idx  = args.chunk_id
    output_dir = args.output_dir

    print(f"Starting PATH B feature extraction (chunk {chunk_idx})...")
    print(f"Chunk file: {chunk_file}")

    # Load the partition — a list of FITS file path strings (produced by split_curves.py)
    with open(chunk_file, "rb") as f:
        file_paths = pickle.load(f)
    print(f"Partition contains {len(file_paths)} file paths.")

    # Load light curve DataFrames from the FITS files
    light_curves = []
    for p in file_paths:
        lc = load_light_curve(p, band=DEFAULT_BAND)
        if lc is not None:
            light_curves.append(lc)
    print(f"Successfully loaded {len(light_curves)} light curves from partition.")

    os.makedirs(output_dir, exist_ok=True)

    # Submit each curve as a chunk of size 1 to the process pool
    with ProcessPoolExecutor(max_workers=85) as exe:
        futures = {}
        for chunk, chunk_pos in chunked(light_curves, size=1):
            fut = exe.submit(process_chunk, chunk, DEFAULT_BAND, output_dir)
            futures[fut] = chunk_pos

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"✗ Curve batch at position {idx} failed: {e}")

    print(f"Chunk {chunk_idx} complete.")


if __name__ == "__main__":
    main()
