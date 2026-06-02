"""
run_feature_extraction.py — PATH A: single-node feature extraction

Loads all eROSITA light curves, extracts 11 statistical features per curve
using multi-process parallelism (ProcessPoolExecutor), and saves the result.

Shared feature logic lives in lib/feature_functions.py.

Usage:
    python run_feature_extraction.py
    sbatch slurm/path_a.slurm

Output:
    data/all/processed/features_chunk_*.pkl  (intermediate)
    data/all/amp_max_features/features.pkl   (final, = FEATURES_FILE)
    data/all/amp_max_features/SIG_NEV_mappings.pkl
"""

import os
import glob
import pickle
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from lib.feature_functions import df_extract_statistical_features_error, chunked
from config import FEATURES_FILE, DEFAULT_BAND, LOAD_SIZE, PROCESSED_DATA_DIR
from helper import load_all_fits_files, load_n_light_curves, DEFAULT_DATA_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--job-id',   type=int, default=0,
                   help="Which job this is (0-based; reserved for future array splitting)")
    p.add_argument('--num-jobs', type=int, default=1,
                   help="Total number of parallel jobs (reserved for future array splitting)")
    return p.parse_args()


def process_chunk(chunk_light_curves, chunk_index, band, output_dir):
    """
    PATH A chunk worker: extract features from a small list of curves and
    save the combined result as a single pickle (features_chunk_<N>.pkl).

    One pickle per chunk of 5 curves — these are later concatenated in main().
    """
    pid = os.getpid()
    print(f"[PID {pid}] Starting chunk {chunk_index} (n={len(chunk_light_curves)} curves)...")
    features_list = []
    for i, lc in enumerate(chunk_light_curves):
        print(f"[Chunk {chunk_index}] Processing {i}/{len(chunk_light_curves)}:")
        result, _sig_nev = df_extract_statistical_features_error(lc)
        if result is not None:
            features_list.append(result)

    if not features_list:
        print(f"[Warning] Chunk {chunk_index} produced no results.")
        return None

    df_chunk = pd.concat(features_list, ignore_index=True)
    chunk_file = os.path.join(output_dir, f"features_chunk_{chunk_index}.pkl")
    df_chunk.to_pickle(chunk_file)
    print(f"  → Saved chunk {chunk_index} ({len(df_chunk)} curves) to {chunk_file}")
    print(f"[PID {pid}] Finished chunk {chunk_index}")
    return chunk_file


def main():
    """Load all light curves, extract features in parallel, save to FEATURES_FILE."""
    print("Starting PATH A feature extraction pipeline...")

    # Load FITS file list
    try:
        fits_files = load_all_fits_files()
        if not fits_files:
            raise ValueError("No FITS files found.")
    except Exception as e:
        print(f"Error loading FITS files: {e}")
        return

    light_curves = load_n_light_curves(LOAD_SIZE, fits_files, band=DEFAULT_BAND)
    print(f"Loaded {len(light_curves)} light curves")

    chunk_size = 5
    out_dir = PROCESSED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    chunk_files = []
    with ProcessPoolExecutor(max_workers=85) as exe:
        futures = {}
        for chunk, chunk_idx in chunked(light_curves, chunk_size):
            fut = exe.submit(process_chunk, chunk, chunk_idx, DEFAULT_BAND, out_dir)
            futures[fut] = chunk_idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                if result:
                    chunk_files.append(result)
                    print(f"✓ Chunk {idx} completed → {result}")
                else:
                    print(f"⚠  Chunk {idx} returned None")
            except Exception as e:
                print(f"✗ Chunk {idx} failed: {e}")

    # Concatenate all chunk files into the final features DataFrame
    dfs = []
    for fn in sorted(chunk_files, key=lambda f: int(Path(f).stem.split("_")[-1])):
        try:
            dfs.append(pd.read_pickle(fn))
        except Exception as e:
            print(f"Failed to read {fn}: {e}")

    if not dfs:
        print("No chunk files found — nothing to combine.")
        return

    final_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    final_df.to_pickle(FEATURES_FILE)
    print(f"\nAll chunks combined → {FEATURES_FILE}")

    # Print sample
    if len(final_df) > 0:
        sample = final_df.iloc[0]
        print("\nSample of extracted features:")
        print(f"File: {sample['file_path']}")
        for name, value in zip(sample['feature_names'], sample['feature_values']):
            print(f"  {name}: {value:.6f}")
        print(f"\nSuccess rate: {len(final_df)/len(light_curves)*100:.1f}%")


if __name__ == "__main__":
    main()
