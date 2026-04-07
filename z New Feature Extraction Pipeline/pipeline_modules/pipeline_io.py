import numpy as np
import pandas as pd
import os
import h5py
from pathlib import Path
from collections import defaultdict
from scipy.stats import scoreatpercentile

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import FEATURES_FILE, KNOWN_LIGHT_CURVES, HDBSCAN_OUTPUT_DIR

def build_index_maps(features_df):
    """Build fast + consistent lookups for files."""
    file_paths = features_df['file_path'].values
    basenames  = np.array([os.path.basename(p) for p in file_paths])

    name_to_idxs = defaultdict(list)
    for i, b in enumerate(basenames):
        name_to_idxs[b].append(i)

    fullpath_to_idx = {p: i for i, p in enumerate(file_paths)}
    return file_paths, basenames, name_to_idxs, fullpath_to_idx

def resolve_unique_index(identifier, name_to_idxs, fullpath_to_idx, *, strict=False):
    """
    Resolve an identifier (full path or basename) to a unique row index.
    Returns an int index, or None if not found/ambiguous when strict=False.
    """
    # Prefer exact full-path match
    if identifier in fullpath_to_idx:
        return fullpath_to_idx[identifier]

    base = os.path.basename(identifier)
    idxs = name_to_idxs.get(base, [])

    if len(idxs) == 1:
        return idxs[0]

    if strict:
        if len(idxs) == 0:
            raise ValueError(f"No file matching {identifier!r} (basename {base!r}) in features_df.")
        raise ValueError(
            f"Ambiguous basename {base!r}: found {len(idxs)} matches. "
            f"Pass a full path or set strict=False to skip."
        )
    # non-strict: silently skip when not found or ambiguous
    raise IndexError
    return None

def warn_on_duplicate_basenames(name_to_idxs):
    dups = {k: v for k, v in name_to_idxs.items() if len(v) > 1}
    if dups:
        print("Warning: duplicate basenames detected; use full paths to disambiguate:")
        for k, v in sorted(dups.items(), key=lambda kv: kv[0]):
            print(f"  {k}: {len(v)} matches")

def load_features():
    """Load the extracted features from file."""
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    return pd.read_pickle(FEATURES_FILE)

def filter_features(features_df, selected_features):
    """Filter feature_values in features_df to only include selected features."""
    def filter_row(row):
        name_to_value = dict(zip(row['feature_names'], row['feature_values']))
        return np.array([name_to_value.get(f, np.nan) for f in selected_features])

    features_df = features_df.copy()
    features_df['feature_values'] = features_df.apply(filter_row, axis=1)
    features_df['feature_names'] = [selected_features] * len(features_df)
    return features_df

def prepare_light_curves(features_df):
    """Extract TIME/RATE/ERRM/ERRP from each row's light_curve field into plain DataFrames."""
    return [pd.DataFrame({
        'TIME': lc['TIME'],
        'RATE': lc['RATE'],
        'ERRM': lc['ERRM'],
        'ERRP': lc['ERRP']
    }) for lc in features_df['light_curve']]


def save_cluster_labels(features_df, hdbscan_labels, output_dir):
    """Save cluster labels and file paths to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    cluster_df = pd.DataFrame({
        'file_path': features_df['file_path'],
        'cluster_label': hdbscan_labels
    })
    output_file = os.path.join(output_dir, 'cluster_assignments.csv')
    cluster_df.to_csv(output_file, index=False)
    print(f"Cluster assignments saved to: {output_file}")
    return output_file


def save_known_light_curve_features(features_df, output_dir):
    """Write feature values for KNOWN_LIGHT_CURVES to a tab-separated text file."""
    known_mask = features_df['basename'].isin(KNOWN_LIGHT_CURVES)
    known_df   = features_df.loc[known_mask]
    if known_df.empty:
        print("Warning: no known curves found in features_df!")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_txt = os.path.join(output_dir, "known_light_curves_features.txt")
    feature_names = features_df['feature_names'].iloc[0]

    with open(out_txt, 'w') as f:
        f.write("curve\t" + "\t".join(feature_names) + "\n")
        for _, row in known_df.iterrows():
            vals = row['feature_values']
            f.write(row['basename'] + "\t" +
                    "\t".join(str(x) for x in vals.tolist()) +
                    "\n")
    print(f"Wrote known‐curve features to: {out_txt}")


def apply_bexvar_filter(features_df, percentile=93):
    """
    Filter features_df to the top (100-percentile)% of sources by bexvar value.

    To re-enable in main(), replace the dropna line with:
        features_df = apply_bexvar_filter(features_df, percentile=93)
    """
    from scipy.stats import scoreatpercentile

    n_before = len(features_df)
    bexvar_values = features_df.apply(
        lambda row: row['feature_values'][row['feature_names'].index('bexvar')]
        if 'bexvar' in row['feature_names'] else 0,
        axis=1
    ).values

    threshold = scoreatpercentile(bexvar_values, percentile)
    print(f"Empirical {percentile}% threshold for bexvar: {threshold:.4f}")

    features_df = features_df[
        features_df.apply(
            lambda row: row['feature_values'][row['feature_names'].index('bexvar')] >= threshold
            if 'bexvar' in row['feature_names'] else False,
            axis=1
        )
    ].copy()

    print(f"Retained {len(features_df)} / {n_before} light curves after bexvar filtering.")
    return features_df


def save_analysis_results(
    features_df,
    hdbscan_labels,
    outlier_results,
    feature_matrix,
    feature_names,
    output_dir,
    umap_embedding=None
):
    """
    Save analysis results to HDF5 files for efficient web visualization.

    Args:
        features_df (pd.DataFrame): DataFrame containing features
        hdbscan_labels (np.ndarray): HDBSCAN clustering labels
        outlier_results (dict): Dictionary containing outlier detection results
        feature_matrix (np.ndarray): Matrix of feature values
        feature_names (list): List of feature names
        output_dir (str): Directory to save results
        umap_embedding (tuple): Optional tuple of (x, y) coordinates from UMAP embedding

    Returns:
        tuple: Paths to the created HDF5 files (features_file, light_curves_file)
    """
    # Create web data directory if it doesn't exist
    web_data_dir = Path(output_dir) / 'web_data'
    web_data_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    features_file = web_data_dir / 'features.h5'
    light_curves_file = web_data_dir / 'light_curves.h5'

    # Save light curves to HDF5
    with h5py.File(light_curves_file, 'w') as f:
        for i, lc in enumerate(features_df['light_curve']):
            file_path = features_df['file_path'].iloc[i]
            file_name = os.path.basename(file_path)

            # Create a group for each light curve
            grp = f.create_group(f'lc_{i}')
            grp.attrs['file_name'] = file_name
            grp.attrs['file_path'] = file_path

            # Store light curve data with compression
            grp.create_dataset('time', data=lc['TIME'].values, compression='gzip')
            grp.create_dataset('rate', data=lc['RATE'].values, compression='gzip')

            if 'ERRM' in lc:
                grp.create_dataset('errm', data=lc['ERRM'].values, compression='gzip')
            if 'ERRP' in lc:
                grp.create_dataset('errp', data=lc['ERRP'].values, compression='gzip')

    # Save features and metadata to HDF5
    with h5py.File(features_file, 'w') as f:
        # Store feature matrix with compression
        f.create_dataset('feature_matrix', data=feature_matrix.astype('float32'), compression='gzip')

        # Store feature names as fixed-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('feature_names', data=np.array(feature_names, dtype=dt))

        f.create_dataset('hdbscan_labels', data=hdbscan_labels.astype('int32'), compression='gzip')

        # Store outlier results in a group
        grp = f.create_group('outliers')
        grp.create_dataset('is_outlier', data=outlier_results['combined_outlier'].astype('bool'), compression='gzip')
        grp.create_dataset('iso_scores', data=outlier_results['iso_score'].astype('float32'), compression='gzip')
        grp.create_dataset('lof_scores', data=outlier_results['lof_score'].astype('float32'), compression='gzip')

        # Store UMAP embedding if available
        if umap_embedding is not None:
            f.create_dataset('umap_x', data=umap_embedding[0].astype('float32'), compression='gzip')
            f.create_dataset('umap_y', data=umap_embedding[1].astype('float32'), compression='gzip')

    print(f"Saved analysis results to {features_file} and {light_curves_file}")
    return str(features_file), str(light_curves_file)
