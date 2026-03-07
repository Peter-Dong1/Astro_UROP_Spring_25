#!/usr/bin/env python3
"""
Analyze similar light curves and generate feature histograms.

This script performs two main tasks:
1. For each known light curve, find the top 100 most similar curves and their cluster assignments
2. Generate normalized histograms for all features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm
import pickle
from config import KNOWN_LIGHT_CURVES, DATA_DIR, FEATURES_FILE, HDBSCAN_OUTPUT_DIR
import seaborn as sns
from collections import Counter
from run_pipeline_on_features import plot_light_curve

import subprocess
from astropy.io import fits

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette('viridis')

# Output directories
BASE_OUTPUT_DIR = os.path.join(DATA_DIR, 'analysis_results')
CLUSTER_ASSIGNMENTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'cluster_assignments_237_real_clippedxvar')
HISTOGRAMS_DIR = os.path.join(BASE_OUTPUT_DIR, 'feature_histograms_237_real_clippedxvar')

SIG_NEV_FILE = os.path.join(os.path.dirname(FEATURES_FILE), 'SIG_NEV_mappings.pkl')
# directory to dump these new plots
SIG_NEV_PLOT_DIR = os.path.join(BASE_OUTPUT_DIR, 'sig_nev_analysis')

# Create output directories
os.makedirs(CLUSTER_ASSIGNMENTS_DIR, exist_ok=True)
os.makedirs(HISTOGRAMS_DIR, exist_ok=True)
os.makedirs(SIG_NEV_PLOT_DIR, exist_ok=True)

def load_features():
    """
    Load the extracted features from file and ensure proper formatting.

    Returns:
        DataFrame with features in a consistent format
    """
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    print(f"Loading features from: {FEATURES_FILE}")
    df = pd.read_pickle(FEATURES_FILE)

    # Debug: Print basic info about the loaded data
    print(f"Loaded DataFrame with shape: {df.shape}")
    print("\nDataFrame columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # Check if we have a 'feature_values' column that needs to be processed
    if 'feature_values' in df.columns and 'feature_names' in df.columns:
        print("\nFound 'feature_values' and 'feature_names' columns")
        try:
            # Check the type of the first element in feature_values
            first_feature = df['feature_values'].iloc[0]
            print(f"Type of first feature_values: {type(first_feature)}")
            if hasattr(first_feature, '__len__'):
                print(f"Length of first feature vector: {len(first_feature)}")

            # If feature_values is a list/array column, convert it to a 2D array
            if isinstance(first_feature, (list, np.ndarray)):
                print("Converting feature_values to 2D array...")
                try:
                    # Try to stack the feature vectors
                    X = np.vstack(df['feature_values'].values)
                    feature_names = df['feature_names'].iloc[0]  # Assuming all rows have same feature names

                    # Create a new DataFrame with features as columns
                    df_features = pd.DataFrame(X, columns=feature_names, index=df.index)

                    # Add back the original columns (excluding the ones we're replacing)
                    other_cols = [col for col in df.columns if col not in ['feature_values', 'feature_names']]
                    if other_cols:
                        df_features = pd.concat([df[other_cols], df_features], axis=1)

                    print(f"Successfully created feature matrix with shape: {df_features.shape}")
                    return df_features

                except Exception as e:
                    print(f"Error converting feature_values to 2D array: {str(e)}")

        except Exception as e:
            print(f"Error processing feature_values: {str(e)}")

    # If we get here, either no processing was needed or processing failed
    print("Returning original DataFrame without feature processing")
    return df

def get_top_similar_indices(features_df, known_lc_idx, top_n=100):
    """
    Find top N most similar light curves to the known light curve.

    Args:
        features_df: DataFrame containing features
        known_lc_idx: Index of the known light curve
        top_n: Number of similar curves to return

    Returns:
        List of indices of top N similar curves
    """
    try:
        print(f"Finding top {top_n} similar curves for index {known_lc_idx}")

        # Debug: Print DataFrame columns and first few rows
        print("\nDataFrame columns:", features_df.columns.tolist())
        print("\nFirst few rows of features_df:")
        print(features_df.head())

        # First, try to find numeric columns directly
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # If no numeric columns found, check if features are stored in a 'feature_values' column
        if not numeric_cols and 'feature_values' in features_df.columns:
            print("Found 'feature_values' column, attempting to extract features...")
            try:
                # If feature_values is a list/array column, convert it to a 2D array
                X = np.vstack(features_df['feature_values'].values)
                print(f"Successfully extracted {X.shape[1]} features from 'feature_values' column")

                # Calculate cosine similarity
                similarities = cosine_similarity(X[known_lc_idx:known_lc_idx+1], X).flatten()

                # Get top N similar indices (excluding self)
                top_indices = np.argsort(similarities)[::-1][1:top_n+1]
                top_indices = top_indices[top_indices < len(features_df)]

                print(f"Found {len(top_indices)} similar curves using 'feature_values'")
                return top_indices

            except Exception as e:
                print(f"Error processing 'feature_values': {str(e)}")
                return []

        # If we have numeric columns, use them for similarity
        if numeric_cols:
            print(f"Using {len(numeric_cols)} numeric features for similarity calculation")

            # Convert to numpy array, ensuring all values are numeric
            X = features_df[numeric_cols].astype(float).values

            # Verify the known_lc_idx is within bounds
            if known_lc_idx >= len(X):
                print(f"Error: Index {known_lc_idx} is out of bounds for feature matrix")
                return []

            # Calculate cosine similarity
            print("Calculating cosine similarities...")
            similarities = cosine_similarity(X[known_lc_idx:known_lc_idx+1], X).flatten()

            if len(similarities) <= 1:
                print("Warning: Not enough data points for similarity calculation")
                return []

            # Get indices of top N similar curves (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:top_n+1]
            top_indices = top_indices[top_indices < len(features_df)]

            print(f"Found {len(top_indices)} similar curves using numeric columns")
            return top_indices

        # If we get here, no valid features were found
        print("Error: Could not find any valid numeric features or feature_values column")
        return []

    except Exception as e:
        print(f"Error in get_top_similar_indices: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def plot_cluster_distribution(clusters, title, output_file):
    """
    Plot a bar chart showing the distribution of clusters.

    Args:
        clusters: List of cluster assignments
        title: Title for the plot
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Count occurrences of each cluster
    cluster_counts = Counter(clusters)

    # Sort clusters by frequency (descending)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    labels = [f'Cluster {k}' if k != -1 else 'Noise' for k, v in sorted_clusters]
    counts = [v for k, v in sorted_clusters]
    cluster_ids = [k for k, v in sorted_clusters]

    # Create bar plot
    bars = plt.bar(labels, counts, color=sns.color_palette('viridis', len(labels)))

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.title(f'Cluster Distribution\n{title}', fontsize=14, pad=20)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Light Curves', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster distribution plot to: {output_file}")

def analyze_cluster_assignments(features_df, run_numbers):
    """
    Analyze cluster assignments for top similar curves across multiple runs.

    Args:
        run_numbers: List of run numbers to analyze
    """

    # Find indices of known light curves
    known_indices = []
    for lc in KNOWN_LIGHT_CURVES:
        try:
            # Make sure we're working with strings for file paths
            if 'file_path' not in features_df.columns:
                print("Error: 'file_path' column not found in features DataFrame")
                return

            # Find matches using string contains
            mask = features_df['file_path'].astype(str).str.contains(os.path.basename(lc), na=False)
            matches = features_df.index[mask].tolist()

            if matches:
                known_indices.append((lc, matches[0]))
                print(f"Found known light curve: {lc} at index {matches[0]}")
            else:
                print(f"Warning: Could not find light curve: {lc}")
        except Exception as e:
            print(f"Error processing known light curve {lc}: {str(e)}")

    if not known_indices:
        print("No known light curves found to analyze")
        return

    for run_num in tqdm(run_numbers, desc="Processing runs"):
        try:
            run_dir = os.path.join(CLUSTER_ASSIGNMENTS_DIR, f'run_{run_num}')
            os.makedirs(run_dir, exist_ok=True)

            # Load cluster assignments for this run
            cluster_file = os.path.join(DATA_DIR, str(run_num), 'hdbscan_data', 'cluster_assignments.csv')
            if not os.path.exists(cluster_file):
                print(f"\nSkipping run {run_num}: {cluster_file} not found")
                continue

            print(f"\nLoading cluster assignments from: {cluster_file}")
            try:
                cluster_df = pd.read_csv(cluster_file)
                print(f"Loaded {len(cluster_df)} cluster assignments")
            except Exception as e:
                print(f"Error loading cluster assignments: {str(e)}")
                continue

            for lc_name, lc_idx in tqdm(known_indices, desc=f"Processing known curves for run {run_num}", leave=False):
                try:
                    print(f"\nProcessing known curve: {os.path.basename(lc_name)}")

                    # Get top 100 similar curves
                    similar_indices = get_top_similar_indices(features_df, lc_idx)
                    if not len(similar_indices):
                        print("No similar curves found")
                        continue

                    # Get file paths and cluster assignments for similar curves
                    similar_files = []
                    similar_clusters = []

                    for idx in similar_indices:
                        try:
                            file_path = features_df.iloc[idx]['file_path']
                            if not isinstance(file_path, str):
                                print(f"Warning: Non-string file path at index {idx}")
                                continue

                            file_name = os.path.basename(file_path)
                            similar_files.append(file_path)

                            # Find matching cluster
                            cluster_match = cluster_df[cluster_df['file_path'].str.contains(file_name, na=False, regex=False)]
                            if not cluster_match.empty:
                                cluster_val = cluster_match.iloc[0]['cluster_label']
                                if isinstance(cluster_val, (list, np.ndarray)):
                                    cluster_val = cluster_val[0]  # Take first element if it's a sequence
                                similar_clusters.append(int(cluster_val))
                            else:
                                similar_clusters.append(-1)  # -1 for not found

                        except Exception as e:
                            print(f"Error processing index {idx}: {str(e)}")
                            similar_clusters.append(-1)
                            continue

                    # Save results
                    if similar_files and similar_clusters:
                        base_name = Path(lc_name).stem

                        # Save text file with assignments
                        output_file = os.path.join(run_dir, f"{base_name}_assignments.txt")
                        try:
                            with open(output_file, 'w') as f:
                                f.write(f"Known Light Curve: {lc_name}\n")
                                f.write(f"Run: {run_num}\n")
                                f.write(f"Total similar curves found: {len(similar_files)}\n")
                                f.write("\nTop Similar Curves Cluster Assignments:\n")
                                f.write("="*50 + "\n")
                                for i, (file, cluster) in enumerate(zip(similar_files, similar_clusters), 1):
                                    f.write(f"{i:3d}. {os.path.basename(str(file))} - Cluster: {cluster}\n")
                            print(f"Saved results to: {output_file}")

                            # Create and save cluster distribution plot
                            plot_file = os.path.join(run_dir, f"{base_name}_cluster_distribution.png")
                            plot_title = f"Known Curve: {os.path.basename(lc_name)}"
                            plot_cluster_distribution(similar_clusters, plot_title, plot_file)

                        except Exception as e:
                            print(f"Error writing output files: {str(e)}")

                except Exception as e:
                    print(f"Error processing {lc_name} in run {run_num}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"Error in run {run_num}: {str(e)}")
            import traceback
            traceback.print_exc()

def generate_feature_histograms(features_df):
    """Generate normalized histograms for all features."""

    # Check if we have the expected feature structure (from run_feature_extraction.py)
    if 'feature_names' in features_df.columns and 'feature_values' in features_df.columns:
        # Original format: features are in 'feature_names' and 'feature_values' columns
        print("Processing features in 'feature_names'/'feature_values' format...")
        # Extract all unique feature names
        all_feature_names = set()
        for names in features_df['feature_names']:
            if isinstance(names, (list, np.ndarray)):
                all_feature_names.update(names)

        # Convert to list and sort for consistent ordering
        feature_names = sorted(list(all_feature_names))
        print(f"Found {len(feature_names)} unique features to plot")

        # Create a dictionary to store values for each feature
        feature_data = {name: [] for name in feature_names}

        # Collect values for each feature
        for idx, row in features_df.iterrows():
            if isinstance(row['feature_names'], (list, np.ndarray)) and \
               isinstance(row['feature_values'], (list, np.ndarray)) and \
               len(row['feature_names']) == len(row['feature_values']):
                for name, value in zip(row['feature_names'], row['feature_values']):
                    if isinstance(value, (int, float, np.number)):
                        feature_data[name].append(float(value))
    else:
        # New format: features are already in separate columns
        print("Processing features in wide format (one column per feature)...")
        # Identify numeric columns (potential features)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Found {len(numeric_cols)} numeric columns to plot")

        # Create feature data dictionary
        feature_data = {}
        for col in numeric_cols:
            feature_data[col] = features_df[col].dropna().values

    # Now plot histograms for each feature
    if not feature_data:
        print("No feature data found to plot!")
        return

    print(f"Starting to generate histograms for {len(feature_data)} features...")
    for feature, values in tqdm(feature_data.items(), desc="Generating histograms"):
        try:
            values = np.array(values)
            if len(values) == 0:
                print(f"Skipping {feature}: No valid numeric data")
                continue

            # Create and save linear scale histogram
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=50, density=False, alpha=0.7, color='skyblue')
            plt.title(f'Distribution of {feature} (Linear Scale)')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            linear_output = os.path.join(HISTOGRAMS_DIR, f"{feature}_linear.png")
            plt.savefig(linear_output, bbox_inches='tight', dpi=150)
            plt.close()

            # Create and save log scale histogram
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=50, density=False, alpha=0.7, color='lightcoral')
            plt.title(f'Distribution of {feature} (Log Scale)')
            plt.xlabel('Value')
            plt.ylabel('log10(Frequency + 1)')
            plt.yscale('log')
            plt.grid(True, alpha=0.3, which='both')
            plt.tight_layout()
            log_output = os.path.join(HISTOGRAMS_DIR, f"{feature}_log.png")
            plt.savefig(log_output, bbox_inches='tight', dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error generating histogram for {feature}: {str(e)}")
            import traceback
            traceback.print_exc()

def load_cluster_assignments(run_num):
    """Load HDBSCAN cluster assignments from the latest run."""
    # Load cluster assignments for this run
    cluster_file = os.path.join(DATA_DIR, str(run_num), 'hdbscan_data', 'cluster_assignments.csv')
    if not os.path.exists(cluster_file):
        print(f"Skipping run {run_num}: {cluster_file} not found")

    return pd.read_csv(cluster_file)

def generate_cluster_feature_histograms(features_df, run_num):
    """Generate histograms for features colored by the top 3 largest HDBSCAN clusters.
    Creates separate subfolders for each cluster's plots."""
    print("Starting to generate cluster feature histograms...")

    try:
        # Load features and cluster assignments
        cluster_df = load_cluster_assignments(run_num)

        if cluster_df is None:
            print("Error: Could not load cluster assignments")
            return

        # Get the top 3 largest clusters (excluding noise/outliers with label -1)
        cluster_counts = cluster_df['cluster_label'].value_counts()
        top_clusters = cluster_counts.head(3).index.tolist()

        if not top_clusters:
            print("No clusters found for visualization")
            return

        print(f"Found top 3 clusters: {top_clusters}")

        # Create subdirectories for each cluster
        cluster_dirs = {}
        for cluster in top_clusters:
            cluster_dir = os.path.join(HISTOGRAMS_DIR, f'cluster_{cluster}')
            os.makedirs(cluster_dir, exist_ok=True)
            cluster_dirs[cluster] = cluster_dir

        # Also create directories for combined plots
        combined_linear_dir = os.path.join(HISTOGRAMS_DIR, 'combined_linear')
        combined_log_dir = os.path.join(HISTOGRAMS_DIR, 'combined_log')
        os.makedirs(combined_linear_dir, exist_ok=True)
        os.makedirs(combined_log_dir, exist_ok=True)


        # Prepare feature data
        if 'feature_names' in features_df.columns and 'feature_values' in features_df.columns:
            # Original format: features are in 'feature_names' and 'feature_values' columns
            print("Processing features in 'feature_names'/'feature_values' format...")

            # Get all unique feature names
            all_feature_names = set()
            for names in features_df['feature_names']:
                if isinstance(names, (list, np.ndarray)):
                    all_feature_names.update(names)
            feature_names = sorted(list(all_feature_names))

            # Create a mapping from file path to feature values
            file_to_features = {}
            for idx, row in features_df.iterrows():
                if isinstance(row['feature_names'], (list, np.ndarray)) and \
                   isinstance(row['feature_values'], (list, np.ndarray)) and \
                   len(row['feature_names']) == len(row['feature_values']):
                    file_to_features[row['file_path']] = dict(zip(row['feature_names'], row['feature_values']))
        else:
            # New format: features are already in separate columns
            print("Processing features in wide format (one column per feature)...")
            # Identify numeric columns (potential features)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_names = numeric_cols

            # Create a mapping from file path to feature values
            file_to_features = {}
            for idx, row in features_df.iterrows():
                if 'file_path' in features_df.columns:
                    file_to_features[row['file_path']] = {col: row[col] for col in numeric_cols}

        if not feature_names:
            print("No features found for visualization")
            return

        # Create a mapping from file path to cluster label
        file_to_cluster = {}
        for _, row in cluster_df.iterrows():
            file_path = row['file_path']
            if isinstance(file_path, str):
                file_name = os.path.basename(file_path)
                file_to_cluster[file_name] = row['cluster_label']

        # Create a mapping from cluster to list of feature values for that cluster
        cluster_data = {cluster: {feature: [] for feature in feature_names} for cluster in top_clusters}

        # Populate cluster data
        for file_path, features in file_to_features.items():
            file_name = os.path.basename(file_path)
            if file_name in file_to_cluster:
                cluster = file_to_cluster[file_name]
                if cluster in top_clusters:
                    for feature in feature_names:
                        if feature in features and pd.notna(features[feature]) and np.isfinite(features[feature]):
                            cluster_data[cluster][feature].append(float(features[feature]))

        # Generate histograms for each feature and cluster
        for feature in tqdm(feature_names, desc="Generating cluster histograms"):
            try:
                # Create a figure with overlaid histograms for all clusters (linear scale)
                plt.figure(figsize=(12, 6))

                # Define more contrasting colors for clusters
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

                # Plot histograms for each cluster
                for i, cluster in enumerate(top_clusters):
                    values = cluster_data[cluster].get(feature, [])
                    if not values:
                        continue

                    # Use the predefined contrasting color for this cluster
                    color = colors[i % len(colors)]

                    # Special handling for excess variance
                    if 'excess_var' in feature.lower():
                        # Filter out non-positive values
                        # values = np.array([v for v in values if v > 0])
                        # if len(values) > 0:  # Only plot if we have positive values
                        #     # Use linear scale with focused range
                        min_val = 0
                        # max_val = np.percentile(values, 99.9)  # Focus on 99th percentile to avoid outliers
                        max_val = max(values)
                        bins = np.linspace(min_val, max_val, 800)  # Fewer bins for better visibility

                        # Plot with linear scale
                        plt.hist(values, bins=bins, alpha=0.6,
                                label=f'Cluster {cluster} (n={len(values)})',
                                color=color, edgecolor=color, linewidth=0.8)
                        plt.xlim(0, 1)

                        # Add a vertical line at the mean for reference
                        plt.axvline(np.mean(values), color=color, linestyle='--',
                                    alpha=0.7, linewidth=1.5)
                    else:
                        # Regular histogram for other features
                        plt.hist(values, bins=30, alpha=0.6,
                               label=f'Cluster {cluster} (n={len(values)})',
                               color=color, edgecolor=color, linewidth=0.8)

                plt.title(f'Distribution of {feature} by Cluster', fontsize=14)
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save the combined linear scale plot
                plt.savefig(os.path.join(combined_linear_dir, f'{feature}_clusters.png'),
                          bbox_inches='tight', dpi=150)
                plt.close()

                # Create a figure with overlaid histograms for all clusters (log scale)
                plt.figure(figsize=(12, 6))

                # Plot histograms for each cluster with log scale
                for i, cluster in enumerate(top_clusters):
                    values = cluster_data[cluster].get(feature, [])
                    if not values:
                        continue

                    # Use the same color as in the linear plot
                    color = colors[i % len(colors)]

                    # Special handling for excess variance
                    if 'excess_var' in feature.lower():
                        # Filter out non-positive values
                        # values = np.array([v for v in values if v > 0])
                        # if len(values) > 0:  # Only plot if we have positive values
                            # Use linear scale with focused range (same as linear plot)
                        min_val = 0
                        # max_val = np.percentile(values, 99.9)  # Focus on 99th percentile
                        max_val = max(values)
                        bins = np.linspace(min_val, max_val, 800)  # Same bin count as linear

                        # Plot with linear x-scale and log y-scale
                        plt.hist(values, bins=bins, alpha=0.6,
                                label=f'Cluster {cluster} (n={len(values)})',
                                color=color, edgecolor=color, linewidth=0.8)
                        plt.xlim(0, 1)

                        # Add a vertical line at the mean for reference
                        plt.axvline(np.mean(values), color=color, linestyle='--',
                                    alpha=0.7, linewidth=1.5)
                    else:
                        # Regular histogram for other features
                        plt.hist(values, bins=100, alpha=0.6,
                               label=f'Cluster {cluster} (n={len(values)})',
                               color=color, edgecolor=color, linewidth=0.8)

                # Set log scale on y-axis for all features
                plt.yscale('log')
                plt.ylabel('log10(Frequency + 1)', fontsize=12)

                # Set title and labels
                plt.title(f'Log Distribution of {feature} by Cluster', fontsize=14)
                plt.xlabel(feature, fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3, which='both')
                plt.tight_layout()

                # Save the combined log scale plot
                plt.savefig(os.path.join(combined_log_dir, f'{feature}_clusters_log.png'),
                          bbox_inches='tight', dpi=150)
                plt.close()

                # Save individual cluster plots (linear and log)
                for i, cluster in enumerate(top_clusters):
                    values = cluster_data[cluster].get(feature, [])
                    if not values:
                        continue

                    # Linear scale individual plot
                    plt.figure(figsize=(10, 6))
                    color = colors[i % len(colors)]

                    # Special handling for excess variance
                    if 'excess_var' in feature.lower():
                        # Filter out non-positive values
                        # values = np.array([v for v in values if v > 0])
                        # if len(values) > 0:  # Only plot if we have positive values
                        # Use linear scale with focused range
                        min_val = 0
                        # max_val = np.percentile(values, 99.9)  # Focus on 99th percentile to avoid outliers
                        max_val  = max(values)
                        bins = np.linspace(min_val, max_val, 800)  # Fewer bins for better visibility

                        # Plot with linear scale
                        plt.hist(values, bins=bins, alpha=0.7, color=color)
                        plt.xlim(0, 1)

                        # Add a vertical line at the mean for reference
                        plt.axvline(np.mean(values), color=color, linestyle='--',
                                    alpha=0.7, linewidth=1.5)
                    else:
                        # Regular histogram for other features
                        plt.hist(values, bins=100, alpha=0.7, color=color)
                    plt.title(f'Cluster {cluster} - {feature} (n={len(values)})', fontsize=12)
                    plt.xlabel(feature, fontsize=10)
                    plt.ylabel('Frequency', fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(cluster_dirs[cluster], f'{feature}_linear.png'),
                               bbox_inches='tight', dpi=150)
                    plt.close()

                    # Log scale individual plot
                    plt.figure(figsize=(10, 6))
                    plt.hist(values, bins=100, alpha=0.7, color=color)
                    plt.title(f'Cluster {cluster} - {feature} (n={len(values)})', fontsize=12)
                    plt.xlabel(feature, fontsize=10)
                    plt.ylabel('log10(Frequency + 1)', fontsize=10)
                    plt.yscale('log')
                    plt.grid(True, alpha=0.3, which='both')
                    plt.tight_layout()
                    plt.savefig(os.path.join(cluster_dirs[cluster], f'{feature}_log.png'),
                               bbox_inches='tight', dpi=150)
                    plt.close()

            except Exception as e:
                print(f"Error generating histograms for feature {feature}: {str(e)}")
                import traceback
                traceback.print_exc()
    finally:
        print('hi')


def analyze_sig_nev_and_examples(features_df, n_examples_per_bin=2):
    """
    1) Plot distribution of SIG_NEV across all light curves
    2) For each integer bin [i, i+1), select up to `n_examples_per_bin` light curves
       and plot their TIME vs. RATE.
    """
    # --- load the mapping: { file_path: {'sig_nev':…, 'excess_var':…}, … }
    with open(SIG_NEV_FILE, 'rb') as f:
        sig_map = pickle.load(f)

    # --- histogram of all values
    all_sig = np.array([v['sig_nev'] for v in sig_map.values()])
    plt.figure(figsize=(8,4))
    plt.hist(all_sig, bins=50, alpha=0.7)
    plt.xlabel('SIG_NEV')
    plt.ylabel('Count')
    plt.title('Distribution of SIG_NEV')
    plt.tight_layout()
    plt.savefig(os.path.join(SIG_NEV_PLOT_DIR, 'sig_nev_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved SIG_NEV distribution → {SIG_NEV_PLOT_DIR}/sig_nev_distribution.png")

    # --- now pick bins 0–1, 1–2, … up to ceil(max)
    max_bin     = int(np.ceil(all_sig.max()))
    max_samples = 25         # up to 25 curves per bin
    nrows, ncols = 5, 5      # 5 rows × 5 columns

    for i in range(max_bin):
        # collect all file_paths in this bin
        bin_files = [
            fp for fp, m in sig_map.items()
            if (m['sig_nev'] >= i and m['sig_nev'] < i+1)
        ]
        if not bin_files:
            continue

        total_in_bin = len(bin_files)

        # take up to 25
        sample_files = bin_files[:max_samples]

        # make a 5×5 grid
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(15, 15),
                                sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, fp in enumerate(sample_files):
            row = features_df.loc[features_df['file_path'] == fp]
            if row.empty:
                continue
            lc = row['light_curve'].iat[0]  # DataFrame with TIME, RATE, ERRM, ERRP

            title = (
                f"{os.path.basename(fp)}\n"
                f"SIG_NEV = {sig_map[fp]['sig_nev']:.3f}"
            )
            plot_light_curve(axes[idx], lc, title=title, is_outlier=False)

        # turn off any unused axes
        for ax in axes[len(sample_files):]:
            ax.axis('off')

        # overall title for the grid
        fig.suptitle(
            f"SIG_NEV ∈ [{i},{i+1}) — {total_in_bin} total curves, showing {len(sample_files)}",
            fontsize=14, y=1.02
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        outname = f"signev_bin_{i}_grid.png"
        fig.savefig(os.path.join(SIG_NEV_PLOT_DIR, outname),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Plotted up to {len(sample_files)} examples for bin [{i},{i+1}) → {SIG_NEV_PLOT_DIR}")

def analyze_sig_nev_by_cluster(features_df, run_num):
    """
    Plot SIG_NEV distribution for the top-3 HDBSCAN clusters in run `run_num`,
    and write per-cluster summary statistics to a text file.
    """
    # --- load sig_nev map
    with open(SIG_NEV_FILE, 'rb') as f:
        sig_map = pickle.load(f)

    # --- load clusters
    cluster_df = load_cluster_assignments(run_num)
    if cluster_df is None:
        print(f"Could not load cluster assignments for run {run_num}")
        return

    # --- pick top-3 clusters (include noise = -1)
    counts = cluster_df['cluster_label'].value_counts()
    top_clusters = counts.head(3).index.tolist()
    if not top_clusters:
        print("No clusters found")
        return
    print(f"Top 3 clusters: {top_clusters}")

    # --- build file→cluster map (basename → label)
    file2clus = {
        os.path.basename(row['file_path']): int(row['cluster_label'])
        for _, row in cluster_df.iterrows()
        if isinstance(row['file_path'], str)
    }

    # --- gather SIG_NEV per cluster
    cluster_signev = {c: [] for c in top_clusters}
    for fp, m in sig_map.items():
        bn = os.path.basename(fp)
        cl = file2clus.get(bn, None)
        if cl in top_clusters:
            cluster_signev[cl].append(m['sig_nev'])

    # --- plot histogram
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.figure(figsize=(8,5))
    for i, cl in enumerate(top_clusters):
        vals = np.array(cluster_signev[cl])
        if len(vals)==0:
            continue
        plt.hist(vals, bins=50, alpha=0.6,
                 label=f'Cluster {cl} (n={len(vals)})',
                 color=colors[i], edgecolor=colors[i], linewidth=0.5)
    plt.xlabel('SIG_NEV')
    plt.ylabel('Count')
    plt.title(f'SIG_NEV Distribution — Run {run_num}, Top-3 Clusters')
    plt.legend()
    plt.grid(alpha=0.3)
    out_png = os.path.join(SIG_NEV_PLOT_DIR, f'sig_nev_by_cluster_run{run_num}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SIG_NEV by-cluster histogram → {out_png}")

    # --- compute and dump summary stats
    stats_file = os.path.join(SIG_NEV_PLOT_DIR, f'sig_nev_stats_run{run_num}.txt')
    with open(stats_file, 'w') as f:
        f.write(f"SIG_NEV summary statistics for run {run_num}\n")
        f.write("="*40 + "\n\n")
        for cl in top_clusters:
            vals = np.array(cluster_signev[cl])
            if vals.size == 0:
                f.write(f"Cluster {cl}: no data\n\n")
                continue
            stats = {
                'count': len(vals),
                'mean':  np.mean(vals),
                'median': np.median(vals),
                'min':   np.min(vals),
                'max':   np.max(vals),
                'std':   np.std(vals, ddof=1),
            }
            f.write(f"Cluster {cl}:\n")
            f.write(f"  count : {stats['count']}\n")
            f.write(f"  mean  : {stats['mean']:.4f}\n")
            f.write(f"  median: {stats['median']:.4f}\n")
            f.write(f"  min   : {stats['min']:.4f}\n")
            f.write(f"  max   : {stats['max']:.4f}\n")
            f.write(f"  std   : {stats['std']:.4f}\n\n")
    print(f"Saved SIG_NEV statistics → {stats_file}")

def plot_sig_nev_sample_grid(sig_map, features_df,
                             lower, upper,
                             max_samples=25,
                             nrows=5, ncols=5,
                             output_dir=SIG_NEV_PLOT_DIR):
    """
    Plot up to `max_samples` light curves whose SIG_NEV ∈ [lower, upper)
    in an nrows×ncols grid, and save to output_dir.
    """
    # 1) select all file_paths in the requested range
    bin_files = [
        fp for fp, m in sig_map.items()
        if (m['sig_nev'] >= lower and m['sig_nev'] < upper)
    ]
    total_in_bin = len(bin_files)
    if total_in_bin == 0:
        print(f"No light curves with SIG_NEV ∈ [{lower},{upper})")
        return

    # 2) sample up to max_samples
    sample_files = bin_files[:max_samples]

    # 3) build the grid
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*3, nrows*2.5),
                             sharex=False, sharey=False)
    axes = axes.flatten()

    # 4) plot each sampled curve
    for idx, fp in enumerate(sample_files):
        row = features_df.loc[features_df['file_path'] == fp]
        if row.empty:
            continue
        lc = row['light_curve'].iat[0]

        # individual subplot title
        subtitle = (
            f"{os.path.basename(fp)}\n"
            f"SIG_NEV={sig_map[fp]['sig_nev']:.3f}"
        )
        plot_light_curve(axes[idx], lc, title=subtitle, is_outlier=False)

    # 5) turn off unused axes
    for ax in axes[len(sample_files):]:
        ax.axis('off')

    # 6) overall title
    fig.suptitle(
        f"SIG_NEV ∈ [{lower},{upper}) — {total_in_bin} total, showing {len(sample_files)}",
        fontsize=14, y=1.02
    )

    # 7) save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"signev_{lower:g}_{upper:g}_grid.png"
    outpath = os.path.join(output_dir, fname)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved SIG_NEV sample grid → {outpath}")

def main():
    features_df = load_features()

    print("\nGenerating feature histograms...")
    generate_feature_histograms(features_df)

    print("\nGenerating cluster feature histograms...")
    generate_cluster_feature_histograms(features_df, run_num=237)

    # Define run numbers to analyze
    run_numbers = range(306, 307)  # 210 to 222 inclusive

    print("Starting analysis of similar curves and cluster assignments...")
    analyze_cluster_assignments(features_df, run_numbers)

    # print("\nGenerating sig_nev histogram...")
    # analyze_sig_nev_and_examples(features_df)


    # with open(SIG_NEV_FILE, 'rb') as f:
    #     sig_map = pickle.load(f)
    # plot_sig_nev_sample_grid(sig_map, features_df, 0, 0.01)
    # plot_sig_nev_sample_grid(sig_map, features_df, 0.01, 1)
    # plot_sig_nev_sample_grid(sig_map, features_df, 0, 0.001)

    print("\nAnalysis complete!")
    print(f"Cluster assignments saved to: {CLUSTER_ASSIGNMENTS_DIR}")
    print(f"Feature histograms saved to: {HISTOGRAMS_DIR}")
    # print(f"SIG_NEV histograms saved to: {SIG_NEV_PLOT_DIR}")
    # print("\nPlotting SIG_NEV by top-3 clusters...")
    # analyze_sig_nev_by_cluster(features_df, run_num=237)

if __name__ == "__main__":
    main()
