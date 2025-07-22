import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
import hdbscan
import umap
import json
import h5py
import numpy as np
import os
from pathlib import Path
import pickle

from config import (
    FEATURES_FILE,
    HDBSCAN_OUTPUT_DIR,
    UMAP_OUTPUT_DIR,
    DATA_DIR,
    FEATURE_OUTPUT_DIR,
    KNOWN_LIGHT_CURVES,
    DEFAULT_CONTAMINATION,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_MIN_DIST,
    DEFAULT_N_COMPONENTS,
    number
)
DEFAULT_MIN_CLUSTER_SIZE = 5 # Smaller TODO: Realisitcally 20+
DEFAULT_EPSILON = 0.13
DEFAULT_EOM = 'eom'
DEFAULT_MIN_SAMPLES = 3

def load_features():
    """Load the extracted features from file."""
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    return pd.read_pickle(FEATURES_FILE)

def save_analysis_results(
    features_df,
    umap_labels,
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
        umap_labels (np.ndarray): UMAP clustering labels
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

        # Store clustering results
        if umap_labels is not None:
            f.create_dataset('umap_labels', data=umap_labels.astype('int32'), compression='gzip')
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
    #     },
    #     'metadata': {
    #         'num_samples': len(features_df),
    #         'num_features': len(feature_names)
    #     },
    #     'light_curves': light_curve_data
    # }

    # Add UMAP embedding if available
    if umap_embedding is not None:
        results['umap_embedding'] = {
            'x': umap_embedding[0].tolist(),
            'y': umap_embedding[1].tolist()
        }

    # Save results
    output_file = web_data_dir / 'analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"Analysis results saved to: {output_file}")
    return str(output_file)

def detect_outliers(df, contamination=0.05, known_light_curves=KNOWN_LIGHT_CURVES):
    """
    Detect outliers in rows of a DataFrame, where each row represents a light curve.
    Each row contains:
      - 'file_path': The file path of the light curve
      - 'feature_names': List of feature names
      - 'feature_values': Numpy array of numerical feature values

    Args:
    - df (pd.DataFrame): Input DataFrame with light curves.
    - contamination (float): The proportion of data expected to be outliers.

    Returns:
    - pd.DataFrame: Original DataFrame with additional columns:
        - 'is_outlier': Boolean flag indicating if the row is an outlier.
        - 'scaled_features': Scaled feature values for further analysis.
        - 'iso_score': Anomaly score from Isolation Forest.
        - 'lof_score': Negative outlier factor from LOF.
        - 'iqr_outlier': Boolean indicating IQR-based outlier.
        - 'z_score_outlier': Boolean indicating Z-score-based outlier.
        - 'combined_outlier': Combined outlier flag across methods.
        - 'outlier_rank': Rank of combined outliers (1 = most outlier-like).
    """
    print(f"\nStarting outlier detection with contamination={contamination}...")
    start_time = time.time()

    # Ensure required columns are present
    required_columns = ['file_path', 'feature_names', 'feature_values']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare features for outlier detection
    feature_matrix = []
    for idx, row in df.iterrows():
        feature_values = row['feature_values']
        if not isinstance(feature_values, np.ndarray):
            raise ValueError(f"Row {idx}: 'feature_values' must be a numpy array.")
        feature_matrix.append(feature_values)

    # Convert to numpy array for processing
    feature_matrix = np.vstack(feature_matrix)

    # Scale features
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Isolation Forest
    print("Running Isolation Forest algorithm...")
    iso_start = time.time()
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        max_samples='auto',
        n_estimators=250
    )
    iso_forest.fit(scaled_features)
    iso_pred = iso_forest.predict(scaled_features)
    iso_scores = iso_forest.decision_function(scaled_features)
    print(f"Isolation Forest completed in {time.time() - iso_start:.2f} seconds")\

    # Local Outlier Factor
    print("Running Local Outlier Factor algorithm...")
    lof_start = time.time()
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        metric='euclidean',
        p=2,
        novelty=False
    )
    lof_scores = lof.fit_predict(scaled_features)

    lof_neg_scores = -lof.negative_outlier_factor_  # Higher = more anomalous
    print(f"Local Outlier Factor completed in {time.time() - lof_start:.2f} seconds")

    # Calculate feature importance using LOF
    print("Calculating feature importance...")
    feat_imp_start = time.time()
    feature_importances = []
    num_features = scaled_features.shape[1]
    for i in range(num_features):
        # Remove one feature at a time
        reduced_features = np.delete(scaled_features, i, axis=1)

        # Recompute LOF on reduced feature set
        lof_reduced = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            metric='euclidean',
            p=2,
            novelty=False
        )
        lof_reduced.fit(reduced_features)
        lof_reduced_neg_scores = -lof_reduced.negative_outlier_factor_

        # Measure the difference in LOF scores
        score_difference = np.mean(np.abs(lof_neg_scores - lof_reduced_neg_scores))
        # if (i+1) % 2 == 0:
        #     print(f"  Feature importance progress: {i+1}/{num_features}")
        feature_importances.append((df['feature_names'][0][i], score_difference))

    # Rank features by importance
    sorted_feature_importances = sorted(
        feature_importances, key=lambda x: x[1], reverse=True
    )
    print("Feature importance ranking:")
    for feat, imp in sorted_feature_importances[:5]:
        print(f"  - {feat}: {imp:.4f}")

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame(sorted_feature_importances, columns=['feature', 'importance'])

    # IQR method
    iqr_mult = 5
    q1 = np.percentile(scaled_features, 25, axis=0)
    q3 = np.percentile(scaled_features, 75, axis=0)
    iqr = q3 - q1
    iqr_outliers = np.any(
        (scaled_features < (q1 - iqr_mult * iqr)) |
        (scaled_features > (q3 + iqr_mult * iqr)),
        axis=1
    )

    # Z-score method
    z_scores = np.abs((scaled_features - np.mean(scaled_features, axis=0)) /
                      np.std(scaled_features, axis=0))
    z_score_outliers = np.any(z_scores > 4, axis=1)

    # Combine outlier predictions
    combined_outliers = (
        ((iso_pred == -1) &
        (lof_scores == -1)) |
        (iqr_outliers &
        z_score_outliers)
    )

    # Compute outlier ranks based on Isolation Forest scores for outliers
    outlier_scores = iso_scores[combined_outliers]
    outlier_ranks = pd.Series(outlier_scores).rank(ascending=True).astype(int)

    # Add results to DataFrame
    df['scaled_features'] = list(scaled_features)
    df['iso_score'] = iso_scores
    df['lof_score'] = lof_scores  # Negative outlier factor (lower = more anomalous)
    df['iqr_outlier'] = iqr_outliers
    df['z_score_outlier'] = z_score_outliers
    df['combined_outlier'] = combined_outliers
    df['outlier_rank'] = None  # Initialize column with None
    df.loc[combined_outliers, 'outlier_rank'] = outlier_ranks.values

    # Store feature importance in the DataFrame
    df['feature_importance'] = [feature_importance_df] * len(df)

    # -----------------------------
    # Similarity-weighted re-ranking (NEW) # CHANGE HERE !!
    # -----------------------------
    if known_light_curves:
        df['basename'] = df['file_path'].apply(lambda p: os.path.basename(p))
        df['is_known'] = df['basename'].isin(known_light_curves)

        if df['is_known'].any():
            known_features = np.vstack(df[df['is_known']]['feature_values'].values)
            similarities = cosine_similarity(scaled_features, known_features)
            max_sim_to_known = similarities.max(axis=1)
            df['max_similarity_to_known'] = max_sim_to_known

            # Normalize
            iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
            cos_norm = (max_sim_to_known - max_sim_to_known.min()) / (max_sim_to_known.max() - max_sim_to_known.min())

            # flip bc how it works
            cos_norm = 1 - cos_norm

            # Blend scores
            alpha = 0.3
            df['custom_outlier_score'] = (1 - alpha) * iso_norm + alpha * cos_norm
            df['custom_outlier_rank'] = df['custom_outlier_score'].rank(ascending=True).astype(int)
        else:
            print("Warning: None of the known light curves were found in this dataset.")
            df['custom_outlier_score'] = df['iso_score']
            df['custom_outlier_rank'] = df['iso_score'].rank(ascending=True).astype(int)
    else:
        print("No known_light_curves provided — skipping similarity reweighting.")
        df['custom_outlier_score'] = df['iso_score']
        df['custom_outlier_rank'] = df['iso_score'].rank(ascending=True).astype(int)

    return df

def visualize_features(scaled_features, outliers, file_paths, output_file=None):
    """
    Visualize features in 2D using PCA.

    Args:
        scaled_features (np.ndarray): Scaled feature matrix
        outliers (np.ndarray): Boolean array indicating outliers
        file_paths (np.ndarray): Array of file paths for each data point
        output_file (str): Path to save the plot
    """
    print("\nVisualizing clusters using PCA...")
    start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        output_file = os.path.join(FEATURE_OUTPUT_DIR, "clusters_pca.png")

    # Ensure scaled_features is a proper 2D numpy array
    if len(scaled_features.shape) != 2:
        raise ValueError("scaled_features must be a 2D array")

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)

    # Create scatter plot
    plt.figure(figsize=(12, 8))


    # Identify known light curves
    known_indices = [i for i, path in enumerate(file_paths)
                    if any(k in path for k in KNOWN_LIGHT_CURVES)]

    # Plot normal points (excluding known light curves)
    normal_indices = [i for i in range(len(features_2d))
                     if not outliers[i] and i not in known_indices]
    if normal_indices:
        plt.scatter(
            features_2d[normal_indices, 0], features_2d[normal_indices, 1],
            c='blue', label='Normal', alpha=0.4, s=30
        )

    # Plot outliers (excluding known light curves)
    outlier_indices = [i for i in np.where(outliers)[0]
                      if i not in known_indices]
    if outlier_indices:
        plt.scatter(
            features_2d[outlier_indices, 0], features_2d[outlier_indices, 1],
            c='red', label='Outliers', alpha=0.6, s=30
        )

    # Plot known light curves with star markers
    for idx in known_indices:
        color = 'red' if outliers[idx] else 'blue'
        label = 'Known (Outlier)' if outliers[idx] else 'Known (Normal)'
        marker = '*'  # Star marker
        size = 200 if outliers[idx] else 150  # Larger size for known curves

        plt.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            c=color, marker=marker, s=size,
            edgecolors='black', linewidth=1,
            label=label if idx == known_indices[0] else ""  # Add label only once
        )

        # Add filename as annotation
        filename = os.path.basename(file_paths[idx])
        plt.annotate(
            filename,
            (features_2d[idx, 0], features_2d[idx, 1]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', alpha=0.8, facecolor='white')
        )

    # Create custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Outlier'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=12, label='Known (Normal)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='Known (Outlier)')
    ]

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(handles=legend_elements, loc='best')
    plt.title('PCA of Light Curve Features')
    plt.grid(True, alpha=0.3)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

    print(f"Visualization completed in {time.time() - start_time:.2f} seconds")

def run_umap_clustering(features_df, light_curves, n_neighbors=DEFAULT_N_NEIGHBORS, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
                       min_dist=DEFAULT_MIN_DIST, n_components=DEFAULT_N_COMPONENTS,
                       output_file=None):
    """
    Run UMAP clustering on the features.

    Args:
        features_df (pd.DataFrame): DataFrame with features
        light_curves (list): List of light curve DataFrames
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        n_components (int): Number of dimensions for UMAP
        output_file (str): Path to save the plot

    Returns:
        tuple: (cluster_labels, feature_matrix, umap_embedding)
    """
    print("\nStarting UMAP clustering...")
    start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        output_file = os.path.join(UMAP_OUTPUT_DIR, "umap_clusters.png")

    # Extract feature matrix
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # Scale features
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Normalize features to unit length for cosine similarity effect
    from sklearn.preprocessing import normalize
    normalized_features = normalize(scaled_features, norm='l2')

    # Run UMAP on normalized features
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        metric='euclidean'  # Use euclidean on normalized data for cosine-like behavior
    )
    embedding = reducer.fit_transform(normalized_features)

    # Cluster the embedding using HDBSCAN with euclidean metric on normalized data
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
        min_samples=DEFAULT_MIN_SAMPLES,
        metric='euclidean',  # Use euclidean on normalized data for cosine-like behavior
        cluster_selection_method=DEFAULT_EOM,
        cluster_selection_epsilon=DEFAULT_EPSILON

    )
    cluster_labels = clustering.fit_predict(embedding)

    # Identify known light curves
    file_paths = features_df['file_path'].values
    known_indices = [i for i, path in enumerate(file_paths)
                    if any(k in path for k in KNOWN_LIGHT_CURVES)]

    # Create the UMAP plot
    plt.figure(figsize=(12, 8))

    # Plot regular points (excluding known light curves)
    regular_indices = [i for i in range(len(embedding)) if i not in known_indices]
    if regular_indices:
        scatter = plt.scatter(
            embedding[regular_indices, 0], embedding[regular_indices, 1],
            c=np.array(cluster_labels)[regular_indices], cmap='Spectral',
            alpha=0.6, s=40, edgecolors='white', linewidths=0.5,
            label='Regular Points'
        )

    # Plot known light curves with star markers
    for idx in known_indices:
        color = plt.cm.Spectral((cluster_labels[idx] % 20) / 20.0)  # Get color from colormap
        plt.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=[color], marker='*', s=300,
            edgecolors='black', linewidth=1.5,
            label=f'Known: {os.path.basename(file_paths[idx])}'
        )

        # Add filename as annotation
        filename = os.path.basename(file_paths[idx])
        plt.annotate(
            filename,
            (embedding[idx, 0], embedding[idx, 1]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', alpha=0.8, facecolor='white')
        )

    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Clustering of Light Curves')
    plt.grid(True, alpha=0.3)

    # Adjust legend to avoid duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Save and show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

    # Create cluster sample plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_plots_dir = os.path.join(os.path.dirname(output_file), f"cluster_plots_{timestamp}")
    os.makedirs(cluster_plots_dir, exist_ok=True)

    # Get unique clusters (excluding noise points labeled as -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

    # Generate cluster statistics
    cluster_stats = []

    # Add statistics for each cluster
    for cluster in np.unique(cluster_labels):
        if cluster == -1:
            cluster_name = "Noise/Outliers"
        else:
            cluster_name = f"Cluster {cluster}"

        # Get indices of points in this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_size = len(cluster_indices)

        # Find known curves in this cluster
        known_in_cluster = []
        for idx in cluster_indices:
            if idx in known_indices:
                known_in_cluster.append(os.path.basename(file_paths[idx]))

        cluster_stats.append({
            'cluster': cluster_name,
            'size': cluster_size,
            'known_curves': known_in_cluster
        })

    # Save cluster statistics to a file
    stats_file = os.path.join(os.path.dirname(output_file), 'cluster_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("UMAP Cluster Statistics\n")
        f.write("======================\n\n")

        # Sort clusters by size (descending)
        cluster_stats_sorted = sorted(cluster_stats, key=lambda x: int(x['size']), reverse=True)

        min_str = f'Min Cluster Size: {DEFAULT_MIN_CLUSTER_SIZE}\n'
        def_eps = f'Default Epsilon: {DEFAULT_EPSILON}\n'

        f.write(min_str)
        f.write(def_eps)


        for stat in cluster_stats_sorted:
            f.write(f"{stat['cluster']}:\n")
            f.write(f"  Number of curves: {stat['size']}\n")
            if stat['known_curves']:
                f.write("  Known curves in this cluster:\n")
                for curve in stat['known_curves']:
                    f.write(f"    - {curve}\n")
            f.write("\n")

    print(f"Cluster statistics saved to: {stats_file}")

    # Second pass: create plots for each significant curve

    sig_plots_dir = os.path.join(UMAP_OUTPUT_DIR, "significant_curves")
    os.makedirs(sig_plots_dir, exist_ok=True)

    for sig_curve in KNOWN_LIGHT_CURVES:
        # Find the index of the significant curve
        try:
            sig_idx = np.where([sig_curve in path for path in file_paths])[0][0]
        except IndexError:
            print(f"Warning: Could not find {sig_curve} in features DataFrame")
            continue

        # Get the cluster of the significant curve
        cluster_num = cluster_labels[sig_idx]
        cluster_indices = np.where(cluster_labels == cluster_num)[0]

        # Determine how many subplots we need (up to 15, or fewer if cluster is smaller)
        n_curves = min(15, len(cluster_indices))
        n_cols = 5
        n_rows = (n_curves + n_cols - 1) // n_cols  # Ceiling division

        # Create figure with appropriate size for the grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), dpi=100)
        axes = axes.ravel()  # Flatten the axes array for easier iteration

        # Set background color for the figure
        fig.patch.set_facecolor('#f0f0f0')

        # Get cluster info for title
        file_name = os.path.basename(sig_curve)
        cluster_name = f"Cluster {cluster_num}" if cluster_num != -1 else "Noise"

        # Set a main title for the entire figure
        fig.suptitle(f"{cluster_name} (n={len(cluster_indices)} curves, showing {n_curves} samples)",
                    fontsize=16, y=1.02)

        # Plot each light curve in the cluster (up to 15)
        for i, idx in enumerate(cluster_indices[:n_curves]):
            lc = light_curves[idx]
            ax = axes[i]

            # Set axis background to white
            ax.set_facecolor('white')

            # Plot with error bars if available
            if 'ERRM' in lc and 'ERRP' in lc:
                ax.errorbar(lc['TIME'], lc['RATE'],
                         yerr=[lc['ERRM'], lc['ERRP']],
                         fmt='o',
                         color='#1f77b4',  # Blue color for all curves
                         ecolor='#888888',
                         elinewidth=0.8,
                         capsize=1.5,
                         capthick=0.8,
                         zorder=3)
            else:
                ax.plot(lc['TIME'], lc['RATE'],
                       'o',
                       color='#1f77b4',
                       linewidth=1.2,
                       zorder=3)

            # Customize the subplot appearance
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Rate (counts/s)', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, color='#e0e0e0', linestyle='-', alpha=0.7)

            # Remove top and right spines
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            # Add a small title with the curve number
            ax.set_title(f'Curve {i+1}', fontsize=10, pad=5)

        # Turn off any unused subplots
        for i in range(n_curves, len(axes)):
            axes[i].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Create output directory if it doesn't exist
        os.makedirs(sig_plots_dir, exist_ok=True)

        # Save with high resolution
        plot_file = os.path.join(
            sig_plots_dir,
            f'cluster_{cluster_num}_samples.png'
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Created sample plot for {cluster_name} with {n_curves} light curves')

    print(f"UMAP clustering completed in {time.time() - start_time:.2f} seconds")
    return cluster_labels, feature_matrix, embedding

def run_hdbscan(
    features,
    min_cluster_size,
    min_samples,
    cluster_selection_method,
    cluster_selection_epsilon
):
    """Wrap a single HDBSCAN fit/predict so we can call it twice."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(features)
    return clusterer, labels

def two_stage_hdbscan(
    normalized_features,
    first_pass_params: dict,
    second_pass_params: dict,
    size_threshold: int = 10000
):
    """
    1) Run HDBSCAN with the “loose” (first_pass_params) settings
    2) For any cluster > size_threshold, rerun HDBSCAN *on just that subset*
       with (tighter) second_pass_params
    3) Remap all of those little subclusters back into a single label array
    """
    # 1st pass
    _, labels = run_hdbscan(normalized_features, **first_pass_params)

    # We’ll fill this with the final label for every point:
    final_labels = labels.copy()
    next_label = labels.max() + 1

    # look for any cluster that’s “too big”
    for cid in set(labels):
        if cid < 0:
            continue   # skip noise
        idxs = np.where(labels == cid)[0]
        if len(idxs) <= size_threshold:
            continue

        # 2nd pass on the subset
        subset = normalized_features[idxs]
        _, sublabels = run_hdbscan(subset, **second_pass_params)

        # sublabels runs from -1,0,1,...  remap each 0,1,2… to new global labels
        for local in set(sublabels):
            if local < 0:
                # keep as noise
                final_labels[idxs[sublabels == local]] = cid
            else:
                final_labels[idxs[sublabels == local]] = next_label
                next_label += 1

    return final_labels

def run_hdbscan_clustering(
    features_df,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    min_samples=None,
    output_file=None,
    two_stage: bool = True,           # ← new toggle
    size_threshold: int = 10000       # ← optional threshold
):
    """
    Run HDBSCAN (1- or 2-stage) on the features.
    two_stage=True will do the “subclustering” pass on large clusters.
    """
    print("\nStarting HDBSCAN clustering...")
    start_time = time.time()

    # 1) prepare output path
    if output_file is None:
        output_file = os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_clusters.png")

    # 2) build feature array
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # 3) scale & normalize
    scaler = RobustScaler()
    scaled = scaler.fit_transform(feature_matrix)
    from sklearn.preprocessing import normalize
    normalized = normalize(scaled, norm='l2')

    # 4) pick params
    first_pass = {
        'min_cluster_size': min_cluster_size,
        'min_samples':       min_samples or min_cluster_size,
        'cluster_selection_method': DEFAULT_EOM,
        'cluster_selection_epsilon': DEFAULT_EPSILON
    }
    second_pass = {
        # tighten up however you like
        'min_cluster_size': max(1, int(min_cluster_size/1.5)),
        'min_samples':      max(1, int((min_samples or min_cluster_size)/1.5)),
        'cluster_selection_method': 'leaf',
        'cluster_selection_epsilon': DEFAULT_EPSILON / 3
    }

    # 5) run HDBSCAN (1- or 2-stage)
    if two_stage:
        cluster_labels = two_stage_hdbscan(
            normalized,
            first_pass_params=first_pass,
            second_pass_params=second_pass,
            size_threshold=size_threshold
        )
    else:
        _, cluster_labels = run_hdbscan(
            normalized,
            **first_pass
        )

    # 6) PCA for plotting
    pca = PCA(n_components=2)
    proj2 = pca.fit_transform(normalized)

    # 7) plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        proj2[:, 0], proj2[:, 1],
        c=cluster_labels, cmap='Spectral',
        alpha=0.8, s=50, edgecolors='white', linewidths=0.5
    )
    plt.colorbar(scatter)
    plt.xscale('symlog', linthresh=1e-2)
    plt.yscale('symlog', linthresh=1e-2)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    title = '2-Stage HDBSCAN' if two_stage else 'HDBSCAN'
    plt.title(f'{title} Clusters')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=15,    # you can tweak
        min_dist=0.1,      # ditto
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(normalized)
    file_paths = features_df['file_path'].values

    # 3) Identify known indices
    known_idxs = [i for i, p in enumerate(file_paths)
                    if any(k == os.path.basename(p) for k in KNOWN_LIGHT_CURVES)]

    # 4) Plot
    plt.figure(figsize=(12,8))

    # Regular points (not known)
    reg = [i for i in range(len(embedding)) if i not in known_idxs]
    if reg:
        plt.scatter(
            embedding[reg,0], embedding[reg,1],
            c=cluster_labels[reg],
            cmap='Spectral',
            alpha=0.6,
            s=40,
            edgecolors='white',
            linewidths=0.5,
            label='Light curves'
        )

    # Known light curves as big stars
    for idx in known_idxs:
        color = plt.cm.Spectral((cluster_labels[idx] % 20)/20.0)
        plt.scatter(
            embedding[idx,0], embedding[idx,1],
            c=[color],
            marker='*',
            s=250,
            edgecolors='black',
            linewidths=1.5,
            label=f"Known: {os.path.basename(file_paths[idx])}"
        )
        # annotate filename
        plt.annotate(
            os.path.basename(file_paths[idx]),
            (embedding[idx,0], embedding[idx,1]),
            xytext=(5,5), textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
        )

    plt.colorbar(label='HDBSCAN cluster')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('HDBSCAN clusters (UMAP projection)')
    plt.grid(alpha=0.3)

    # avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', bbox_to_anchor=(1,1))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote UMAP plot → {output_file}")

    print(f"HDBSCAN clustering completed in {time.time() - start_time:.2f} seconds")
    return cluster_labels, feature_matrix

def analyze_cluster_feature_importance(feature_matrix, feature_names, cluster_labels, output_dir=None):
    """
    Analyze feature importance for each cluster.

    Args:
        feature_matrix (np.ndarray): Feature matrix
        feature_names (list): Feature names
        cluster_labels (np.ndarray): Cluster labels
        output_dir (str): Directory to save output
    """
    if output_dir is None:
        output_dir = HDBSCAN_OUTPUT_DIR

    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_output_file = os.path.join(output_dir, f"cluster_feature_importance_{timestamp}.txt")

    cluster_importance = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        if cluster != -1:  # Skip noise points
            cluster_mask = cluster_labels == cluster
            other_mask = cluster_labels != cluster

            # Calculate feature importance using mean difference
            cluster_means = np.mean(feature_matrix[cluster_mask], axis=0)
            other_means = np.mean(feature_matrix[other_mask], axis=0)
            importance = np.abs(cluster_means - other_means)

            # Normalize importance scores
            importance = importance / np.sum(importance) * 100

            # Store results
            cluster_importance[f"Cluster {cluster}"] = pd.DataFrame({
                'feature': feature_names,
                'importance_percent': importance
            }).sort_values('importance_percent', ascending=False)

    # Write results to file
    with open(text_output_file, 'w') as f:
        f.write("Cluster Feature Importance Analysis\n")
        f.write("=" * 50 + "\n\n")

        if cluster_importance:
            for cluster, importance_df in cluster_importance.items():
                f.write(f"{cluster} is distinguished by:\n")
                for _, row in importance_df.head(5).iterrows():
                    f.write(f"  - {row['feature']}: {row['importance_percent']:.1f}% importance\n")
                f.write("\n")
        else:
            f.write("No significant cluster feature importance found.\n")

    print(f"Cluster feature importance saved to: {text_output_file}")

def plot_light_curve(ax, lc, title=None, is_outlier=False):
    """Plot a single light curve on the given axis."""
    ax.errorbar(lc['TIME'], lc['RATE'],
                yerr=[lc['ERRM'], lc['ERRP']],
                fmt='o', markersize=2,
                elinewidth=0.5, capsize=0,
                color='red' if is_outlier else 'blue')

    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('Time', fontsize=7)
    ax.set_ylabel('Rate', fontsize=7)

def create_grid_plots(light_curves, results, output_dir, timestamp, plot_num=200, per_file=25):
    """Create grid plots of outliers and regular light curves."""
    print("\nCreating grid plots of outliers and regular light curves...")
    grid_start_time = time.time()

    # Get indices of outliers and regular light curves
    outlier_indices = np.where(results['combined_outlier'])[0]
    regular_indices = np.where(~results['combined_outlier'])[0]

    # Select samples for plotting
    n_outliers = min(plot_num, len(outlier_indices))
    n_regular = per_file
    sampled_outliers = np.random.choice(outlier_indices, n_outliers, replace=False)
    sampled_regular = np.random.choice(regular_indices, n_regular, replace=False)

    # Create directories for plots
    grid_plots_dir = os.path.join(output_dir, f"grid_plots_{timestamp}")
    os.makedirs(grid_plots_dir, exist_ok=True)

    # Plot outliers in multiple files (25 plots per file, max 8 files)
    if len(sampled_outliers) > 0:
        # Calculate how many outlier files we need (max 8)
        n_outlier_files = min(8, (len(sampled_outliers) + per_file - 1) // per_file)

        for file_idx in range(n_outlier_files):
            start_idx = file_idx * per_file
            end_idx = min(start_idx + per_file, len(sampled_outliers))
            current_outliers = sampled_outliers[start_idx:end_idx]

            # Create 5x5 grid for current batch
            fig_outliers, axes_outliers = plt.subplots(5, 5, figsize=(20, 20))
            axes_outliers = axes_outliers.ravel()

            for i, idx in enumerate(current_outliers):
                if i < len(axes_outliers):
                    lc = light_curves[idx]
                    title = f"Outlier {start_idx + i + 1}"
                    plot_light_curve(axes_outliers[i], lc, title, is_outlier=True)

            # Hide unused subplots
            for i in range(len(current_outliers), len(axes_outliers)):
                axes_outliers[i].axis('off')

            plt.tight_layout()
            outlier_grid_file = os.path.join(grid_plots_dir, f"outlier_grid_{file_idx + 1}.png")
            fig_outliers.savefig(outlier_grid_file, dpi=300)
            plt.close(fig_outliers)
            print(f"Outlier grid plot {file_idx + 1} saved to: {outlier_grid_file}")
    else:
        print("No outliers found to plot")

    # Plot regular curves (25 plots per file)
    if len(sampled_regular) > 0:
        # Calculate how many regular files we need
        n_regular_files = (len(sampled_regular) + per_file - 1) // per_file

        for file_idx in range(n_regular_files):
            start_idx = file_idx * per_file
            end_idx = min(start_idx + per_file, len(sampled_regular))
            current_regular = sampled_regular[start_idx:end_idx]

            # Create 5x5 grid for current batch
            fig_regular, axes_regular = plt.subplots(5, 5, figsize=(20, 20))
            axes_regular = axes_regular.ravel()

            for i, idx in enumerate(current_regular):
                if i < len(axes_regular):
                    lc = light_curves[idx]
                    title = f"Regular {start_idx + i + 1}"
                    plot_light_curve(axes_regular[i], lc, title)

            # Hide unused subplots
            for i in range(len(current_regular), len(axes_regular)):
                axes_regular[i].axis('off')

            plt.tight_layout()
            regular_grid_file = os.path.join(grid_plots_dir, f"regular_grid_{file_idx + 1}.png")
            fig_regular.savefig(regular_grid_file, dpi=300)
            plt.close(fig_regular)
            print(f"Regular grid plot {file_idx + 1} saved to: {regular_grid_file}")
    else:
        print("No regular light curves found to plot")

    print(f"Grid plots created in {time.time() - grid_start_time:.2f} seconds")
    return grid_plots_dir

def plot_cluster_samples(light_curves, features_df, cluster_labels, output_dir, timestamp):
    """
    Create grid plots for each cluster showing sample light curves.
    """
    print("\nCreating cluster sample plots...")
    cluster_start_time = time.time()

    # Create directory for cluster plots
    cluster_plots_dir = os.path.join(output_dir, f"cluster_plots_{timestamp}")
    os.makedirs(cluster_plots_dir, exist_ok=True)

    # Get unique clusters (not excluding noise points labeled as -1)
    unique_clusters = np.unique(cluster_labels)
    # unique_clusters = unique_clusters[unique_clusters != -1]

    for cluster in unique_clusters:
        # Get indices of light curves in this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]

        # Sample up to 10 light curves from this cluster
        sample_size = min(10, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)

        # Create subplot grid
        n_rows = (sample_size + 4) // 5
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4*n_rows))
        axes = axes.ravel()

        # Plot each sampled light curve
        for i, idx in enumerate(sampled_indices):
            if i < len(axes):
                lc = light_curves[idx]
                title = f"Sample {i+1}"
                plot_light_curve(axes[i], lc, title)
            else:
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(sampled_indices), len(axes)):
            axes[i].axis('off')

        cluster_name = f"Cluster_{cluster}"
        plt.suptitle(f'{cluster_name} Samples ({len(cluster_indices)} Found)',
                   fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        cluster_file = os.path.join(cluster_plots_dir, f"{cluster_name}_samples.png")
        fig.savefig(cluster_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Created sample plot for {cluster_name} with {sample_size} light curves")

    print(f"Cluster sample plots created in {time.time() - cluster_start_time:.2f} seconds")
    return cluster_plots_dir

def plot_significant_curves_with_cluster(significant_curves, light_curves, features_df, cluster_labels, output_dir):
    """
    Plot significant light curves alongside other members of their clusters.

    Args:
        significant_curves: List of file paths to significant light curves
        light_curves: List of light curve data dictionaries
        features_df: DataFrame containing features and file paths
        cluster_labels: Array of cluster assignments for each light curve
        output_dir: Directory to save output files

    Returns:
        Path to directory containing the generated plots
    """
    print("\nPlotting significant curves with their clusters...")
    sig_start_time = time.time()

    # Create directories for output
    sig_plots_dir = os.path.join(output_dir, "significant_curves")
    os.makedirs(sig_plots_dir, exist_ok=True)

    # File to store cluster information
    cluster_info_file = os.path.join(output_dir, "cluster_statistics.txt")

    # Get file paths from features DataFrame
    file_paths = features_df['file_path'].values

    # Dictionary to store cluster information
    cluster_info = {}
    known_curve_clusters = {}

    # First pass: collect cluster information
    unique_clusters = np.unique(cluster_labels)
    for cluster_num in unique_clusters:
        cluster_name = f"Cluster {cluster_num}"
        cluster_indices = np.where(cluster_labels == cluster_num)[0]
        cluster_info[cluster_num] = {
            'name': cluster_name,
            'size': len(cluster_indices),
            'contains_known': False,
            'known_curves': []
        }

    # Track which cluster each known curve is in
    for sig_curve in significant_curves:
        try:
            sig_idx = np.where([sig_curve in path for path in file_paths])[0][0]
            cluster_num = cluster_labels[sig_idx]
            known_curve_clusters[sig_curve] = cluster_num

            # Update cluster info
            if cluster_num in cluster_info:
                cluster_info[cluster_num]['contains_known'] = True
                cluster_info[cluster_num]['known_curves'].append(os.path.basename(sig_curve))
        except IndexError:
            print(f"Warning: Could not find {sig_curve} in features DataFrame")

    # Write cluster statistics to file
    with open(cluster_info_file, 'w') as f:
        f.write("Cluster Statistics\n")
        f.write("==================\n\n")
        min_str = f'Min Cluster Size: {DEFAULT_MIN_CLUSTER_SIZE}\n'
        def_eps = f'Default Epsilon: {DEFAULT_EPSILON}\n'
        f.write(min_str)
        f.write(def_eps)

        # Write known curve information first
        f.write("Known Light Curves and Their Clusters:\n")
        f.write("-" * 40 + "\n")
        for curve, cluster_num in known_curve_clusters.items():
            f.write(f"{os.path.basename(curve)}: {cluster_info[cluster_num]['name']}\n")

        # Write cluster statistics
        f.write("\nCluster Summary:\n")
        f.write("-" * 40 + "\n")
        for cluster_num, info in sorted(cluster_info.items()):
            f.write(f"{info['name']}: {info['size']} light curves")
            if info['contains_known']:
                f.write(f" (Contains known curves: {', '.join(info['known_curves'])})")
            f.write("\n")

        # Write noise cluster info
        if -1 in cluster_info:
            noise_count = cluster_info[-1]['size']
            f.write(f"\nLight curves in noise cluster: {noise_count} ({noise_count/len(cluster_labels):.1%})\n")

    print(f"\nCluster statistics saved to: {cluster_info_file}")

    # Second pass: create plots for each significant curve
    for sig_curve in significant_curves:
        # Find the index of the significant curve
        try:
            sig_idx = np.where([sig_curve in path for path in file_paths])[0][0]
        except IndexError:
            print(f"Warning: Could not find {sig_curve} in features DataFrame")
            continue

        # Get the cluster of the significant curve
        cluster_num = cluster_labels[sig_idx]
        cluster_indices = np.where(cluster_labels == cluster_num)[0]

        # Determine how many subplots we need (up to 15, or fewer if cluster is smaller)
        n_curves = min(15, len(cluster_indices))
        n_cols = 5
        n_rows = (n_curves + n_cols - 1) // n_cols  # Ceiling division

        # Create figure with appropriate size for the grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), dpi=100)
        axes = axes.ravel()  # Flatten the axes array for easier iteration

        # Set background color for the figure
        fig.patch.set_facecolor('#f0f0f0')

        # Get cluster info for title
        file_name = os.path.basename(sig_curve)
        cluster_name = f"Cluster {cluster_num}" if cluster_num != -1 else "Noise"

        # Set a main title for the entire figure
        fig.suptitle(f"{cluster_name} (n={len(cluster_indices)} curves, showing {n_curves} samples)",
                    fontsize=16, y=1.02)

        # Plot each light curve in the cluster (up to 15)
        for i, idx in enumerate(cluster_indices[:n_curves]):
            lc = light_curves[idx]
            ax = axes[i]

            # Set axis background to white
            ax.set_facecolor('white')

            # Plot with error bars if available
            if 'ERRM' in lc and 'ERRP' in lc:
                ax.errorbar(lc['TIME'], lc['RATE'],
                         yerr=[lc['ERRM'], lc['ERRP']],
                         fmt='o',
                         color='#1f77b4',  # Blue color for all curves
                         ecolor='#888888',
                         elinewidth=0.8,
                         capsize=1.5,
                         capthick=0.8,
                         zorder=3)
            else:
                ax.plot(lc['TIME'], lc['RATE'],
                       'o',
                       color='#1f77b4',
                       linewidth=1.2,
                       zorder=3)

            # Customize the subplot appearance
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Rate (counts/s)', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, color='#e0e0e0', linestyle='-', alpha=0.7)

            # Remove top and right spines
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            # Add a small title with the curve number
            ax.set_title(f'Curve {i+1}', fontsize=10, pad=5)

        # Turn off any unused subplots
        for i in range(n_curves, len(axes)):
            axes[i].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Create output directory if it doesn't exist
        os.makedirs(sig_plots_dir, exist_ok=True)

        # Save with high resolution
        plot_file = os.path.join(
            sig_plots_dir,
            f'cluster_{cluster_num}_samples.png'
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Created sample plot for {cluster_name} with {n_curves} light curves')

    print(f'Finished creating cluster sample plots in {time.time() - sig_start_time:.2f} seconds')
    return sig_plots_dir

def plot_top_similar_curves(light_curves, features_df, known_light_curves, output_dir=None, n_similar=25):
    """Plot the top N most similar light curves for each known light curve."""
    print("\nPlotting top similar curves...")
    similar_start_time = time.time()

    if output_dir is None:
        output_dir = os.getcwd()

    # Create directory for similarity plots
    similar_plots_dir = os.path.join(output_dir, "similar_curves")
    os.makedirs(similar_plots_dir, exist_ok=True)

    # Get feature matrix and file paths
    feature_matrix = np.vstack(features_df['feature_values'].values)
    file_paths = features_df['file_path'].values

    # Calculate similarity for each known light curve
    for known_lc in known_light_curves:
        try:
            # Find the known light curve in our dataset
            known_idx = np.where([os.path.basename(fp) == known_lc for fp in file_paths])[0][0]
        except IndexError:
            print(f"Warning: Known light curve {known_lc} not found in dataset")
            continue

        # Calculate cosine similarity with all other light curves
        known_features = feature_matrix[known_idx].reshape(1, -1)
        similarities = cosine_similarity(known_features, feature_matrix)[0]

        # Get indices of top similar curves (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        top_similarities = similarities[top_indices]

        # Create subplot grid
        n_rows = (n_similar + 4) // 5
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4*n_rows))
        axes = axes.ravel()

        # Plot the known light curve first
        plot_light_curve(axes[0], light_curves[known_idx],
                        title="Reference Curve", is_outlier=True)

        # Plot similar curves
        for i, (idx, sim_score) in enumerate(zip(top_indices, top_similarities)):
            if i + 1 < len(axes):  # +1 because we used the first plot for reference
                plot_light_curve(axes[i+1], light_curves[idx],
                               title=f'Similarity: {sim_score:.3f}')

        # Hide unused subplots
        for i in range(len(top_indices) + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Most Similar Curves to {known_lc}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save plot
        output_file = os.path.join(
            similar_plots_dir,
            f'similar_curves_{os.path.splitext(known_lc)[0]}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Created similarity plot for {known_lc}")

    print(f"Similarity plots created in {time.time() - similar_start_time:.2f} seconds")
    return similar_plots_dir

def plot_correlation_matrix(feature_matrix, feature_names, output_file=None):
    """
    Create and visualize a correlation matrix for the extracted features.

    Parameters:
        feature_matrix (np.ndarray): Matrix of feature values (rows=samples, columns=features)
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        output_file (str): Path to save the plot (if None, a default name will be used)

    Returns:
        str: Path to the saved correlation matrix plot
    """
    # Create a DataFrame with the features
    feature_df = pd.DataFrame(feature_matrix, columns=feature_names)

    # Calculate correlation matrix
    corr_matrix = feature_df.corr()

    # Create the heatmap
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle

    # Generate the heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,  # Show correlation values
        mask=mask,   # Only show the lower triangle
        cmap='coolwarm',  # Color map
        vmin=-1, vmax=1,  # Correlation range
        square=True,      # Make cells square
        linewidths=0.5,   # Width of cell borders
        fmt='.2f'         # Format for correlation values
    )

    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()

    # Save the plot
    if output_file is None:
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"correlation_matrix.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to: {output_file}")

    return output_file

def plot_feature_pairplot(feature_matrix, feature_names, cluster_labels, output_file=None, remove_noise=True):
    """
    Create a pairplot of features color-coded by clusters using seaborn.

    Parameters:
        feature_matrix (np.ndarray): Matrix of features where each row is a light curve and each column is a feature
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        cluster_labels (np.ndarray): Cluster labels from clustering algorithm
        output_file (str): Path to save the plot (if None, a default name will be used)
        remove_noise (bool): Whether to exclude noise points (cluster label -1) from the plot

    Returns:
        str: Path to the saved pairplot
    """
    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"corner_plot")

    # Get the output directory from the output file path
    output_dir = os.path.dirname(output_file)

    # Create a DataFrame with the features and cluster labels
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_features['cluster'] = cluster_labels

    # Convert cluster labels to string for better visualization
    df_features['cluster'] = df_features['cluster'].apply(lambda x: f'Cluster {x}' if x >= 0 else 'Noise')

    # Filter out noise points if requested
    if remove_noise:
        df_features_filtered = df_features[df_features['cluster'] != 'Noise']
        if len(df_features_filtered) == 0:
            print("Warning: All points were classified as noise. Showing all data points.")
            df_features_filtered = df_features
        else:
            print(f"Removed {len(df_features) - len(df_features_filtered)} noise points from the pairplot.")
    else:
        df_features_filtered = df_features
        print("Including noise points in the pairplot.")

    # Create a custom palette for the clusters
    unique_clusters = df_features_filtered['cluster'].unique()
    n_clusters = len(unique_clusters)

    # Create a colormap for the clusters
    cluster_colors = plt.get_cmap('viridis', max(3, n_clusters))
    palette = {}

    # Assign colors to clusters
    for i, cluster in enumerate(unique_clusters):
        palette[cluster] = cluster_colors(i)

    # Create the corner plot
    noise_status = "Noise Excluded" if remove_noise else "Noise Included"
    print(f"Creating corner plot of features colored by clusters ({noise_status})...")

    output1 = output_file + f"_reg_{noise_status}.png"

    # Create the pairplot with cluster coloring
    corner_plot1 = sns.pairplot(
        df_features_filtered,
        hue='cluster',
        palette=palette,
        plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'none'},
        diag_kind='kde',
        corner=True,  # False - full pairplot, True - corner plot
    )

    # Set log scale for all axes
    for ax in corner_plot1.axes.flatten():
        if ax is not None:
            ax.set_xscale('symlog', linthresh=1e-2)
            ax.set_yscale('symlog', linthresh=1e-2)

    # Add a main title for the entire figure
    corner_plot1.fig.suptitle(f'Feature Relationships by Cluster ({noise_status})', fontsize=24, y=1.02)

    # Save the plot
    corner_plot1.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"Corner plot 1 saved to: {output1}")

    output2 = output_file + f"_KD_{noise_status}.png"

    # Create the KD pairplot with cluster coloring
    corner_plot2 = sns.pairplot(
        df_features_filtered,
        hue='cluster',
        palette=palette,
        plot_kws={'alpha': 0.7},
        kind='kde', # reduce the number of contours - levels/significatnce etc
        diag_kind='kde',
        corner=True,  # False - full pairplot, True - corner plot
    )

    # Set log scale for all axes
    for ax in corner_plot2.axes.flatten():
        if ax is not None:
            ax.set_xscale('symlog', linthresh=1e-2)
            ax.set_yscale('symlog', linthresh=1e-2)

    # Add a main title for the entire figure
    corner_plot2.fig.suptitle(f'Feature Relationships by Cluster ({noise_status})', fontsize=24, y=1.02)

    # Save the plot
    corner_plot2.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"Corner plot 2 saved to: {output2}")

    return output_file

def save_cluster_size_histogram(file_paths, labels, outdir, tag):
    """
    Given an array of cluster labels,
     1) save 'tag_cluster_sizes.txt' with each label and its count
     2) save 'tag_cluster_sizes.png' bar‐plot of that distribution
    """
    os.makedirs(outdir, exist_ok=True)

    # compute counts
    counts = pd.Series(labels).value_counts().sort_index()

    # figure out which clusters contain known curves
    # map basename -> cluster
    basename_to_cluster = {
        os.path.basename(fp): lbl
        for fp, lbl in zip(file_paths, labels)
    }
    known_clusters = { basename_to_cluster[lc]
                       for lc in KNOWN_LIGHT_CURVES
                       if lc in basename_to_cluster }

    # 1) dump to text
    txtfile = os.path.join(outdir, f"{tag}_cluster_sizes.txt")
    with open(txtfile, 'w') as f:
        f.write("cluster_label\tcount\tknown?\n")
        for lbl, cnt in counts.items():
            mark = "*" if lbl in known_clusters else ""
            f.write(f"{lbl}\t{cnt}\t{mark}\n")
    print(f"→ Wrote cluster counts to {txtfile}")

    # 2) bar‐plot with hatching for known clusters
    plt.figure(figsize=(12,5))
    bars = plt.bar(
        counts.index.astype(str),
        counts.values,
        log=True
    )
    # hatch the bars that correspond to known clusters
    for bar, lbl in zip(bars, counts.index):
        if lbl in known_clusters:
            bar.set_hatch("///")

    plt.xlabel("Cluster label")
    plt.ylabel("Number of light curves")
    plt.title(f"Cluster‐size distribution ({tag})")
    plt.tight_layout()

    pngfile = os.path.join(outdir, f"{tag}_cluster_sizes.png")
    plt.savefig(pngfile, dpi=150)
    plt.close()
    print(f"→ Wrote cluster histogram to {pngfile}")

def plot_subcluster_histograms(orig_labels, new_labels,
                               threshold, outdir):
    """
    For every original cluster in orig_labels with size > threshold,
    build and save a bar‐histogram of how its points got reassigned
    in new_labels.

    Args:
        orig_labels (array‐like of int): 1st‐pass cluster IDs
        new_labels  (array‐like of int): 2nd‐pass cluster IDs (same length)
        threshold   (int): minimum number of points in orig cluster to plot
        outdir      (str): directory to save PNGs
    """
    os.makedirs(outdir, exist_ok=True)
    orig = pd.Series(orig_labels, name="orig")
    new  = pd.Series(new_labels, name="sub")

    # group by original cluster
    for cluster_id, idxs in orig.groupby(orig).groups.items():
        size = len(idxs)
        if size <= threshold:
            continue

        # count how many points went into each subcluster
        subcounts = new.iloc[list(idxs)].value_counts().sort_index()

        # plot
        plt.figure(figsize=(8,4))
        subcounts.plot(kind='bar', log=True)
        plt.title(f"Orig cluster {cluster_id} (n={size}) → {len(subcounts)} subclusters")
        plt.xlabel("Subcluster label")
        plt.ylabel("Count (log scale)")
        plt.tight_layout()
        fname = os.path.join(outdir, f"subcluster_hist_{cluster_id}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Wrote {fname}")

def main():
    """Main function to run the analysis pipeline."""
    print("Starting light curve analysis pipeline...")
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = timestamp[5:11] + "_" + timestamp[10:12]

    try:
        # Load features
        features_df = load_features()
        print(f"Loaded {len(features_df)} light curves")

        # Extract light curves from features DataFrame
        light_curves = [pd.DataFrame({
            'TIME': lc['TIME'],
            'RATE': lc['RATE'],
            'ERRM': lc['ERRM'],
            'ERRP': lc['ERRP']
        }) for lc in features_df['light_curve']]

        # --- load SIG_NEV map ---------------------------------------
        # sig_nev_file = os.path.join(os.path.dirname(FEATURES_FILE), 'SIG_NEV_mappings.pkl')
        # with open(sig_nev_file, 'rb') as f:
        #     sig_map = pickle.load(f)

        # # annotate features_df
        # features_df['sig_nev'] = features_df['file_path'].map(
        #     lambda p: sig_map.get(p, {}).get('sig_nev', np.nan)
        # )

        # # mask of curves we keep (sig_nev >= 0.01)
        # keep_mask = features_df['sig_nev'] >= 0.001
        # n_before = len(features_df)
        # features_df = features_df.loc[keep_mask].reset_index(drop=True)
        # light_curves = [lc for lc, keep in zip(light_curves, keep_mask) if keep]
        # print(f"Filtered out {n_before - len(features_df)} curves with SIG_NEV < 0.01, {len(features_df)} remain")

        features_df['basename'] = features_df['file_path'].apply(lambda p: os.path.basename(p))

        # --- SAVE KNOWN LIGHT‐CURVE FEATURES TO TEXT ---
        known_mask = features_df['basename'].isin(KNOWN_LIGHT_CURVES)
        known_df   = features_df.loc[known_mask]
        if not known_df.empty:
            out_txt = os.path.join(HDBSCAN_OUTPUT_DIR, "known_light_curves_features.txt")
            os.makedirs(HDBSCAN_OUTPUT_DIR, exist_ok=True)

            # Grab the feature names (they’re the same for each row)
            feature_names = features_df['feature_names'].iloc[0]

            with open(out_txt, 'w') as f:
                # header
                f.write("curve\t" + "\t".join(feature_names) + "\n")
                # one line per known curve
                for _, row in known_df.iterrows():
                    vals = row['feature_values']
                    # basename then all its feature values
                    f.write(row['basename'] + "\t" +
                            "\t".join(str(x) for x in vals.tolist()) +
                            "\n")
            print(f"Wrote known‐curve features to: {out_txt}")
        else:
            print("Warning: no known curves found in features_df!")

        # Detect outliers
        results = detect_outliers(features_df)
        print("Finished Detecting Outliers")

        # Visualize outliers (make sure scaled_features is 2D)
        scaled_arr = np.vstack(results['scaled_features'].values)
        outlier_mask = results['combined_outlier'].values
        paths       = features_df['file_path'].values
        visualize_features(scaled_arr, outlier_mask, paths)

        # Run UMAP clustering
        umap_labels, feature_matrix, umap_embedding = run_umap_clustering(features_df, light_curves)

        # Run HDBSCAN clustering
        # hdbscan_labels, _ = run_hdbscan_clustering(features_df)
        hdbscan_labels, mat = run_hdbscan_clustering(
                        features_df,
                        two_stage=True,
                        size_threshold=10000
                    )

        # Save cluster assignments to CSV
        save_cluster_labels(
            features_df=features_df,
            hdbscan_labels=hdbscan_labels,
            output_dir=DATA_DIR + f'/{number}/hdbscan_data'
        )

        save_cluster_labels(
            features_df=features_df,
            hdbscan_labels=umap_labels,
            output_dir=DATA_DIR + f'/{number}/umap_data'
        )

        # single‐pass
        labels_single, _ = run_hdbscan_clustering(
            features_df,
            two_stage=False,
            output_file=os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_single_pass.png")
        )
        save_cluster_size_histogram(paths, labels_single, HDBSCAN_OUTPUT_DIR, "single_pass")

        # two‐stage
        labels_two, _ = run_hdbscan_clustering(
            features_df,
            two_stage=True,
            size_threshold=10000,
            output_file=os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_two_stage.png")
        )
        save_cluster_size_histogram(paths, labels_two, HDBSCAN_OUTPUT_DIR, "two_stage")

        # --- NEW: sample 100 light‐curves from the noise cluster (-1) into 4 files of 25 ---
        noise_idxs = np.where(labels_two == -1)[0]
        n_sample   = min(100, len(noise_idxs))
        if n_sample == 0:
            print("No noise points to plot.")
        else:
            sampled = np.random.choice(noise_idxs, n_sample, replace=False)
            os.makedirs(HDBSCAN_OUTPUT_DIR, exist_ok=True)
            for file_ix in range(4):
                # pick 25 (or whatever remains)
                start = file_ix * 25
                end   = min(start + 25, n_sample)
                sel   = sampled[start:end]
                if len(sel) == 0:
                    break

                fig, axes = plt.subplots(5, 5, figsize=(20, 20))
                axes = axes.ravel()
                for i, idx in enumerate(sel):
                    lc = light_curves[idx]
                    # reuse your plot_light_curve helper
                    plot_light_curve(axes[i], lc, title=f"Noise #{i+1}", is_outlier=True)
                # turn off unused panels
                for ax in axes[len(sel):]:
                    ax.axis("off")
                plt.tight_layout()

                out_fn = os.path.join(
                    HDBSCAN_OUTPUT_DIR,
                    f"noise_cluster_sample_{file_ix+1}.png"
                )
                fig.savefig(out_fn, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Wrote noise‐sample plot: {out_fn}")

        # plot only those original clusters with >10 000 members
        plot_subcluster_histograms(
            orig_labels=labels_single,
            new_labels=labels_two,
            threshold=10000,
            outdir=HDBSCAN_OUTPUT_DIR + "/subcluster_hists"
        )


        df_split = pd.DataFrame({
            "single": labels_single,
            "two":    labels_two
        })
        split_counts = df_split.groupby("single")["two"].nunique()
        clusters_that_split = split_counts[split_counts > 1].index.tolist()
        print("Original clusters subdivided by two-stage pass:", clusters_that_split)

        # --- NEW: Visualize each split on the UMAP embedding ---
        # (make sure you still have `umap_embedding` from your UMAP step)
        for orig in clusters_that_split:
            mask    = (labels_single == orig)
            coords  = umap_embedding[mask]   # shape (n_points, 2)
            new_lbls = labels_two[mask]

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                coords[:,0], coords[:,1],
                c=new_lbls, cmap="tab10",
                s=30, alpha=0.8, edgecolor="k", linewidth=0.2
            )
            plt.title(f"Cluster {orig} → split into {new_lbls.max()-new_lbls.min()+1} parts")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.colorbar(sc, label="Two-stage subcluster ID")
            plt.tight_layout()
            plt.show()

            # Create correlation matrix and pairplot
            feature_names = features_df['feature_names'].iloc[0]
            corr_matrix_file = plot_correlation_matrix(feature_matrix, feature_names)
            # pairplot_file1 = plot_feature_pairplot(feature_matrix, feature_names, hdbscan_labels, remove_noise=False)
            # pairplot_file2 = plot_feature_pairplot(feature_matrix, feature_names, hdbscan_labels, remove_noise=True)
            # Create grid plots of outliers and regular curves
            grid_plots_dir = create_grid_plots(
                light_curves, results, FEATURE_OUTPUT_DIR, timestamp
            )

            # Create cluster sample plots
            # cluster_plots_dir = plot_cluster_samples(
            #     light_curves, features_df, hdbscan_labels, HDBSCAN_OUTPUT_DIR, timestamp
            # )

            # Plot significant curves (known light curves) with their clusters
            sig_plots_dir = plot_significant_curves_with_cluster(
                KNOWN_LIGHT_CURVES, light_curves, features_df, hdbscan_labels, HDBSCAN_OUTPUT_DIR
            )

            # Plot similar curves to known light curves
            similar_plots_dir = plot_top_similar_curves(
                light_curves, features_df, KNOWN_LIGHT_CURVES, FEATURE_OUTPUT_DIR
            )

            # Analyze feature importance for HDBSCAN clusters
            analyze_cluster_feature_importance(
                feature_matrix, feature_names, hdbscan_labels
            )

            print(f"\nAnalysis pipeline completed in {time.time() - start_time:.2f} seconds")
            print("\nOutput directories:")
            print(f"Grid plots: {grid_plots_dir}")
            # print(f"Cluster plots: {cluster_plots_dir}")
            print(f"Significant curve plots: {sig_plots_dir}")
            print(f"Similar curves plots: {similar_plots_dir}")
            print(f"Correlation matrix: {corr_matrix_file}")
            # print(f"Feature pairplot (Noise Included): {pairplot_file1}")
            # print(f"Feature pairplot (Noise Excluded): {pairplot_file2}")

            # Extract UMAP embedding coordinates for web visualization
            umap_x = umap_embedding[:, 0] if umap_embedding is not None else None
            umap_y = umap_embedding[:, 1] if umap_embedding is not None else None
            umap_coords = (umap_x, umap_y) if umap_x is not None and umap_y is not None else None


            # Save results for web visualization using HDF5
            # features_file, light_curves_file = save_analysis_results(
            #     features_df,
            #     umap_labels,
            #     hdbscan_labels,
            #     results,
            #     feature_matrix,
            #     feature_names,
            #     output_dir=DATA_DIR,
            #     umap_embedding=umap_embedding
            # )
            # print(f"Saved features to: {features_file}")
            # print(f"Saved light curves to: {light_curves_file}")

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        raise

def save_cluster_labels(features_df, hdbscan_labels, output_dir):
    """
    Save cluster labels along with file paths to a CSV file.

    Args:
        features_df (pd.DataFrame): DataFrame containing file paths
        hdbscan_labels (np.ndarray): Cluster labels from HDBSCAN
        output_dir (str): Directory to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a DataFrame with file paths and cluster labels
    cluster_df = pd.DataFrame({
        'file_path': features_df['file_path'],
        'cluster_label': hdbscan_labels
    })

    # Save to CSV
    output_file = os.path.join(output_dir, 'cluster_assignments.csv')
    cluster_df.to_csv(output_file, index=False)
    print(f"Cluster assignments saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    main()
