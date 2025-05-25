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
from pathlib import Path

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
    DEFAULT_MIN_CLUSTER_SIZE
)

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
    output_dir
):
    """
    Save analysis results to JSON files for web visualization.

    Args:
        features_df (pd.DataFrame): DataFrame containing features
        umap_labels (np.ndarray): UMAP clustering labels
        hdbscan_labels (np.ndarray): HDBSCAN clustering labels
        outlier_results (dict): Dictionary containing outlier detection results
        feature_matrix (np.ndarray): Matrix of feature values
        feature_names (list): List of feature names
        output_dir (str): Directory to save results
        timestamp (str): Timestamp for file naming
    """
    # Create web data directory if it doesn't exist
    web_data_dir = Path(output_dir) / 'web_data'
    web_data_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    results = {
        'features': {
            'names': feature_names,
            'matrix': feature_matrix.tolist()
        },
        'clustering': {
            'umap_labels': umap_labels.tolist() if umap_labels is not None else None,
            'hdbscan_labels': hdbscan_labels.tolist()
        },
        'outliers': {
            'is_outlier': outlier_results['combined_outlier'].tolist(),
            'outlier_scores': {
                'isolation_forest': outlier_results['iso_scores'].tolist(),
                'lof': outlier_results['lof_scores'].tolist()
            }
        },
        'metadata': {
            'num_samples': len(features_df),
            'num_features': len(feature_names)
        }
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

    # Plot normal points
    plt.scatter(
        features_2d[~outliers, 0], features_2d[~outliers, 1],
        c='blue', label='Normal', alpha=0.4
    )

    # Plot outliers with annotations
    outlier_points = plt.scatter(
        features_2d[outliers, 0], features_2d[outliers, 1],
        c='red', label='Outliers', alpha=0.6
    )

    # Add annotations for outliers
    for idx in np.where(outliers)[0]:
        filename = os.path.basename(file_paths[idx])
        plt.annotate(
            filename,
            (features_2d[idx, 0], features_2d[idx, 1]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.7
        )

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.title('Light Curves in 2D Feature Space')

    # Save and show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

    print(f"Visualization completed in {time.time() - start_time:.2f} seconds")

def run_umap_clustering(features_df, light_curves, n_neighbors=DEFAULT_N_NEIGHBORS,
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

    # Run UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    embedding = reducer.fit_transform(scaled_features)

    # Cluster the embedding using HDBSCAN
    clustering = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
    cluster_labels = clustering.fit_predict(embedding)

    # Create the UMAP plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=cluster_labels, cmap='Spectral',
        alpha=0.8, s=50, edgecolors='white', linewidths=0.5
    )
    plt.colorbar(scatter)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Clustering of Light Curves')

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

    # Plot samples for each cluster
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        sample_size = min(9, len(cluster_indices))  # Show up to 9 samples per cluster
        sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)

        # Create a grid plot
        n_rows = int(np.ceil(sample_size / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()

        # Plot each sample light curve
        for i, idx in enumerate(sample_indices):
            lc = light_curves[idx]
            plot_light_curve(axes[i], lc, title=f'Sample {i+1}')

        # Turn off any unused subplots
        for i in range(sample_size, len(axes)):
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

    print(f"UMAP clustering completed in {time.time() - start_time:.2f} seconds")
    return cluster_labels, feature_matrix, embedding

def run_hdbscan_clustering(features_df, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
                          min_samples=None, output_file=None):
    """
    Run HDBSCAN clustering on the features.

    Args:
        features_df (pd.DataFrame): DataFrame with features
        min_cluster_size (int): Minimum cluster size for HDBSCAN
        min_samples (int): Minimum samples for HDBSCAN
        output_file (str): Path to save the plot

    Returns:
        tuple: (cluster_labels, feature_matrix)
    """
    print("\nStarting HDBSCAN clustering...")
    start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        output_file = os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_clusters.png")

    # Extract feature matrix
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # Scale features
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(scaled_features)

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)

    # Create the plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=cluster_labels, cmap='Spectral',
        alpha=0.8, s=50, edgecolors='white', linewidths=0.5
    )
    plt.colorbar(scatter)
    plt.xscale('symlog', linthresh=1e-2)
    plt.yscale('symlog', linthresh=1e-2)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('HDBSCAN Clusters')

    # Save and show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

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

    # Get unique clusters (excluding noise points labeled as -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

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
    """Plot significant light curves alongside other members of their clusters."""
    print("\nPlotting significant curves with their clusters...")
    sig_start_time = time.time()

    # Create directory for significant curve plots
    sig_plots_dir = os.path.join(output_dir, "significant_curves")
    os.makedirs(sig_plots_dir, exist_ok=True)

    # Get file paths from features DataFrame
    file_paths = features_df['file_path'].values

    for sig_curve in significant_curves:
        # Find the index and cluster of the significant curve
        try:
            # Find the known light curve in our dataset
            sig_idx = np.where([os.path.basename(fp) == sig_curve for fp in file_paths])[0][0]
            cluster_num = cluster_labels[sig_idx]
        except IndexError:
            print(f"Warning: Significant curve {sig_curve} not found in dataset")
            continue

        if cluster_num == -1:
            print(f"Warning: Significant curve {sig_curve} is classified as noise")
            continue

        # Get other members of the same cluster
        cluster_indices = np.where(cluster_labels == cluster_num)[0]
        cluster_indices = cluster_indices[cluster_indices != sig_idx]  # Exclude the significant curve

        # Sample up to 8 other curves from the same cluster
        sample_size = min(8, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)

        # Create subplot grid (3x3)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()

        # Plot the significant curve in the center
        plot_light_curve(axes[4], light_curves[sig_idx],
                        title="Significant Curve", is_outlier=True)

        # Plot sampled cluster members around it
        for i, idx in enumerate(sampled_indices):
            plot_idx = i if i < 4 else i + 1  # Skip the center plot
            plot_light_curve(axes[plot_idx], light_curves[idx],
                           title=f"Cluster Member {i+1}")

        # Hide unused subplots
        for i in range(len(sampled_indices) + 1, 9):
            if i != 4:  # Don't hide the center plot
                axes[i].axis('off')

        plt.suptitle(f'Significant Curve and Cluster {cluster_num} Members',
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save plot
        plot_file = os.path.join(sig_plots_dir,
                                f"significant_curve_{os.path.splitext(sig_curve)[0]}_cluster_{cluster_num}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Created plot for significant curve {sig_curve} (Cluster {cluster_num})")

    print(f"Significant curve plots created in {time.time() - sig_start_time:.2f} seconds")
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
        output_file = os.path.join(similar_plots_dir, f'similar_curves_{os.path.splitext(known_lc)[0]}.png')
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

        # Detect outliers
        results = detect_outliers(features_df)
        print("Finished Detecting Outliers")

        # Visualize outliers
        visualize_features(results['scaled_features'], results['combined_outlier'], features_df['file_path'])

        # Run UMAP clustering
        umap_labels, feature_matrix, umap_embedding = run_umap_clustering(features_df, light_curves)

        # Run HDBSCAN clustering
        hdbscan_labels, _ = run_hdbscan_clustering(features_df)

        # Create correlation matrix and pairplot
        feature_names = features_df['feature_names'].iloc[0]
        corr_matrix_file = plot_correlation_matrix(feature_matrix, feature_names)
        pairplot_file1 = plot_feature_pairplot(feature_matrix, feature_names, hdbscan_labels, remove_noise=False)
        pairplot_file2 = plot_feature_pairplot(feature_matrix, feature_names, hdbscan_labels, remove_noise=True)
        # Create grid plots of outliers and regular curves
        grid_plots_dir = create_grid_plots(
            light_curves, results, FEATURE_OUTPUT_DIR, timestamp
        )

        # Create cluster sample plots
        cluster_plots_dir = plot_cluster_samples(
            light_curves, features_df, hdbscan_labels, HDBSCAN_OUTPUT_DIR, timestamp
        )

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
        print(f"Cluster plots: {cluster_plots_dir}")
        print(f"Significant curve plots: {sig_plots_dir}")
        print(f"Similar curves plots: {similar_plots_dir}")
        print(f"Correlation matrix: {corr_matrix_file}")
        print(f"Feature pairplot (Noise Included): {pairplot_file1}")
        print(f"Feature pairplot (Noise Excluded): {pairplot_file2}")

        web_results_file = save_analysis_results(
            features_df,
            umap_labels,
            hdbscan_labels,
            results,
            feature_matrix,
            feature_names,
            DATA_DIR
        )

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
