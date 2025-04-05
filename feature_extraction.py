import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from astropy.io import fits
import random
import math
import time
from datetime import datetime
import seaborn as sns

from helper import (
    load_light_curve,
    load_n_light_curves,
    load_all_fits_files,
    create_dataframe_of_light_curves,
    DEFAULT_DATA_DIR
)

size = 191_000
# Create output directory for plots if it doesn't exist
FILE_DIR = '/home/pdong/Astro UROP/plots/feature_extraction_plots' + f"/{size}"
# Create directories for both HDBSCAN and UMAT outputs
FEATURE_OUTPUT_DIR = FILE_DIR + "/FEATURES"
HDBSCAN_OUTPUT_DIR = FILE_DIR + "/HDBSCAN"
UMAT_OUTPUT_DIR = FILE_DIR + "/UMAT"
OUTPUT_DIR = HDBSCAN_OUTPUT_DIR  # Default to HDBSCAN for backward compatibility
os.makedirs(HDBSCAN_OUTPUT_DIR, exist_ok=True)
os.makedirs(UMAT_OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

print("Starting feature extraction process...")
print("Loading FITS files...")
start_time = time.time()
fits_files = load_all_fits_files()
print(f"Loaded {len(fits_files)} FITS files in {time.time() - start_time:.2f} seconds")

print("Loading light curves...")
start_time = time.time()
light_curves = load_n_light_curves(size, fits_files, band = "med")
print(f"Loaded {len(light_curves)} light curves in {time.time() - start_time:.2f} seconds")


def df_extract_statistical_features_error(df):
    """Extract statistical features from light curve with error handling"""
    # Print file being processed
    print(f"Processing file: {df.attrs.get('FILE_NAME', 'Unknown')}")
    """
    Extract statistical features from the light curve, accounting for measurement errors,
    and return as a single-row DataFrame.

    Parameters:
        df (pd.DataFrame): Light curve DataFrame with 'RATE', 'ERRM', and 'ERRP'.

    Returns:
        pd.DataFrame: A DataFrame containing 'file_path', 'feature_names', and 'feature_values'.
    """
    # Weighted Mean (Symmetric Errors)
    weights = 1 / df['SYM_ERR']**2  # Inverse variance weighting
    weighted_mean = np.sum(df['RATE'] * weights) / np.sum(weights)

    # Weighted Variance
    weighted_variance = np.sum(weights * (df['RATE'] - weighted_mean)**2) / np.sum(weights)

    # Robust Statistics
    median = np.median(df['RATE'])
    iqr = np.percentile(df['RATE'], 75) - np.percentile(df['RATE'], 25)

    # Beyond 1 sigma (error-aware)
    beyond_1_sigma = np.sum(np.abs(df['RATE'] - weighted_mean) > np.sqrt(weighted_variance)) / len(df['RATE'])

    # Median Absolute Deviation (MAD)
    mad = stats.median_abs_deviation(df['RATE'])

    # Skewness and Kurtosis
    # skewness = ((df['RATE'] - weighted_mean)**3 * weights).sum() / (weights.sum() * weighted_variance**1.5)
    # kurtosis = ((df['RATE'] - weighted_mean)**4 * weights).sum() / (weights.sum() * weighted_variance**2)

    # Additional Features
    max_rate = df['RATE'].max()
    min_rate = df['RATE'].min()
    max_amp = max_rate - min_rate

    # Flux Percentile Ratio (95th - 5th percentile)
    flux_percentile_ratio = np.percentile(df['RATE'], 95) - np.percentile(df['RATE'], 5)

    # Create a row with features
    # feature_names = [
    #     "weighted_mean", "weighted_variance", "median", "iqr", "beyond_1_sigma",
    #     "mad", "skewness", "kurtosis", "max_amp", "flux_percentile_ratio"
    # ]
    feature_names = [
        "weighted_mean", "weighted_variance", "median", "iqr", "beyond_1_sigma",
        "mad", "max_amp", "flux_percentile_ratio"
    ]
    # feature_values = np.array([
    #     weighted_mean, weighted_variance, median, iqr, beyond_1_sigma,
    #     mad, skewness, kurtosis, max_amp, flux_percentile_ratio
    # ])
    feature_values = np.array([
        weighted_mean, weighted_variance, median, iqr, beyond_1_sigma,
        mad, max_amp, flux_percentile_ratio
    ])

    # Create the DataFrame row
    return pd.DataFrame({
        "file_path": [df.attrs['FILE_NAME']],
        "feature_names": [feature_names],
        "feature_values": [feature_values]
    })

def df_process_all_light_curves_error(light_curves):
    """
    Process a list of DataFrames containing light curve data and combine results.

    Parameters:
        light_curves (list of pd.DataFrame): List of light curve DataFrames.

    Returns:
        pd.DataFrame: Combined DataFrame with 'file_path', 'feature_names', and 'feature_values'.
    """
    print(f"\nProcessing {len(light_curves)} light curves for feature extraction...")
    start_time = time.time()
    features_list = []

    # Add progress tracking
    for i, lc in enumerate(light_curves):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(light_curves) - i) / rate if rate > 0 else 0
            print(f"Progress: {i}/{len(light_curves)} ({i/len(light_curves)*100:.1f}%) - {rate:.1f} curves/sec - Est. remaining: {remaining:.1f} sec")
        features_list.append(df_extract_statistical_features_error(lc))

    # Concatenate all rows into a single DataFrame
    result = pd.concat(features_list, ignore_index=True)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return result


# This method works with a pandas dataframe consisting of more information
def df_detect_outliers_extreme(df, contamination=0.05):
    print(f"\nStarting outlier detection with contamination={contamination}...")
    start_time = time.time()
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

    return df


def df_visualize_clusters(scaled_features, combined_outliers, output_file=None):
    """
    Visualize clusters in 2D using PCA, highlighting normal points and outliers.

    Args:
    - scaled_features (np.ndarray): Scaled features of the dataset.
    - combined_outliers (np.ndarray): Boolean array indicating outliers.
    - output_file (str): Path to save the plot (if None, a default name will be used)
    """
    print("\nVisualizing clusters using PCA...")
    start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_clusters_{timestamp}.png")

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        features_2d[~combined_outliers, 0], features_2d[~combined_outliers, 1],
        c='blue', label='Normal', alpha=0.4
    )
    plt.scatter(
        features_2d[combined_outliers, 0], features_2d[combined_outliers, 1],
        c='red', label='Outliers', alpha=0.6
    )
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.title('Light Curves in 2D Feature Space')

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Also display the plot
    plt.show()
    print(f"Visualization completed in {time.time() - start_time:.2f} seconds")


def df_analyze_light_curves(light_curves, contamination=0.05, vis=True, output_file=None):
    print("\nStarting light curve analysis pipeline...")
    total_start_time = time.time()
    """
    Analyze light curves to extract features, detect outliers, and visualize results.

    Args:
    - light_curves (pd.DataFrame): DataFrame containing light curves.
    - contamination (float): Proportion of data expected to be outliers.
    - vis (bool): Whether to visualize the results.

    Returns:
    - pd.DataFrame: DataFrame with extracted features and outlier analysis results.
    """
    # Detect outliers using the provided function
    features = df_process_all_light_curves_error(light_curves)
    results = df_detect_outliers_extreme(features, contamination=contamination)

    scaled_features = np.vstack(results['scaled_features'].values)
    combined_outliers = results['combined_outlier'].values

    if vis:
        # Visualize results and save to file
        df_visualize_clusters(scaled_features, combined_outliers, output_file)

    return results
print("\n" + "="*50)
print("STARTING MAIN EXECUTION")
print("="*50)

# Define functions for grid plotting
def plot_light_curve(ax, lc, title=None, is_outlier=False):
    """Plot a single light curve on the given axis"""
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

def create_grid_plots(light_curves, results, output_dir, timestamp):
    """Create grid plots of outliers and regular light curves"""
    print("\nCreating grid plots of outliers and regular light curves...")
    grid_start_time = time.time()

    # Get indices of outliers and regular light curves
    outlier_indices = np.where(results['combined_outlier'])[0]
    regular_indices = np.where(~results['combined_outlier'])[0]

    # Select 25 outliers and 25 regular light curves
    # If there are fewer than 25 outliers, select all available
    n_outliers = min(25, len(outlier_indices))
    n_regular = min(25, len(regular_indices))

    # Sort outliers by outlier rank (most extreme first)
    if 'outlier_rank' in results.columns:
        # Lower rank means more extreme outlier
        outlier_indices = outlier_indices[np.argsort(results.loc[outlier_indices, 'outlier_rank'])]

    # Randomly select from regular if there are too many
    if len(regular_indices) > n_regular:
        regular_indices = np.random.choice(regular_indices, n_regular, replace=False)

    # Create a 5x5 grid for outliers
    if n_outliers > 0:
        fig_outliers, axes_outliers = plt.subplots(5, 5, figsize=(15, 15))
        axes_outliers = axes_outliers.flatten()

        for i in range(25):
            if i < n_outliers:
                idx = outlier_indices[i]
                lc = light_curves[idx]
                file_name = os.path.basename(results.iloc[idx]['file_path'])
                rank = results.iloc[idx].get('outlier_rank', 'N/A')
                title = f"Outlier #{i+1} (Rank: {rank})\n{file_name[-18:-9]}"
                plot_light_curve(axes_outliers[i], lc, title, is_outlier=True)
            else:
                axes_outliers[i].axis('off')  # Hide unused subplots

        plt.tight_layout()
        outlier_grid_file = os.path.join(output_dir, f"outlier_grid_{timestamp}.png")
        fig_outliers.savefig(outlier_grid_file, dpi=300)
        plt.close(fig_outliers)
        # print(f"Outlier grid plot saved to: {outlier_grid_file}")
    else:
        print("No outliers found to plot")

    # Create a 5x5 grid for regular light curves
    if n_regular > 0:
        fig_regular, axes_regular = plt.subplots(5, 5, figsize=(15, 15))
        axes_regular = axes_regular.flatten()

        for i in range(25):
            if i < n_regular:
                idx = regular_indices[i]
                lc = light_curves[idx]
                file_name = os.path.basename(results.iloc[idx]['file_path'])
                title = f"Regular #{i+1}\n{file_name}"
                plot_light_curve(axes_regular[i], lc, title)
            else:
                axes_regular[i].axis('off')  # Hide unused subplots

        plt.tight_layout()
        regular_grid_file = os.path.join(output_dir, f"regular_grid_{timestamp}.png")
        fig_regular.savefig(regular_grid_file, dpi=300)
        plt.close(fig_regular)
        # print(f"Regular grid plot saved to: {regular_grid_file}")
    else:
        print("No regular light curves found to plot")

    print(f"Grid plots created in {time.time() - grid_start_time:.2f} seconds")
    return outlier_grid_file, regular_grid_file

# Track overall execution time
overall_start_time = time.time()

# Run the analysis and save visualization
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_analysis_main_{timestamp}.png")
results = df_analyze_light_curves(light_curves, output_file=output_file)

# Print summary of results
print("\n" + "="*50)
print("EXECUTION SUMMARY")
print("="*50)


print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")
print(f"Processed {len(light_curves)} light curves")
print(f"Detected {results['combined_outlier'].sum()} outliers ({results['combined_outlier'].sum()/len(results)*100:.2f}%)")
print("Top 5 most important features:")
if 'feature_importance' in results.columns:
    top_features = results['feature_importance'].iloc[0].head(5)
    for idx, row in top_features.iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")

    # Save feature importance to a CSV file
    importance_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_importance_{timestamp}.csv")
    results['feature_importance'].iloc[0].to_csv(importance_file, index=False)
    print(f"Feature importance saved to: {importance_file}")
else:
    print("  Feature importance information not available")
print("="*50)

# Create grid plots of outliers and regular light curves
outlier_grid_file, regular_grid_file = create_grid_plots(light_curves, results, FEATURE_OUTPUT_DIR, timestamp)


# =============================================================================
# HDBSCAN Clustering Pipeline
# =============================================================================

def run_umat_clustering(light_curves, n_neighbors=15, min_dist=0.1, n_components=2, output_file=None):
    """
    Run UMAT (Uniform Manifold Approximation and Transformation) for dimensionality reduction
    and clustering on the statistical features of light curves.

    Parameters:
        light_curves (list of pd.DataFrame): List of light curve DataFrames.
        n_neighbors (int): The number of neighbors to consider for each point in UMAT.
        min_dist (float): The minimum distance between points in the embedding.
        n_components (int): The number of dimensions in the embedding.
        output_file (str): Path to save the plot (if None, a default name will be used).

    Returns:
        tuple: (cluster_labels, feature_matrix, umat_embedding)
    """
    # Import UMAP here to ensure it's available when needed
    # Note: You need to install umap-learn package first: pip install umap-learn
    try:
        import umap
    except ImportError:
        print("UMAP (umap-learn) is not installed. Please install it using:")
        print("pip install umap-learn")
        print("or")
        print("conda install -c conda-forge umap-learn")
        return None, None, None

    # For clustering after UMAT reduction
    from sklearn.cluster import DBSCAN

    print("\n" + "="*50)
    print("STARTING UMAT CLUSTERING PIPELINE")
    print("="*50)

    # Track execution time
    clustering_start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(UMAT_OUTPUT_DIR, f"umat_clusters_{timestamp}.png")

def run_hdbscan_clustering(light_curves, min_cluster_size=5, min_samples=None, output_file=None):
    """
    Run HDBSCAN clustering on the statistical features of light curves and visualize
    the clusters using PCA.

    Parameters:
        light_curves (list of pd.DataFrame): List of light curve DataFrames.
        min_cluster_size (int): The minimum size of clusters for HDBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        output_file (str): Path to save the plot (if None, a default name will be used).

    Returns:
        tuple: (cluster_labels, feature_matrix, pca_result)
    """
    import hdbscan  # Import here to ensure it's available when needed

    print("\n" + "="*50)
    print("STARTING HDBSCAN CLUSTERING PIPELINE")
    print("="*50)

    # Track execution time
    clustering_start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(HDBSCAN_OUTPUT_DIR, f"hdbscan_clusters_{timestamp}.png")

    # Extract statistical features
    print("Extracting statistical features for clustering...")
    features_df = df_process_all_light_curves_error(light_curves)

    # Prepare feature matrix
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # Scale features
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Run HDBSCAN
    print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")
    hdbscan_start = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass
    )
    cluster_labels = clusterer.fit_predict(scaled_features)
    print(f"HDBSCAN completed in {time.time() - hdbscan_start:.2f} seconds")

    # Get cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")

    # Get cluster sizes
    if n_clusters > 0:
        # print("Cluster sizes:")
        for i in range(n_clusters):
            cluster_size = list(cluster_labels).count(i)
            # print(f"  - Cluster {i}: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.2f}%)")

    # Reduce to 2D using PCA for visualization
    print("Reducing dimensions with PCA for visualization...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)

    # Create a colormap for visualization
    # -1 (noise) will be black, other clusters will have distinct colors
    unique_labels = set(cluster_labels)
    n_clusters_with_noise = len(unique_labels)

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis', max(3, n_clusters))

    # Plot the clusters
    plt.figure(figsize=(12, 10))

    # Plot each cluster with a different color
    for label in unique_labels:
        mask = cluster_labels == label
        if label == -1:  # Noise points in gray with higher transparency
            plt.scatter(
                features_2d[mask, 0], features_2d[mask, 1],
                c='gray', marker='.', label='Noise', alpha=0.2, s=160
            )
        else:  # Regular clusters with colors from colormap and higher visibility
            plt.scatter(
                features_2d[mask, 0], features_2d[mask, 1],
                c=np.array([cmap(label)]), label=f'Cluster {label}', alpha=0.9, s=60,
                edgecolors='w', linewidths=0.5
            )

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('HDBSCAN Clustering of Light Curves in 2D Feature Space')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add variance explained by PCA components
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Show the plot
    plt.show()

    print(f"HDBSCAN clustering completed in {time.time() - clustering_start_time:.2f} seconds")
    return cluster_labels, feature_matrix, features_2d

# Continue the UMAT function implementation
def continue_umat_clustering(light_curves, features_df, scaled_features, n_neighbors=15, min_dist=0.1, n_components=2, output_file=None, eps=0.5, min_samples=5):
    """
    Continue the UMAT clustering process with the extracted features.
    This is a helper function to be called after feature extraction.

    Parameters:
        light_curves (list): List of light curve DataFrames
        features_df (pd.DataFrame): DataFrame with features
        scaled_features (np.ndarray): Scaled feature matrix
        n_neighbors (int): Number of neighbors for UMAT
        min_dist (float): Minimum distance for UMAT
        n_components (int): Number of components for UMAT
        output_file (str): Output file path
        eps (float): Maximum distance between samples for DBSCAN
        min_samples (int): Minimum samples for DBSCAN

    Returns:
        tuple: (cluster_labels, feature_matrix, umat_embedding)
    """
    import umap
    from sklearn.cluster import DBSCAN

    # Track execution time
    umat_start = time.time()

    # Run UMAT for dimensionality reduction
    print(f"Running UMAT with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        random_state=42
    )
    umat_embedding = reducer.fit_transform(scaled_features)
    print(f"UMAT completed in {time.time() - umat_start:.2f} seconds")

    # Run DBSCAN on the UMAT embedding for clustering
    print(f"Running DBSCAN on UMAT embedding with eps={eps}, min_samples={min_samples}...")
    dbscan_start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(umat_embedding)
    print(f"DBSCAN completed in {time.time() - dbscan_start:.2f} seconds")

    # Get cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")

    # Get cluster sizes
    if n_clusters > 0:
        # print("Cluster sizes:")
        for i in range(n_clusters):
            cluster_size = list(cluster_labels).count(i)
            # print(f"  - Cluster {i}: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.2f}%)")

    # Create a colormap for visualization
    unique_labels = set(cluster_labels)
    n_clusters_with_noise = len(unique_labels)

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis', max(3, n_clusters))

    # Plot the clusters
    plt.figure(figsize=(12, 10))

    # Plot each cluster with a different color
    for label in unique_labels:
        mask = cluster_labels == label
        if label == -1:  # Noise points in gray with higher transparency
            plt.scatter(
                umat_embedding[mask, 0], umat_embedding[mask, 1],
                c='gray', marker='.', label='Noise', alpha=0.2, s=160
            )
        else:  # Regular clusters with colors from colormap and higher visibility
            plt.scatter(
                umat_embedding[mask, 0], umat_embedding[mask, 1],
                c=np.array([cmap(label)]), label=f'Cluster {label}', alpha=0.9, s=60,
                edgecolors='w', linewidths=0.5
            )

    plt.xlabel('UMAT Dimension 1')
    plt.ylabel('UMAT Dimension 2')
    plt.title('UMAT+DBSCAN Clustering of Light Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Show the plot
    plt.show()

    return cluster_labels, scaled_features, umat_embedding

def run_umap_clustering_complete(light_curves, n_neighbors=15, min_dist=0.1, n_components=2, output_file=None, eps=0.5, min_samples=5):
    """
    Complete UMAP clustering pipeline, from feature extraction to visualization.

    Parameters:
        light_curves (list): List of light curve DataFrames
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        n_components (int): Number of components for UMAP
        output_file (str): Output file path
        eps (float): Maximum distance between samples for DBSCAN
        min_samples (int): Minimum samples for DBSCAN

    Returns:
        tuple: (cluster_labels, feature_matrix, umat_embedding)
    """
    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(UMAT_OUTPUT_DIR, f"umat_clusters_{timestamp}.png")

    clustering_start_time = time.time()

    # Extract statistical features
    print("Extracting statistical features for clustering...")
    features_df = df_process_all_light_curves_error(light_curves)

    # Prepare feature matrix
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # Scale features
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Run the UMAT clustering
    cluster_labels, _, umat_embedding = continue_umat_clustering(
        light_curves, features_df, scaled_features,
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        output_file=output_file, eps=eps, min_samples=min_samples
    )

    print(f"UMAT clustering completed in {time.time() - clustering_start_time:.2f} seconds")
    return cluster_labels, feature_matrix, umat_embedding

def plot_cluster_samples(light_curves, features_df, cluster_labels, output_dir, timestamp):
    """
    Create grid plots for each cluster detected by HDBSCAN.
    For each cluster, randomly sample up to 10 light curves and display them in a 2x5 grid.

    Parameters:
        light_curves (list): List of light curve DataFrames
        features_df (pd.DataFrame): DataFrame with features
        cluster_labels (np.array): Cluster labels from HDBSCAN
        output_dir (str): Directory to save the plots
        timestamp (str): Timestamp for the filenames
    """
    print("\nCreating cluster sample plots...")
    cluster_start_time = time.time()

    # Create a subdirectory for the cluster plots
    cluster_plots_dir = os.path.join(output_dir, f"cluster_samples_{timestamp}")
    os.makedirs(cluster_plots_dir, exist_ok=True)
    print(f"Created directory for cluster samples: {cluster_plots_dir}")

    # Get unique cluster labels (excluding noise which is -1)
    unique_clusters = sorted(list(set(cluster_labels)))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    # Also create a plot for noise points
    all_cluster_labels = sorted(list(set(cluster_labels)))

    # Define a colormap for consistency with the main plot
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap('viridis', max(3, n_clusters))

    # Process each cluster (including noise)
    for cluster_label in all_cluster_labels:
        # Get indices of light curves in this cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]

        # Skip if no light curves in this cluster (shouldn't happen)
        if len(cluster_indices) == 0:
            continue

        # Determine cluster name and color
        if cluster_label == -1:
            cluster_name = "Noise"
            color = 'gray'
        else:
            cluster_name = f"Cluster_{cluster_label}"
            color = cmap(cluster_label)

        # Sample up to 10 light curves from this cluster
        sample_size = min(10, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)

        # Create a 2x5 grid for the sampled light curves
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        # Plot each sampled light curve
        for i, ax_idx in enumerate(range(10)):
            ax = axes[ax_idx]

            if i < sample_size:
                # Get the light curve and its file path
                lc_idx = sampled_indices[i]
                lc = light_curves[lc_idx]
                file_path = features_df.iloc[lc_idx]['file_path']
                file_name = os.path.basename(file_path)

                # Plot the light curve
                ax.errorbar(lc['TIME'], lc['RATE'],
                           yerr=[lc['ERRM'], lc['ERRP']],
                           fmt='o', markersize=2,
                           elinewidth=0.5, capsize=0,
                           color=color)

                # Set title and labels
                ax.set_title(f"{file_name[-18:-9]}", fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=6)

                if ax_idx >= 5:  # Only add x-label to bottom row
                    ax.set_xlabel('Time', fontsize=7)
                if ax_idx % 5 == 0:  # Only add y-label to leftmost column
                    ax.set_ylabel('Rate', fontsize=7)
            else:
                # Hide unused subplots
                ax.axis('off')

        # Add a main title for the entire figure
        if cluster_label == -1:
            plt.suptitle(f"Noise Points (Sample of {sample_size} out of {len(cluster_indices)} points)",
                       fontsize=14, y=0.98)
        else:
            plt.suptitle(f"Cluster {cluster_label} (Sample of {sample_size} out of {len(cluster_indices)} points)",
                       fontsize=14, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

        # Save the plot
        cluster_file = os.path.join(cluster_plots_dir, f"{cluster_name}_samples_{timestamp}.png")
        fig.savefig(cluster_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Created sample plot for {cluster_name} with {sample_size} light curves")

    print(f"Cluster sample plots created in {time.time() - cluster_start_time:.2f} seconds")
    return cluster_plots_dir

def plot_feature_corner_plot(feature_matrix, feature_names, cluster_labels, output_file=None):
    """
    Create a corner plot of features color-coded by HDBSCAN clusters using seaborn's pairplot.

    Parameters:
        feature_matrix (np.ndarray): Matrix of features where each row is a light curve and each column is a feature
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        cluster_labels (np.ndarray): Cluster labels from HDBSCAN clustering
        output_file (str): Path to save the plot (if None, a default name will be used)

    Returns:
        str: Path to the saved plot
    """

    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_corner_plot_{timestamp}.png")

    # Create a DataFrame with the features and cluster labels
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_features['cluster'] = cluster_labels

    # Convert cluster labels to string for better visualization
    df_features['cluster'] = df_features['cluster'].apply(lambda x: f'Cluster {x}' if x >= 0 else 'Noise')

    # Create a custom palette where noise points are gray
    unique_clusters = df_features['cluster'].unique()
    n_clusters = len([c for c in unique_clusters if c != 'Noise'])

    # Create a colormap for the clusters
    cluster_colors = plt.cm.get_cmap('viridis', max(3, n_clusters))
    palette = {}

    # Assign colors to clusters
    cluster_idx = 0
    for cluster in unique_clusters:
        if cluster == 'Noise':
            palette[cluster] = 'gray'
        else:
            palette[cluster] = cluster_colors(cluster_idx)
            cluster_idx += 1

    # Create the corner plot
    print("Creating corner plot of features colored by clusters...")
    plt.figure(figsize=(15, 15))

    # Create the pairplot with cluster coloring
    corner_plot = sns.pairplot(
        df_features,
        hue='cluster',
        palette=palette,
        plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'none'},
        diag_kind='kde',
        corner=True,  # False - full pairplot, True - corner plot
    )

    # Adjust the plot
    corner_plot.fig.suptitle('Feature Relationships by Cluster', fontsize=24, y=1.02)

    # Save the plot
    corner_plot.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved to: {output_file}")

    return output_file

def plot_correlation_matrix(feature_matrix, feature_names, output_file=None):
    """
    Create and visualize a correlation matrix for the extracted features.

    Parameters:
        feature_matrix (np.ndarray): Matrix of feature values (rows=samples, columns=features)
        feature_names (list): List of feature names corresponding to columns in feature_matrix
        output_file (str): Path to save the plot (if None, a default name will be used)

    Returns:
        str: Path to the saved correlation matrix plot
    """
    # Create a DataFrame with the features
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)

    # Calculate the correlation matrix
    corr_matrix = df_features.corr()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_correlation_matrix_{timestamp}.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Correlation matrix saved to: {output_file}")

    return output_file

# Run the HDBSCAN clustering pipeline
print("\n" + "="*50)
print("RUNNING HDBSCAN CLUSTERING")
print("="*50)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
hdbscan_output_file = os.path.join(HDBSCAN_OUTPUT_DIR, f"hdbscan_clusters_{timestamp}.png")

# Run HDBSCAN with default parameters
cluster_labels, feature_matrix, pca_result = run_hdbscan_clustering(
    light_curves,
    min_cluster_size=10,  # Adjust based on dataset size
    output_file=hdbscan_output_file
)

# Create a corner plot of features colored by clusters
feature_names = [
    "weighted_mean", "weighted_variance", "median", "iqr", "beyond_1_sigma",
    "mad", "max_amp", "flux_percentile_ratio"
]
corner_plot_file = plot_feature_corner_plot(
    feature_matrix,
    feature_names,
    cluster_labels,
    os.path.join(HDBSCAN_OUTPUT_DIR, f"feature_corner_plot_{timestamp}.png")
)
print(f"Feature corner plot saved to: {corner_plot_file}")

# Create a correlation matrix for all features
correlation_matrix_file = plot_correlation_matrix(
    feature_matrix,
    feature_names,
    os.path.join(FEATURE_OUTPUT_DIR, f"feature_correlation_matrix_{timestamp}.png")
)
print(f"Feature correlation matrix saved to: {correlation_matrix_file}")

# Run the UMAT clustering pipeline
print("\n" + "="*50)
print("RUNNING UMAP CLUSTERING")
print("="*50)
umat_output_file = os.path.join(UMAT_OUTPUT_DIR, f"umap_clusters_{timestamp}.png")

# Run UMAT with default parameters
# Note: This will only run if umap-learn is installed
try:
    umat_cluster_labels, umat_feature_matrix, umat_embedding = run_umap_clustering_complete(
        light_curves,
        n_neighbors=15,
        min_dist=0.1,
        output_file=umat_output_file,
        eps=0.5,
        min_samples=5
    )
    umat_success = True
except ImportError:
    print("UMAP clustering skipped - umap-learn package not installed")
    print("To install, run: pip install umap-learn")
    umat_success = False

# Generate sample plots for each cluster
features_df = df_process_all_light_curves_error(light_curves)

# Plot HDBSCAN cluster samples
hdbscan_samples_dir = plot_cluster_samples(light_curves, features_df, cluster_labels, HDBSCAN_OUTPUT_DIR, timestamp)
print(f"\nHDBSCAN cluster sample plots saved to: {hdbscan_samples_dir}")

# Plot UMAT cluster samples if UMAT was run successfully
if 'umat_success' in locals() and umat_success:
    umat_samples_dir = plot_cluster_samples(light_curves, features_df, umat_cluster_labels, UMAT_OUTPUT_DIR, timestamp)
    print(f"\nUMAP cluster sample plots saved to: {umat_samples_dir}")

    # Create a corner plot of features colored by UMAP clusters
    umap_corner_plot_file = plot_feature_corner_plot(
        umat_feature_matrix,
        feature_names,
        umat_cluster_labels,
        os.path.join(UMAT_OUTPUT_DIR, f"umap_feature_corner_plot_{timestamp}.png")
    )
    print(f"UMAP feature corner plot saved to: {umap_corner_plot_file}")
