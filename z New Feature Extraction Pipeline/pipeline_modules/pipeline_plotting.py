import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler, normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os
import time
from datetime import datetime

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (
    FEATURE_OUTPUT_DIR, UMAP_OUTPUT_DIR, KNOWN_LIGHT_CURVES,
    DEFAULT_N_NEIGHBORS, DEFAULT_MIN_DIST
)
from pipeline_modules.pipeline_io import build_index_maps, resolve_unique_index
from pipeline_modules.pipeline_clustering import (
    _discrete_cmap_for_labels, DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_EPSILON
)

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
    # known_indices = [i for i, path in enumerate(file_paths)
    #                 if any(k in path for k in KNOWN_LIGHT_CURVES)]

    basenames = np.array([os.path.basename(p) for p in file_paths])
    known_set = set(KNOWN_LIGHT_CURVES)
    known_indices = [i for i, b in enumerate(basenames) if b in known_set]

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
        sample_size = min(25, len(cluster_indices))
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

def plot_noise_cluster_samples(light_curves, labels, output_dir, n_sample=100):
    """Sample up to n_sample noise-cluster (-1) light curves and save them as grid PNGs (25 per file)."""
    noise_idxs = np.where(labels == -1)[0]
    n_sample   = min(n_sample, len(noise_idxs))
    if n_sample == 0:
        print("No noise points to plot.")
        return

    sampled = np.random.choice(noise_idxs, n_sample, replace=False)
    os.makedirs(output_dir, exist_ok=True)
    for file_ix in range(4):
        start = file_ix * 25
        end   = min(start + 25, n_sample)
        sel   = sampled[start:end]
        if len(sel) == 0:
            break

        fig, axes = plt.subplots(5, 5, figsize=(20, 20))
        axes = axes.ravel()
        for i, idx in enumerate(sel):
            plot_light_curve(axes[i], light_curves[idx], title=f"Noise #{i+1}", is_outlier=True)
        for ax in axes[len(sel):]:
            ax.axis("off")
        plt.tight_layout()

        out_fn = os.path.join(output_dir, f"noise_cluster_sample_{file_ix+1}.png")
        fig.savefig(out_fn, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote noise‐sample plot: {out_fn}")


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
    file_paths, basenames, name_to_idxs, fullpath_to_idx = build_index_maps(features_df)

    # Track which cluster each known curve is in
    for sig_curve in significant_curves:
        try:
            # sig_idx = np.where([sig_curve in path for path in file_paths])[0][0]
            sig_idx = resolve_unique_index(sig_curve, name_to_idxs, fullpath_to_idx)
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
    file_paths, basenames, name_to_idxs, fullpath_to_idx = build_index_maps(features_df)
    # Second pass: create plots for each significant curve
    for sig_curve in significant_curves:
        # Find the index of the significant curve
        try:
            # sig_idx = np.where([sig_curve in path for path in file_paths])[0][0]
            sig_idx = resolve_unique_index(sig_curve, name_to_idxs, fullpath_to_idx)
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
    file_paths, basenames, name_to_idxs, fullpath_to_idx = build_index_maps(features_df)

    # Calculate similarity for each known light curve
    for known_lc in known_light_curves:
        try:
            # Find the known light curve in our dataset
            # known_idx = np.where([os.path.basename(fp) == known_lc for fp in file_paths])[0][0]
            known_idx = resolve_unique_index(known_lc, name_to_idxs, fullpath_to_idx)
        except IndexError:
            print(f"Warning: Known light curve {known_lc} not found in dataset")
            continue

        full_path_known = file_paths[known_idx]
        print(f"Creating similarity plot for {os.path.basename(known_lc)}")
        print(f"  → full path: {full_path_known}")

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

        axes[0].set_title(f"[REF]\n{os.path.basename(file_paths[known_idx])}", fontsize=7)

        # Plot similar curves
        for i, (idx, sim_score) in enumerate(zip(top_indices, top_similarities)):
            if i + 1 < len(axes):  # +1 because we used the first plot for reference
                plot_light_curve(axes[i+1], light_curves[idx],
                               title=f'Similarity: {sim_score:.3f}')
                axes[i+1].set_title(
                    f"Sim: {sim_score:.3f}\n{os.path.basename(file_paths[idx])}",
                    fontsize=7
                )

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

def histogram_similar_curve_cluster_hits(
    features_df,
    cluster_labels,
    known_light_curves,
    output_dir,
    n_similar=200
):
    """
    For each known curve:
      - find the top-n_similar most similar curves (cosine in feature space)
      - plot a histogram of their cluster IDs
      - save a CSV with the counts

    Saves plots to <output_dir>/similar_hits and a combined CSV per known curve.
    """
    os.makedirs(output_dir, exist_ok=True)
    hits_dir = os.path.join(output_dir, "similar_hits")
    os.makedirs(hits_dir, exist_ok=True)

    feature_matrix = np.vstack(features_df['feature_values'].values)
    file_paths, basenames, name_to_idxs, fullpath_to_idx = build_index_maps(features_df)

    sims = cosine_similarity(feature_matrix, feature_matrix)  # (N, N)

    # Discrete palette consistent with clustering
    label_to_idx, idx_to_label, cmap, norm, ticks, ticklabels = _discrete_cmap_for_labels(cluster_labels, base_cmap='tab20', noise_label=-1)

    for known in known_light_curves:
        try:
            kidx = resolve_unique_index(known, name_to_idxs, fullpath_to_idx)
            if not kidx:
                print(f"Warning: known curve {known} not found; skipping histogram.")
                continue
        except IndexError:
            print(f"Warning: known curve {known} not found; skipping histogram.")
            continue

        # top-K excluding self
        row = sims[kidx].copy()
        row[kidx] = -np.inf
        top = np.argsort(row)[::-1][:n_similar]
        top_clusters = cluster_labels[top]

        # Save ranked CSV of top-N similar curves
        top_paths = file_paths[top]
        top_names = [os.path.basename(p) for p in top_paths]
        top_sims  = row[top]   # cosine similarity values (row already has self excluded)

        similar_df = pd.DataFrame({
            'rank':             range(1, len(top) + 1),
            'file_name':        top_names,
            'file_path':        top_paths,
            'cosine_similarity': top_sims,
            'cluster_label':    top_clusters,
        })
        csv_ranked = os.path.join(hits_dir, f"{os.path.splitext(known)[0]}_top{n_similar}_similar.csv")
        similar_df.to_csv(csv_ranked, index=False)
        print(f"→ wrote {csv_ranked}")

        # count
        counts = pd.Series(top_clusters).value_counts().sort_index()
        # write CSV
        csv_path = os.path.join(hits_dir, f"{os.path.splitext(known)[0]}_similar_cluster_counts.csv")
        counts.to_csv(csv_path, header=['count'])
        print(f"→ wrote {csv_path}")

        # plot histogram (bars colored by the **cluster palette**)
        labs = counts.index.tolist()
        idxs = [label_to_idx[int(l)] for l in labs]
        colors = [cmap(norm(i)) for i in idxs]

        plt.figure(figsize=(10, 4))
        plt.bar([str(l) for l in labs], counts.values, color=colors)
        plt.yscale('log')
        plt.xlabel("Cluster ID")
        plt.ylabel(f"Top-{n_similar} similar (count, log)")
        title_suffix = f"(known curve cluster: {cluster_labels[kidx]})"
        plt.title(f"Where most-similar to {known} land {title_suffix}")
        plt.tight_layout()

        out_png = os.path.join(hits_dir, f"{os.path.splitext(known)[0]}_similar_cluster_hist.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"→ wrote {out_png}")

def plot_umap_colored_by_feature(
    features_df,
    umap_embedding=None,                 # (n_samples, 2) array; if None we compute it from features_df
    feature_list="all",                  # "all" or list of feature names
    output_dir=UMAP_OUTPUT_DIR,
    n_neighbors=DEFAULT_N_NEIGHBORS,
    min_dist=DEFAULT_MIN_DIST,
    highlight_known=True,
    robust_color_limits=True,            # use 1–99th percentile for colorbar limits (helps with outliers)
    log_color=False                      # log-scale colorbar for highly skewed features
):
    """
    Create UMAP scatter plots colored by individual feature values.

    Saves one PNG per feature in: <output_dir>/umap_color_by/<feature>.png

    Args:
        features_df (pd.DataFrame): Must contain 'feature_values', 'feature_names', 'file_path'
        umap_embedding (np.ndarray|None): If None, UMAP will be computed from filtered features_df
        feature_list (str|list): "all" for every feature in features_df, or an explicit list of names
        highlight_known (bool): Draw KNOWN_LIGHT_CURVES as star markers
        robust_color_limits (bool): Clip colormap to [1st, 99th] percentiles to reduce saturation
        log_color (bool): Apply log scaling to color values (small epsilon added to avoid log(0))
    """
    os.makedirs(output_dir, exist_ok=True)
    outdir = os.path.join(output_dir, "umap_color_by")
    os.makedirs(outdir, exist_ok=True)

    # --- Prepare feature matrix (post-filter) ---
    # Assumes filter_features(...) already made every row share same feature_names ordering.
    feature_names = features_df['feature_names'].iloc[0]
    feature_matrix = np.vstack(features_df['feature_values'].values)
    file_paths = features_df['file_path'].values

    # Determine which features to plot
    if feature_list == "all":
        names_to_plot = list(feature_names)
    else:
        # keep only valid ones
        names_to_plot = [f for f in feature_list if f in feature_names]
        missing = set(feature_list) - set(names_to_plot)
        if missing:
            print(f"Warning: features not found and skipped: {sorted(missing)}")

    # --- UMAP embedding (if not supplied) ---
    if umap_embedding is None:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(feature_matrix)
        scaled = np.nan_to_num(scaled, nan=0.0)
        normed = normalize(scaled, norm='l2')

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        umap_embedding = reducer.fit_transform(normed)

    emb = umap_embedding
    assert emb.shape[1] == 2, "umap_embedding must be (n_samples, 2)"

    # Known curve indices
    known_idxs = []
    if highlight_known:
        basenames = [os.path.basename(p) for p in file_paths]
        known_set = set(KNOWN_LIGHT_CURVES)
        known_idxs = [i for i, b in enumerate(basenames) if b in known_set]

    # Helper to extract a feature column by name
    name_to_col = {name: j for j, name in enumerate(feature_names)}

    for fname in names_to_plot:
        col = name_to_col[fname]
        vals = feature_matrix[:, col].astype(float)
        vals = np.nan_to_num(vals, nan=np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0)

        # Optional log color
        if log_color:
            eps = max(1e-12, np.nanmin(vals[vals > 0.0]) if np.any(vals > 0.0) else 1e-12)
            color_vals = np.log(vals + eps)
            cbar_label = f"log({fname})"
        else:
            color_vals = vals
            cbar_label = fname

        # Robust color limits to avoid a few outliers blowing out the colormap
        vmin = vmax = None
        if robust_color_limits:
            lo = np.percentile(color_vals, 1)
            hi = np.percentile(color_vals, 99)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                vmin, vmax = lo, hi

        plt.figure(figsize=(12, 8))

        # Main scatter
        sc = plt.scatter(
            emb[:, 0], emb[:, 1],
            c=color_vals,
            s=35, alpha=0.85,
            cmap='viridis',
            edgecolors='white',
            linewidths=0.3,
            vmin=vmin, vmax=vmax,
            label='Light curves'
        )

        # Known curves as stars on top
        if highlight_known and known_idxs:
            plt.scatter(
                emb[known_idxs, 0], emb[known_idxs, 1],
                c=color_vals[known_idxs],
                cmap='viridis',
                marker='*',
                s=220,
                edgecolors='black',
                linewidths=1.0,
                vmin=vmin, vmax=vmax,
                label='Known curves'
            )

        cbar = plt.colorbar(sc)
        cbar.set_label(cbar_label)

        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"UMAP colored by {fname}")
        plt.grid(alpha=0.3)

        # Avoid duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"umap_color_by_{fname}.png")
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {outfile}")

    return outdir
