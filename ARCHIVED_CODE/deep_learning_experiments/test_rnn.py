import os
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
from datetime import datetime
import time

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from helper import load_light_curve, load_n_light_curves, load_all_fits_files, partition_data

from RNN_9_model import ELBO, Poisson_NLL, RNN_VAE, Decoder, Encoder, collate_fn_err_mult
import json
import hdbscan

# Function to save hyperparameters to a config file
def save_config(config, save_path):
    """
    Save model configuration to a JSON file

    Args:
        config (dict): Dictionary containing model hyperparameters
        save_path (str): Path to save the config file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {save_path}")

# Function to load hyperparameters from a config file
def load_config(config_path):
    """
    Load model configuration from a JSON file

    Args:
        config_path (str): Path to the config file

    Returns:
        dict: Dictionary containing model hyperparameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Config loaded from {config_path}")
    return config

def plot_sample_reconstructions(model, test_dataset, device, plot_dir, model_str, num_samples=25):
    """
    Plot sample reconstructions from the model

    Args:
        model (RNN_VAE): The trained model
        test_dataset (list): The test dataset
        device (torch.device): Device to run the model on
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames
        num_samples (int): Number of samples to plot
    """
    with torch.no_grad():
        fig, axes = plt.subplots(5, 5, figsize=(15, 20))
        axes = axes.flatten()

        # Determine the maximum sequence length in the validation dataset
        max_length = max(len(sample[1]['RATE']) for sample in test_dataset)

        for i in range(num_samples):
            idx = random.randint(0, len(test_dataset) - 1)
            sample = test_dataset[idx]
            x, lengths = collate_fn_err_mult([sample])
            x = x.to(device)
            lengths = lengths.cpu().to(torch.int64)

            x_hat, _, _ = model(x, lengths)
            x_hat_rate = x_hat[..., 0].cpu().numpy()

            axes[i].plot(x.cpu().numpy()[0, :, 0], label='Original RATE')
            axes[i].plot(x_hat_rate[0], label='Reconstructed RATE')
            axes[i].legend()
            axes[i].set_title(f'Light Curve {i+1}')
            axes[i].set_xlim(0, max_length)  # Set uniform x-axis limits

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Plot_Reconstruction_{model_str}.png"))
        plt.close()


def compute_reconstruction_losses(model, test_loader, device):
    """
    Compute reconstruction losses for each sample in the test dataset

    Args:
        model (RNN_VAE): The trained model
        test_loader (DataLoader): DataLoader for the test dataset
        device (torch.device): Device to run the model on

    Returns:
        list: List of tuples (file_name, loss)
    """
    file_losses = []  # List to store (file_name, loss) pairs

    with torch.no_grad():
        for batch_idx, (x, lengths) in enumerate(test_loader):
            x = x.to(device)
            x_hat, mu, logvar = model(x, lengths)
            x_hat_rate = x_hat[..., 0]

            # Compute reconstruction loss for each sample in the batch
            for i in range(len(lengths)):
                # Calculate normalized loss using relative error
                original = x[i, :lengths[i], 0]
                predicted = x_hat_rate[i, :lengths[i]]
                # Add small epsilon to avoid division by zero
                epsilon = 1e-8
                # Calculate relative error: |y_true - y_pred| / (|y_true| + epsilon)
                relative_error = torch.abs(original - predicted) / (torch.abs(original) + epsilon)
                sample_loss = torch.mean(relative_error).item()
                # Get the actual index in the dataset
                dataset_idx = batch_idx * test_loader.batch_size + i
                if dataset_idx >= len(test_loader.dataset):  # Skip if we're past the end of the dataset
                    continue
                file_losses.append((f"Sample_{dataset_idx}", sample_loss))

    return file_losses


def plot_top_losses(file_losses, plot_dir, model_str, n=100):
    """
    Plot the top N samples with highest reconstruction losses

    Args:
        file_losses (list): List of tuples (file_name, loss)
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames
        n (int): Number of top losses to plot
    """
    # Sort losses in descending order and get top n
    sorted_losses = sorted(file_losses, key=lambda x: x[1], reverse=True)
    top_n_losses = sorted_losses[:n]

    # Create the plot for top n losses
    plt.figure(figsize=(15, 8))
    file_names = [f[0] for f in top_n_losses]
    losses = [f[1] for f in top_n_losses]

    plt.scatter(range(len(top_n_losses)), losses, alpha=0.6)
    plt.xticks(range(len(top_n_losses)), file_names, rotation=45, ha='right')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Loss')
    plt.title(f'Top {n} Reconstruction Losses per Light Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"top_{n}_reconstruction_losses_{model_str}.png"))
    plt.close()

    # Save losses to text file
    loss_file_path = os.path.join(plot_dir, f"reconstruction_losses_{model_str}.txt")
    with open(loss_file_path, 'w') as f:
        f.write("File Name\tLoss\n")
        for file_name, loss in sorted_losses:
            f.write(f"{file_name}\t{loss:.6f}\n")


def plot_top_25_losses(model, test_loader, file_losses, plot_dir, model_str):
    """
    Plot the light curves and their reconstructions for the 25 samples with highest losses.

    Args:
        model (RNN_VAE): The trained VAE model
        test_loader (DataLoader): DataLoader containing the test data
        file_losses (list): List of tuples containing (file_name, loss) pairs
        plot_dir (str): Directory to save the plots
        model_str (str): Model name for plot filenames
    """
    # Sort losses in descending order and get top 25
    sorted_losses = sorted(file_losses, key=lambda x: x[1], reverse=True)[:25]
    top_25_indices = [int(fname.split('_')[1]) for fname, _ in sorted_losses]

    # Create a figure with 5x5 subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.ravel()

    with torch.no_grad():
        for idx, (sample_idx, ax) in enumerate(zip(top_25_indices, axes)):
            # Calculate which batch and position within batch
            batch_idx = sample_idx // test_loader.batch_size
            pos_in_batch = sample_idx % test_loader.batch_size

            # Get the specific batch
            for i, (x, lengths) in enumerate(test_loader):
                if i == batch_idx:
                    x = x.to(model.device)
                    x_hat, mu, logvar = model(x, lengths)

                    # Get original and reconstructed data for the specific sample
                    original = x[pos_in_batch, :lengths[pos_in_batch], 0].cpu().numpy()
                    reconstructed = x_hat[pos_in_batch, :lengths[pos_in_batch], 0].cpu().numpy()

                    # Plot
                    ax.plot(original, label='Original', alpha=0.7)
                    ax.plot(reconstructed, label='Reconstructed', alpha=0.7)
                    ax.set_title(f'Sample {sample_idx}\nLoss: {sorted_losses[idx][1]:.4f}')
                    ax.legend()
                    break

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Top_25_Highest_Losses_{model_str}.png"))
    plt.close()


def extract_latent_representations(model, test_loader, device):
    """
    Extract latent representations from the model

    Args:
        model (RNN_VAE): The trained model
        test_loader (DataLoader): DataLoader for the test dataset
        device (torch.device): Device to run the model on

    Returns:
        numpy.ndarray: Array of latent vectors
    """
    print("Extracting latent representations...")
    latent_vectors = []
    with torch.no_grad():
        for x, lengths in test_loader:
            x = x.to(device)
            _, mu, _ = model(x, lengths)
            latent_vectors.append(mu.cpu().numpy())

    return np.vstack(latent_vectors)


def apply_dimensionality_reduction(latent_vectors, plot_dir, model_str):
    """
    Apply dimensionality reduction techniques (PCA, t-SNE) to latent vectors

    Args:
        latent_vectors (numpy.ndarray): Array of latent vectors
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames

    Returns:
        tuple: (pca, latent_pca, latent_2d_pca)
    """
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=6)
    latent_pca = pca.fit_transform(latent_vectors)
    latent_2d_pca = latent_pca[:, :2]  # Keep only the first two components

    # Create PCA plot
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], alpha=0.5)
    plt.title('PCA Visualization of Latent Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(os.path.join(plot_dir, f"latent_space_pca_{model_str}.png"))
    plt.close()

    # Print explained variance ratio for PCA
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    return pca, latent_pca, latent_2d_pca


def apply_clustering(latent_pca, latent_2d_pca, plot_dir, model_str):
    """
    Apply clustering to the latent representations

    Args:
        latent_pca (numpy.ndarray): PCA-transformed latent vectors
        latent_2d_pca (numpy.ndarray): 2D PCA-transformed latent vectors
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames

    Returns:
        numpy.ndarray: Cluster labels
    """
    # Apply HDBSCAN clustering
    print("Applying HDBSCAN clustering...")

    # Cluster using PCA representation
    print("Clustering PCA representation...")
    hdbscan_pca = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
    cluster_labels_pca = hdbscan_pca.fit_predict(latent_pca)

    # Plot clustered PCA results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1],
                         c=cluster_labels_pca, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title('HDBSCAN Clustering on PCA Representation')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(os.path.join(plot_dir, f"hdbscan_pca_{model_str}.png"))
    plt.close()

    # Print clustering statistics
    print("\nClustering Statistics:")
    print("PCA-based clustering:")
    print(f"Number of clusters: {len(set(cluster_labels_pca)) - (1 if -1 in cluster_labels_pca else 0)}")
    print(f"Number of noise points: {sum(1 for label in cluster_labels_pca if label == -1)}")

    return cluster_labels_pca


def plot_cluster_representatives(model, test_loader, cluster_labels, analysis_type, plot_dir):
    """
    Plot representative light curves from each cluster.

    Args:
        model (RNN_VAE): The trained VAE model
        test_loader (DataLoader): DataLoader containing the test data
        cluster_labels (numpy.ndarray): Array of cluster labels
        analysis_type (str): String indicating the type of analysis ('PCA' or 't-SNE')
        plot_dir (str): Base directory for saving plots
    """
    # Create directory for this analysis type
    cluster_plot_dir = os.path.join(plot_dir, f"{analysis_type}_cluster_representatives")
    os.makedirs(cluster_plot_dir, exist_ok=True)

    # Get unique cluster labels (excluding noise points labeled as -1)
    unique_clusters = sorted(list(set(cluster_labels)))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    # Store all data for easier access
    all_data = []
    with torch.no_grad():
        for x, lengths in test_loader:
            all_data.extend([(x[i], lengths[i]) for i in range(len(lengths))])

    # Plot representatives for each cluster
    for cluster in unique_clusters:
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]

        # Select random samples from this cluster
        sample_indices = np.random.choice(cluster_indices, min(10, len(cluster_indices)), replace=False)

        # Create plot
        fig, axes = plt.subplots(1, round(len(sample_indices)), figsize=(20, 4))
        if len(sample_indices) == 1:
            axes = [axes]

        for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
            # Get the data for this sample
            x, length = all_data[sample_idx]
            x = x.unsqueeze(0).to(model.device)  # Add batch dimension
            length = torch.tensor([length])

            # Get reconstruction
            x_hat, _, _ = model(x, length)

            # Plot original and reconstructed
            original = x[0, :length, 0].cpu().numpy()
            reconstructed = x_hat[0, :length, 0].cpu().detach().numpy()

            ax.plot(original, label='Original', alpha=0.7)
            ax.plot(reconstructed, label='Reconstructed', alpha=0.7)
            ax.set_title(f'Sample {idx+1}')
            ax.legend()

        plt.suptitle(f'Cluster {cluster} Representatives')
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_plot_dir, f"cluster_{cluster}_representatives.png"))
        plt.close()


def detect_outliers(latent_vectors, latent_2d_pca, plot_dir, model_str):
    """
    Detect outliers in the latent space using Isolation Forest

    Args:
        latent_vectors (numpy.ndarray): Array of latent vectors
        latent_2d_pca (numpy.ndarray): 2D PCA-transformed latent vectors
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames

    Returns:
        numpy.ndarray: Boolean mask (True for inliers, False for outliers)
    """
    # Apply Isolation Forest for outlier detection
    print("\nApplying Isolation Forest for outlier detection...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = isolation_forest.fit_predict(latent_vectors)

    # Convert predictions to boolean mask (True for inliers, False for outliers)
    is_inlier = outlier_labels == 1

    # Plot PCA visualization with outliers highlighted
    plt.figure(figsize=(12, 8))
    plt.scatter(latent_2d_pca[is_inlier, 0], latent_2d_pca[is_inlier, 1],
                c='blue', alpha=0.5, label='Normal')
    plt.scatter(latent_2d_pca[~is_inlier, 0], latent_2d_pca[~is_inlier, 1],
                c='red', alpha=0.7, label='Outlier')
    plt.title('PCA Visualization with Isolation Forest Outliers')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"isolation_forest_pca_{model_str}.png"))
    plt.close()

    # Print outlier detection statistics
    print("\nIsolation Forest Statistics:")
    print(f"Number of normal samples: {sum(is_inlier)}")
    print(f"Number of outliers: {sum(~is_inlier)}")
    print(f"Outlier percentage: {(sum(~is_inlier) / len(is_inlier)) * 100:.2f}%")

    return is_inlier


def plot_outlier_representatives(model, test_loader, is_inlier, plot_dir, model_str):
    """
    Plot representative outlier light curves

    Args:
        model (RNN_VAE): The trained model
        test_loader (DataLoader): DataLoader for the test dataset
        is_inlier (numpy.ndarray): Boolean mask (True for inliers, False for outliers)
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames
    """
    print("\nPlotting representative outlier light curves...")

    # Get indices of outlier samples
    outlier_indices = np.where(~is_inlier)[0]

    # Select up to 15 random outlier samples
    num_samples = min(15, len(outlier_indices))
    sample_indices = np.random.choice(outlier_indices, num_samples, replace=False)

    # Create plot
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()

    # Store all data for easier access
    all_data = []
    with torch.no_grad():
        for x, lengths in test_loader:
            all_data.extend([(x[i], lengths[i]) for i in range(len(lengths))])

    for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
        # Get the data for this sample
        x, length = all_data[sample_idx]
        x = x.unsqueeze(0).to(model.device)  # Add batch dimension
        length = torch.tensor([length])

        # Get reconstruction
        x_hat, _, _ = model(x, length)

        # Plot original and reconstructed
        original = x[0, :length, 0].cpu().numpy()
        reconstructed = x_hat[0, :length, 0].cpu().detach().numpy()

        ax.plot(original, label='Original', alpha=0.7)
        ax.plot(reconstructed, label='Reconstructed', alpha=0.7)
        ax.set_title(f'Outlier {idx+1}')
        ax.legend()

    # Hide empty subplots if any
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Representative Outlier Light Curves')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"outlier_representatives_{model_str}.png"))
    plt.close()


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

    plt.title('Latent Space Correlation Matrix', fontsize=16)
    plt.tight_layout()

    # Save the plot
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(output_file), f"latent_correlation_matrix_{timestamp}.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Correlation matrix saved to: {output_file}")
    plt.close()

    return output_file


def plot_feature_corner_plot(feature_matrix, feature_names, cluster_labels, output_file=None):
    """
    Create a corner plot of features color-coded by clusters using seaborn's pairplot.

    Parameters:
        feature_matrix (np.ndarray): Matrix of features where each row is a light curve and each column is a feature
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        cluster_labels (np.ndarray): Cluster labels from clustering algorithm
        output_file (str): Path to save the plot (if None, a default name will be used)

    Returns:
        str: Path to the saved corner plot
    """
    # Create a DataFrame with the features and cluster labels
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_features['cluster'] = cluster_labels

    # Count the number of points in each cluster
    cluster_counts = df_features['cluster'].value_counts()

    # Create a palette with distinct colors for each cluster
    # Use a different color for noise points (cluster label -1)
    unique_clusters = sorted(df_features['cluster'].unique())
    palette = {}

    # Use grey for noise points (cluster -1)
    if -1 in unique_clusters:
        palette[-1] = 'grey'
        unique_clusters.remove(-1)

    # Use a colormap for the actual clusters
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10', max(10, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        palette[cluster] = cmap(i)

    # Create the pairplot with cluster coloring
    corner_plot = sns.pairplot(
        df_features,
        hue='cluster',
        palette=palette,
        plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'none'},
        diag_kind='kde',
        corner=True,  # True for corner plot, False for full pairplot
    )

    # Adjust the plot
    corner_plot.fig.suptitle('Latent Space Relationships by Cluster', fontsize=24, y=1.02)

    # Save the plot
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(output_file), f"latent_corner_plot_{timestamp}.png")

    corner_plot.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved to: {output_file}")
    plt.close()

    return output_file


def plot_latent_feature_analysis(latent_vectors, cluster_labels, plot_dir, model_str):
    """
    Generate comprehensive visualization of latent space features.

    Args:
        latent_vectors (numpy.ndarray): Array of latent vectors
        cluster_labels (numpy.ndarray): Cluster labels from clustering
        plot_dir (str): Directory to save plots
        model_str (str): Model name for plot filenames

    Returns:
        tuple: Paths to the saved visualization files
    """
    print("\nGenerating latent space feature analysis visualizations...")

    # Create feature names for the latent dimensions
    latent_size = latent_vectors.shape[1]
    feature_names = [f"latent_dim_{i+1}" for i in range(latent_size)]

    # 1. Create correlation matrix for latent dimensions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corr_matrix_file = plot_correlation_matrix(
        latent_vectors,
        feature_names,
        os.path.join(plot_dir, f"latent_correlation_matrix_{model_str}_{timestamp}.png")
    )

    # 2. Create corner plot of latent dimensions colored by clusters
    corner_plot_file = plot_feature_corner_plot(
        latent_vectors,
        feature_names,
        cluster_labels,
        os.path.join(plot_dir, f"latent_corner_plot_{model_str}_{timestamp}.png")
    )

    return corr_matrix_file, corner_plot_file


def plot_light_curve_simple(ax, times, rates, errors=None, title=None, color='blue'):
    """Plot a single light curve on the given axis"""
    if errors is not None:
        ax.errorbar(times, rates,
                    yerr=errors,
                    fmt='o', markersize=2,
                    elinewidth=0.5, capsize=0,
                    color=color)
    else:
        ax.plot(times, rates, 'o', markersize=2, color=color)

    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('Time', fontsize=7)
    ax.set_ylabel('Rate', fontsize=7)


def plot_cluster_samples_rnn(model, test_loader, cluster_labels, plot_dir, model_str):
    """
    Create grid plots for each cluster detected in the latent space.
    For each cluster, randomly sample up to 10 light curves and display them in a 2x5 grid.

    Parameters:
        model (RNN_VAE): The trained VAE model
        test_loader (DataLoader): DataLoader containing the test data
        cluster_labels (np.array): Cluster labels from clustering algorithm
        plot_dir (str): Directory to save the plots
        model_str (str): Model name for plot filenames
    """
    print("\nCreating cluster sample plots...")
    cluster_start_time = time.time()

    # Create a timestamp for the filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a subdirectory for the cluster plots
    cluster_plots_dir = os.path.join(plot_dir, f"cluster_samples_{timestamp}")
    os.makedirs(cluster_plots_dir, exist_ok=True)
    print(f"Created directory for cluster samples: {cluster_plots_dir}")

    # Get unique cluster labels (excluding noise which is -1)
    unique_clusters = sorted(list(set(cluster_labels)))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    # Also include noise points
    all_cluster_labels = sorted(list(set(cluster_labels)))

    # Define a colormap for consistency with the main plot
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap('viridis', max(3, n_clusters))

    # Collect all data from the test loader
    all_data = []
    all_lengths = []
    with torch.no_grad():
        for batch_idx, (x, lengths) in enumerate(test_loader):
            all_data.append(x)
            all_lengths.append(lengths)

    # Concatenate all batches
    if len(all_data) > 0:
        all_data = torch.cat(all_data, dim=0)
        all_lengths = torch.cat(all_lengths, dim=0)
    else:
        print("No data found in test_loader")
        return None

    # Process each cluster (including noise)
    for cluster_label in all_cluster_labels:
        # Get indices of light curves in this cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]

        # Skip if no light curves in this cluster
        if len(cluster_indices) == 0:
            continue

        # Determine cluster name and color
        if cluster_label == -1:
            cluster_name = "Noise"
            color = 'gray'
        else:
            cluster_name = f"Cluster_{cluster_label}"
            color = cmap(cluster_label % n_clusters)

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
                # Get the light curve
                lc_idx = sampled_indices[i]

                # Skip if index is out of range
                if lc_idx >= len(all_data):
                    ax.axis('off')
                    continue

                x = all_data[lc_idx:lc_idx+1]
                length = all_lengths[lc_idx:lc_idx+1]

                # Extract data for plotting
                device = next(model.parameters()).device
                x = x.to(device)
                original = x[0, :length[0], 0].cpu().numpy()
                times = np.arange(len(original))

                # Plot original only
                ax.plot(times, original, 'o', markersize=2, color=color, alpha=0.7, label='Original')

                # Set title and labels
                ax.set_title(f"Sample {i+1}", fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=6)

                if ax_idx >= 5:  # Only add x-label to bottom row
                    ax.set_xlabel('Time', fontsize=7)
                if ax_idx % 5 == 0:  # Only add y-label to leftmost column
                    ax.set_ylabel('Rate', fontsize=7)

                # Add legend to the first plot only
                if i == 0:
                    ax.legend(fontsize=6, loc='upper right')
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

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        cluster_plot_file = os.path.join(
            cluster_plots_dir,
            f"{model_str}_cluster_{cluster_label}_samples.png"
        )
        plt.savefig(cluster_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    cluster_end_time = time.time()
    print(f"Cluster sample plots created in {cluster_end_time - cluster_start_time:.2f} seconds")
    return cluster_plots_dir


def main():
    """Main function to run the testing pipeline"""
    # Define a dictionary with all hyperparameters
    print('Start Running File')

    model_str = "30000_files_RNN"

    print(f'model_str = {model_str}')

    old_config_dir = "/home/pdong/Astro UROP/training_plots/RNN Models/" + model_str
    old_config_path = os.path.join(old_config_dir, "config.json")
    hyperparams = load_config(old_config_path)

    model_dir = "/home/pdong/Astro UROP/models/RNN Models/" + model_str + ".h5"

    model_str = hyperparams["model_name"]
    learning_rate = hyperparams["learning_rate"]
    data_size = hyperparams["data_size"]
    num_epochs = hyperparams["num_epochs"]
    latent_size = hyperparams["latent_size"]
    KLD_coef = hyperparams["KLD_coef"]
    hidden_size = hyperparams["hidden_size"]
    input_size = hyperparams["input_size"]
    output_size = hyperparams["output_size"]
    batch_size = hyperparams["batch_size"]


    test_data_size = 190000

    plot_dir = "/home/pdong/Astro UROP/plots/RNN plots/" + model_str + "/" + str(test_data_size)
    os.makedirs(plot_dir, exist_ok=True)

    # Save hyperparameters to config file in the plotting folder
    new_config_path = os.path.join(plot_dir, "config.json")
    save_config(hyperparams, new_config_path)

    # set the device we're using for training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Torch Diagnostics')
    print(f'Available: {torch.cuda.is_available()}')
    print(f'Version: {torch.__version__}, {torch.version.cuda}')

    print('Start Loading Files ...')
    fits_files = load_all_fits_files()
    print('Loaded Files Complete')

    print(f'Start Loading {test_data_size} Light Curves ...')
    lc_low, lc_med, lc_high = load_n_light_curves(test_data_size, fits_files, band = "all")
    light_curves_sample = list(zip(lc_low, lc_med, lc_high))
    print(f'Loading Light Curves Complete. Length: {len(light_curves_sample)}')

    test_dataset = light_curves_sample
    test_loader = DataLoader(light_curves_sample, batch_size=batch_size, collate_fn=collate_fn_err_mult)

    print('Finished Loading LCs')


    model = RNN_VAE(
        input_size=hyperparams["input_size"],
        hidden_size=hyperparams["hidden_size"],
        latent_size=hyperparams["latent_size"],
        output_size=hyperparams["output_size"],
        device=device
    ).to(device)

    model.load_state_dict(torch.load(model_dir))

    print("Starting to use model")

    # Plot sample reconstructions
    plot_sample_reconstructions(model, test_dataset, device, plot_dir, model_str)

    # Compute reconstruction losses
    file_losses = compute_reconstruction_losses(model, test_loader, device)

    # Plot top losses
    plot_top_losses(file_losses, plot_dir, model_str)

    # Plot top 25 losses with reconstructions
    plot_top_25_losses(model, test_loader, file_losses, plot_dir, model_str)

    # Extract latent representations
    latent_vectors = extract_latent_representations(model, test_loader, device)

    # Apply dimensionality reduction
    pca, latent_pca, latent_2d_pca = apply_dimensionality_reduction(latent_vectors, plot_dir, model_str)

    # Apply clustering
    cluster_labels_pca = apply_clustering(latent_pca, latent_2d_pca, plot_dir, model_str)

    # Generate additional latent space visualizations
    # corr_matrix_file, corner_plot_file = plot_latent_feature_analysis(
    #     latent_pca, cluster_labels_pca, plot_dir, model_str
    # )
    # print(f"Latent space correlation matrix saved to: {corr_matrix_file}")
    # print(f"Latent space corner plot saved to: {corner_plot_file}")

    # Plot cluster representatives
    print("\nPlotting PCA cluster representatives...")
    plot_cluster_representatives(model, test_loader, cluster_labels_pca, 'PCA', plot_dir)

    # Detect outliers
    is_outlier = detect_outliers(latent_vectors, latent_2d_pca, plot_dir, model_str)

    # Plot outlier representatives
    plot_outlier_representatives(model, test_loader, is_outlier, plot_dir, model_str)

    # Plot samples from each cluster with original and reconstructed curves
    cluster_samples_dir = plot_cluster_samples_rnn(model, test_loader, cluster_labels_pca, plot_dir, model_str)
    print(f"Cluster sample plots saved to: {cluster_samples_dir}")

    print("Done with all visualizations")


if __name__ == "__main__":
    main()
