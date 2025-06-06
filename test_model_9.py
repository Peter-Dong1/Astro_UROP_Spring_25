import os
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import random
import math
import json

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from helper import load_light_curve, load_n_light_curves, load_all_fits_files, partition_data

# Import the model from the RNN_9_model.py file
from RNN_9_model import ELBO, Poisson_NLL, RNN_VAE, Decoder, Encoder, collate_fn_err_mult

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

# Define a dictionary with all hyperparameters
hyperparams = {
    "model_name": "RNN_Large_3bands",
    "learning_rate": 1e-5,
    "data_size": 16384,
    "num_epochs": 3000,
    "latent_size": 22,
    "KLD_coef": 0.0035,
    "hidden_size": 512,
    "input_size": 9,
    "output_size": 1,
    "batch_size": 32
}

# Save hyperparameters to config file
config_path = "./config.json"
save_config(hyperparams, config_path)

# set the device we're using for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())

print('torch diagonostics')
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)
print(torch.cuda.is_available())

print('start to load light curves')

fits_files = load_all_fits_files()

# light_curves_sample = load_n_light_curves(16384, fits_files, band = "med")
lc_low, lc_med, lc_high = load_n_light_curves(hyperparams["data_size"], fits_files, band = "all")
# lc_low, lc_med, lc_high = load_n_light_curves(10, fits_files, band = "all")

light_curves_sample = list(zip(lc_low, lc_med, lc_high))
print('finished loading lcs')
print(len(light_curves_sample))

test_dataset = light_curves_sample
test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], collate_fn=collate_fn_err_mult, shuffle=True)

print('finish loading lcs')

model = RNN_VAE(
    input_size=hyperparams["input_size"],
    hidden_size=hyperparams["hidden_size"],
    latent_size=hyperparams["latent_size"],
    output_size=hyperparams["output_size"],
    device=device
).to(device)

model_str = hyperparams["model_name"]
model.load_state_dict(torch.load('./models/' + model_str + '.h5'))

plot_dir = "/home/pdong/Astro UROP/plots/" + model_str + "_test_large"
# Ensure the directory exists
os.makedirs(plot_dir, exist_ok=True)

# Save hyperparameters to config file in the plotting folder
config_path = os.path.join(plot_dir, "config.json")
save_config(hyperparams, config_path)

model.eval()
with torch.no_grad():
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))
    axes = axes.flatten()

    # Determine the maximum sequence length in the validation dataset
    max_length = max(len(sample[1]['RATE']) for sample in test_dataset)

    for i in range(25):
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
    plt.savefig(os.path.join(plot_dir, f"Plot_Reconstruction {model_str}.png"))
    plt.show()

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
            if dataset_idx >= len(test_dataset):  # Skip if we're past the end of the dataset
                continue
            file_losses.append((f"Sample_{dataset_idx}", sample_loss))

    # Sort losses in descending order and get top 100
    sorted_losses = sorted(file_losses, key=lambda x: x[1], reverse=True)
    top_100_losses = sorted_losses[:100]

    # Create the plot for top 100 losses
    plt.figure(figsize=(15, 8))
    file_names = [f[0] for f in top_100_losses]
    losses = [f[1] for f in top_100_losses]

    plt.scatter(range(len(top_100_losses)), losses, alpha=0.6)
    plt.xticks(range(len(top_100_losses)), file_names, rotation=45, ha='right')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Loss')
    plt.title('Top 100 Reconstruction Losses per Light Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"top_100_reconstruction_losses_{model_str}.png"))
    plt.close()

    # Save losses to text file
    loss_file_path = os.path.join(plot_dir, f"reconstruction_losses_{model_str}.txt")
    with open(loss_file_path, 'w') as f:
        f.write("File Name\tLoss\n")
        for file_name, loss in sorted_losses:
            f.write(f"{file_name}\t{loss:.6f}\n")

def plot_top_25_losses(model, test_loader, file_losses, plot_dir):
    """Plot the light curves and their reconstructions for the 25 samples with highest losses.

    Args:
        model: The trained VAE model
        test_loader: DataLoader containing the test data
        file_losses: List of tuples containing (file_name, loss) pairs
        plot_dir: Directory to save the plots
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
                    x = x.to(device)
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
    plt.savefig(os.path.join(plot_dir, f"Top_25_Highest_Losses {model_str}.png"))
    plt.show()

# Plot the top 25 highest loss light curves
plot_top_25_losses(model, test_loader, file_losses, plot_dir)

# Extract latent representations
print("Extracting latent representations...")
latent_vectors = []
with torch.no_grad():
    for x, lengths in test_loader:
        x = x.to(device)
        _, mu, _ = model(x, lengths)
        latent_vectors.append(mu.cpu().numpy())

latent_vectors = np.vstack(latent_vectors)

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=6)
latent_pca = pca.fit_transform(latent_vectors)
latent_2d_pca = latent_pca[:, :2]  # Keep only the first two components

# Apply t-SNE
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
latent_2d_tsne = tsne.fit_transform(latent_vectors)

# Create PCA plot
plt.figure(figsize=(10, 8))
plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], alpha=0.5)
plt.title('PCA Visualization of Latent Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(os.path.join(plot_dir, f"latent_space_pca {model_str}.png"))
plt.close()

# Create t-SNE plot
plt.figure(figsize=(10, 8))
plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig(os.path.join(plot_dir, f"latent_space_tsne {model_str}.png"))
plt.close()

# Print explained variance ratio for PCA
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

# Apply HDBSCAN clustering
print("Applying HDBSCAN clustering...")
import hdbscan

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


def plot_cluster_representatives(model, test_loader, cluster_labels, analysis_type, plot_dir):
    """Plot representative light curves from each cluster.

    Args:
        model: The trained VAE model
        test_loader: DataLoader containing the test data
        cluster_labels: Array of cluster labels
        analysis_type: String indicating the type of analysis ('PCA' or 't-SNE')
        plot_dir: Base directory for saving plots
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

        # Select 5 random samples from this cluster
        sample_indices = np.random.choice(cluster_indices, min(10, len(cluster_indices)), replace=False)

        # Create plot
        fig, axes = plt.subplots(1, round(len(sample_indices)), figsize=(20, 4))
        if len(sample_indices) == 1:
            axes = [axes]

        for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
            # Get the data for this sample
            x, length = all_data[sample_idx]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
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

# Plot representative light curves for PCA-based clustering
print("\nPlotting PCA cluster representatives...")
plot_cluster_representatives(model, test_loader, cluster_labels_pca, 'PCA', plot_dir)

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

# Plot t-SNE visualization with outliers highlighted
plt.figure(figsize=(12, 8))
plt.scatter(latent_2d_tsne[is_inlier, 0], latent_2d_tsne[is_inlier, 1],
            c='blue', alpha=0.5, label='Normal')
plt.scatter(latent_2d_tsne[~is_inlier, 0], latent_2d_tsne[~is_inlier, 1],
            c='red', alpha=0.7, label='Outlier')
plt.title('t-SNE Visualization with Isolation Forest Outliers')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.savefig(os.path.join(plot_dir, f"isolation_forest_tsne_{model_str}.png"))
plt.close()

# Print outlier detection statistics
print("\nIsolation Forest Statistics:")
print(f"Number of normal samples: {sum(is_inlier)}")
print(f"Number of outliers: {sum(~is_inlier)}")
print(f"Outlier percentage: {(sum(~is_inlier) / len(is_inlier)) * 100:.2f}%")

# Plot representative outlier light curves
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
    x = x.unsqueeze(0).to(device)  # Add batch dimension
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
