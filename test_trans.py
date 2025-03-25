import os
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import random
import math

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from helper import load_light_curve, load_n_light_curves, load_all_fits_files
from trans_model import TransformerVAE, partition_data, collate_fn_err, ELBO

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Load and prepare data
print('Loading light curves...')
fits_files = load_all_fits_files()
print('finished getting all files')
lc_low, lc_med, lc_high = load_n_light_curves(10, fits_files, band="all")
light_curves_sample = list(zip(lc_low, lc_med, lc_high))
print(f'Loaded {len(light_curves_sample)} light curves')

# Partition data
train_set, val_set, test_set = partition_data(light_curves_sample)
print(f"Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}")

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_err)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_err)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_err)

# Initialize model
model = TransformerVAE(
    input_size=9,  # 1 time +  3 bands x 3 values (RATE, ERRM, ERRP)
    d_model=256,
    nhead=4,
    num_encoder_layers=2,
    latent_size=40,
    hidden_size=256,
    num_decoder_blocks=3,
    dropout=0.05
).to(device)

model_str = 'trans_100k_1500'
model.load_state_dict(torch.load('./models/' + model_str + '.h5'))

# Create output directory for plots
plot_dir = "/home/pdong/Astro UROP/plots/" + model_str
os.makedirs(plot_dir, exist_ok=True)



def plot_top_25_losses(model, data_loader, losses, plot_dir):
    """
    Plot the light curves and their reconstructions for the 25 samples with highest losses.
    """
    model.eval()
    os.makedirs(plot_dir, exist_ok=True)

    # Sort losses in descending order and get top 25
    top_25_losses = sorted(losses, key=lambda x: x[1], reverse=True)[:25]

    # Create a dictionary to store data for the top 25 samples
    top_samples_data = {}
    current_batch_idx = -1
    current_batch_data = None
    current_batch_lengths = None

    with torch.no_grad():
        for batch_idx, (data, lengths) in enumerate(data_loader):
            # Check if this batch contains any of our top 25 samples
            batch_start_idx = batch_idx * data_loader.batch_size
            batch_end_idx = batch_start_idx + len(data)

            # Store batch data if it contains any of our top samples
            for sample_idx, loss in top_25_losses:
                if batch_start_idx <= sample_idx < batch_end_idx:
                    if current_batch_idx != batch_idx:
                        current_batch_idx = batch_idx
                        current_batch_data = data.to(device)
                        current_batch_lengths = lengths
                        x_hat, mu, logvar = model(current_batch_data, current_batch_lengths)

                    # Get the index within the batch
                    batch_pos = sample_idx - batch_start_idx

                    # Store the original and reconstructed data
                    x = current_batch_data[batch_pos].cpu().numpy()
                    x_hat_sample = x_hat[batch_pos].cpu().numpy()

                    # Create the plot
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
                    bands = ['Low', 'Medium', 'High']

                    for i, (band, ax) in enumerate(zip(bands, axes)):
                        # Original data
                        rate_idx = i * 3
                        ax.errorbar(range(len(x)), x[:, rate_idx],
                              yerr=[x[:, rate_idx+1], x[:, rate_idx+2]],
                              fmt='o', label=f'Original {band}')

                        # Reconstructed data
                        ax.scatter(range(len(x_hat_sample)), x_hat_sample[:, rate_idx],
                                 marker='x', label=f'Reconstructed {band}')

                        ax.set_title(f'{band} Energy Band')
                        ax.set_xlabel('Time Step')
                        ax.set_ylabel('Rate')
                        ax.legend()

                    plt.suptitle(f'Light Curve Reconstruction\nSample Index: {sample_idx}\nLoss: {loss:.4f}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'high_loss_{sample_idx}.png'))
                    plt.close()

def compute_and_plot_latent_space(model, loader, plot_dir):
    """
    Extract latent representations and visualize using t-SNE and PCA
    """
    model.eval()
    # latent_vectors = []

    # # Get latent representations
    # with torch.no_grad():
    #     for batch in loader:
    #         data, lengths = batch
    #         data = data.to(device)
    #         # Get latent representations without sampling
    #         mu, _ = model.encode(data, lengths)
    #         latent_vectors.append(mu.cpu().numpy())

    # latent_vectors = np.vstack(latent_vectors)

    # # t-SNE visualization
    # tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    # latent_2d_tsne = tsne.fit_transform(latent_vectors)

    # plt.figure(figsize=(10, 10))
    # plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], alpha=0.5)
    # plt.title('t-SNE visualization of latent space')
    # plt.xlabel('t-SNE dimension 1')
    # plt.ylabel('t-SNE dimension 2')
    # plt.savefig(os.path.join(plot_dir, 'tsne_latent_space.png'))
    # plt.close()

    # # PCA visualization
    # pca = PCA(n_components=2)
    # latent_2d_pca = pca.fit_transform(latent_vectors)

    # plt.figure(figsize=(10, 10))
    # plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], alpha=0.5)
    # plt.title('PCA visualization of latent space')
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.savefig(os.path.join(plot_dir, 'pca_latent_space.png'))
    # plt.close()
    print("Extracting latent representations...")
    latent_vectors = []
    with torch.no_grad():
        for x, lengths in test_loader:
            x = x.to(device)
            _, mu, _ = model(x, lengths)
            # Ensure mu is 2D before appending
            if len(mu.shape) > 2:
                mu = mu.reshape(mu.shape[0], -1)  # Flatten all dimensions after batch
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)
    print(f"Latent vectors shape: {latent_vectors.shape}")

    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)

    # Create PCA plot
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], alpha=0.5)
    plt.title('PCA Visualization of Latent Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(os.path.join(plot_dir, f"latent_space_pca {model_str}.png"))
    plt.close()

    # Only do t-SNE if we have enough samples (at least 10)
    n_samples = latent_vectors.shape[0]
    print(f"Number of samples: {n_samples}")

    if n_samples >= 10:
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d_tsne = tsne.fit_transform(latent_vectors)

        # Create t-SNE plot
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization of Latent Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig(os.path.join(plot_dir, f"latent_space_tsne {model_str}.png"))
        plt.close()
    else:
        print("Skipping t-SNE due to insufficient samples (need at least 10)")

    # Print explained variance ratio for PCA
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)


def plot_reconstruction_samples(model, test_loader, plot_dir, num_samples=5):
    """
    Plot random samples of original vs reconstructed light curves
    """
    model.eval()
    os.makedirs(plot_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (data, lengths) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            data = data.to(device)
            x_hat, mu, logvar = model(data, lengths)

            # Plot for each sample in the batch
            for i in range(min(data.size(0), 2)):  # Plot up to 2 samples per batch
                original = data[i].cpu().numpy()
                reconstructed = x_hat[i].cpu().numpy()

                fig, axes = plt.subplots(3, 1, figsize=(12, 12))
                bands = ['Low', 'Medium', 'High']

                for j, (band, ax) in enumerate(zip(bands, axes)):
                    # Original data
                    rate_idx = j * 3
                    ax.errorbar(range(len(original)), original[:, rate_idx],
                              yerr=[original[:, rate_idx+1], original[:, rate_idx+2]],
                              fmt='o', label=f'Original {band}')

                    # Reconstructed data
                    ax.errorbar(range(len(reconstructed)), reconstructed[:, rate_idx],
                              yerr=[reconstructed[:, rate_idx+1], reconstructed[:, rate_idx+2]],
                              fmt='o', label=f'Reconstructed {band}')

                    ax.set_title(f'{band} Energy Band')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Rate')
                    ax.legend()

                plt.suptitle(f'Sample {batch_idx}_{i} Reconstruction')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'reconstruction_sample_{batch_idx}_{i}.png'))
                plt.close()

# Compute losses for all test samples
print("Computing losses for test samples...")
losses = []
model.eval()

with torch.no_grad():
    for batch_idx, (data, lengths) in enumerate(train_loader):
        data = data.to(device)
        x_hat, mu, logvar = model(data, lengths)

        # Compute loss for each sample in the batch
        for i in range(len(data)):
            loss = ELBO(x_hat[i:i+1], data[i:i+1], mu[i:i+1], logvar[i:i+1]).item()
            sample_idx = batch_idx * train_loader.batch_size + i
            losses.append((sample_idx, loss))

# Generate plots
print("Generating plots...")
plot_top_25_losses(model, train_loader, losses, plot_dir)
# plot_reconstruction_samples(model, train_loader, plot_dir)
compute_and_plot_latent_space(model, train_loader, plot_dir)

print(f"Testing and visualization complete. Check the {plot_dir} directory for results.")
