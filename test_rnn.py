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
from helper import load_light_curve, load_n_light_curves, load_all_fits_files, partition_data

from RNN_9_model import ELBO, Poisson_NLL, RNN_VAE, Decoder, Encoder, collate_fn_err_mult
import json

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
model_str = "30000_files_RNN"
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


test_data_size = 10

plot_dir = "/home/pdong/Astro UROP/plots/RNN Models/" + model_str + "/" + str(test_data_size)
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

print('Start Partitioning Data ...')
train_dataset, val_dataset, test_dataset = partition_data(light_curves_sample)
print('Partitioning Complete')

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

model.eval()

test_loss = 0.0
test_reconstruction_loss = 0.0  # if you'd like a separate reconstruction metric
num_samples = 0




test_loader = DataLoader(light_curves_sample, batch_size=32, collate_fn=collate_fn_err)
model.eval()

num_curves = 100
grid_size = 10

with torch.no_grad():
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))  # Create a 10x10 grid
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    i = 0
    for batch in test_loader:
        if i >= num_curves:  # Stop after plotting 100 light curves
            break

        x_test, lengths_test = batch
        x_test = x_test.to(device)

        # Forward pass
        x_hat, _, _ = model(x_test, lengths_test)

        # Iterate over the batch and plot each curve
        for j in range(x_test.size(0)):  # Batch size could be >1
            if i >= num_curves:  # Stop if we've already plotted 100 curves
                break

            original_curve = x_test[j].squeeze(-1).cpu().numpy()  # shape: (seq_len,)
            reconstructed_curve = x_hat[j].squeeze(-1).cpu().numpy()  # shape: (seq_len,)

            # Plot the curve in the corresponding subplot
            axes[i].plot(original_curve, label='Original', alpha=0.7)
            axes[i].plot(reconstructed_curve, label='Reconstructed', alpha=0.7)
            axes[i].set_title(f"Curve {i+1}")
            axes[i].set_xlabel("Time Index")
            axes[i].set_ylabel("Rate")
            axes[i].legend()
            i += 1

    plt.tight_layout()
    plt.show()
    plt.savefig('plot2.png')

with torch.no_grad():
    for batch in test_loader:
        x_test, lengths_test = batch
        x_test = x_test.to(device)

        # Forward pass
        x_hat, mu, logvar = model(x_test, lengths_test)

        # 1. ELBO Loss
        loss = ELBO(x_hat, x_test, mu, logvar)
        test_loss += loss.item()

        # 2. Additional reconstruction metric (e.g., MSE)
        #    We'll assume x_hat and x_test have shape (batch_size, seq_length, features=1)
        recon_error = torch.nn.functional.mse_loss(x_hat, x_test, reduction='sum')
        test_reconstruction_loss += recon_error.item()

        # Keep track of how many total points you had in the batch
        num_samples += x_test.size(0) * x_test.size(1)

# Divide by the total number of samples to get average losses
test_loss /= len(test_loader.dataset)
test_reconstruction_loss /= num_samples

print(f"Average ELBO on test set: {test_loss:.4f}")
print(f"Average MSE Reconstruction Error on test set: {test_reconstruction_loss:.4f}")

latents = []
with torch.no_grad():
    for batch in test_loader:
        x_test, lengths_test = batch
        x_test = x_test.to(device)

        _, mu, _ = model(x_test, lengths_test)
        latents.append(mu.cpu().numpy())

latents = np.concatenate(latents, axis=0)

# PCA Visualization
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latents)

plt.figure(figsize=(8, 8))
plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.7)
plt.title("Latent Space Visualization (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
plt.savefig('Latent Space PCA.png')

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(latents)

plt.figure(figsize=(8, 8))
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.7)
plt.title("Latent Space Visualization (t-SNE)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()
plt.savefig('Latent Space TSNE.png')

reconstruction_errors = []
with torch.no_grad():
    for batch in test_loader:
        x_test, lengths_test = batch
        x_test = x_test.to(device)

        x_hat, _, _ = model(x_test, lengths_test)
        recon_error = torch.nn.functional.mse_loss(x_hat, x_test, reduction='none').cpu().numpy()
        reconstruction_errors.append(recon_error.mean(axis=(1, 2)))  # Mean per sample

reconstruction_errors = np.concatenate(reconstruction_errors, axis=0)

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_errors, bins=30, alpha=0.7)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()
plt.savefig('Reconstruction error dist.png')


# This part not working rn idk why
top_outlier_indices = np.argsort(reconstruction_errors)[-10:]  # Indices of top 10 outliers

with torch.no_grad():
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))  # Create a 5x2 grid for subplots
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for subplot_idx, i in enumerate(top_outlier_indices):
        x_test, lengths_test = test_loader.dataset[i]
        x_test = x_test.unsqueeze(0).to(device)  # Add batch dimension
        lengths_test = torch.tensor([len(x_test)], dtype=torch.long).to(device)

        x_hat, _, _ = model(x_test, lengths_test)

        original_curve = x_test.squeeze().cpu().numpy()
        reconstructed_curve = x_hat.squeeze().cpu().numpy()

        # Plot each outlier in a subplot
        axes[subplot_idx].plot(original_curve, label="Original", alpha=0.7)
        axes[subplot_idx].plot(reconstructed_curve, label="Reconstructed", alpha=0.7)
        axes[subplot_idx].set_title(f"Outlier Curve {i+1} (Reconstruction Error: {reconstruction_errors[i]:.4f})")
        axes[subplot_idx].set_xlabel("Time Index")
        axes[subplot_idx].set_ylabel("Rate")
        axes[subplot_idx].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("outlierplot.png")
