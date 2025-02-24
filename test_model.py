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

# set the device we're using for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())

print('torch diagonostics')
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)
print(torch.cuda.is_available())

# the evidence lower bound loss for training autoencoders
# TODO: pass in another variable call mask
# the evidence lower bound loss for training autoencoders
def ELBO(x_hat, x, mu, logvar):
    # the reconstruction loss
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)

    # the KL-divergence between the latent distribution and a multivariate normal
    KLD = -0.01 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE
def Poisson_NLL(x_hat, x, mu, logvar):
    """
    Poisson Negative Log-Likelihood (NLL) Loss with KL Divergence.

    Args:
    - x_hat (Tensor): Predicted Poisson log-rates.
    - x (Tensor): Observed photon counts (must be non-negative integers).
    - mu (Tensor): Mean from the VAE's latent space.
    - logvar (Tensor): Log-variance from the VAE's latent space.

    Returns:
    - loss (Tensor): Computed Poisson NLL loss.
    """
    # Ensure positive predicted rates by exponentiating x_hat
    lambda_pred = torch.exp(x_hat)  # Poisson rate (must be positive)

    # Poisson negative log-likelihood loss
    lambda_pred = torch.exp(x_hat) + 1e-4  # Avoid too small values
    poisson_nll = lambda_pred - x * torch.log(lambda_pred + 1e-8)

    # Optional: Add the factorial term for completeness (can be ignored)
    # poisson_nll += torch.lgamma(x + 1)  # lgamma(x+1) computes log(x!)

    # KL Divergence to regularize the latent space
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Return total loss (Poisson NLL + KL divergence)
    return poisson_nll.mean() # Scale KL by a small factor
# our encoder class
class Encoder(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=8, num_layers=1, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, lengths):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # lengths: tensor of shape (batch_size), containing the lengths of each sequence in the batch

        # print(f"EF Input shape: {x.shape}")               # (num_layers, batch_size, hidden_size)

        # NOTE: Here we use the pytorch functions pack_padded_sequence and pad_packed_sequence, which
        # allow us to
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # print(f"EF Packed Output shape: {packed_output.data.shape}")  # Packed sequences
        # print(f"EF Output shape: {output.shape} | Hidden shape: {hidden.shape}")                    # (batch_size, seq_length, hidden_size)
        return output, hidden

# our decoder class
class Decoder(torch.nn.Module):
    def __init__(
        self, input_size=3, hidden_size=8, output_size=1, num_layers=1, dropout=0.2 # change hidden size to 128 later
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, lengths=None):
        # print(f"DF Input shape: {x.shape}")
        if lengths is not None: # not being used
            # unpad the light curves so that our latent representations learn only from real data
            packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed_x, hidden)

            # re-pad the light curves so that they can be processed elsewhere
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            # print(f"DF1 Packed Output shape: {packed_output.data.shape}")  # Packed sequences
            # print(f"DF1 Output shape: {output.shape}")                    # (batch_size, seq_length, hidden_size)
            # print(f"DF1 Hidden shape: {hidden.shape}")
        else:
            output, hidden = self.gru(x, hidden)
            # print(f"DF2 Output shape: {output.shape} | Hidden shape: {hidden.shape}")                    # (batch_size, seq_length, hidden_size)
        prediction = self.fc(output)
        prediction = torch.exp(prediction)  # Ensure positive outputs for Poisson rate
        return prediction, hidden

class RNN_VAE(torch.nn.Module): # TODO: Print out shapes of the things
    """RNN-VAE: A Variational Auto-Encoder with a Recurrent Neural Network Layer as the Encoder."""

    def __init__(
        self, input_size=3, hidden_size=64, latent_size=50, dropout=0.2, output_size=1
    ):
        """
        input_size: int, batch_size x sequence_length x input_dim
        hidden_size: int, output size
        latent_size: int, latent z-layer size
        num_gru_layer: int, number of layers
        """
        super(RNN_VAE, self).__init__()
        self.device = device

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = 2
        self.dropout = dropout

        self.enc = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers, dropout=self.dropout)

        self.dec = Decoder(
            input_size=latent_size,
            output_size=output_size,
            hidden_size=hidden_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )

        self.fc21 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = torch.nn.Linear(self.latent_size, self.hidden_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn(mu.shape).to(device)*torch.exp(0.5*logvar)
        else:
            return mu

    def forward(self, x, lengths):
        batch_size, seq_len, feature_dim = x.shape

        # encode input space
        enc_output, enc_hidden = self.enc(x, lengths)

        # Correctly accessing the hidden state of the last layer
        enc_h = enc_hidden[-1].to(device)  # This is now [batch_size, hidden_size]
        # print(f"Hidden State of Last Layer: {enc_hidden[-1].shape}")

        # extract latent variable z
        mu = self.fc21(enc_h)
        logvar = self.fc22(enc_h)
        z = self.reparameterize(mu, logvar)
        # print(f"mu: {mu} | logvar: {logvar} | z: {z}")
        # print(f"Mean of mu: {mu.mean().item()}, Std of mu: {mu.std().item()}")
        # print(f"Mean of logvar: {logvar.mean().item()}, Std of logvar: {logvar.std().item()}")

        # initialize hidden state
        h_ = self.fc3(z) # Shape: (batch_size, hidden_size)
        h_ = h_.unsqueeze(0)  # Add an extra dimension for num_layers
        # Repeat the hidden state for each layer
        h_ = h_.repeat(self.dec.num_layers, 1, 1)  # Now h_ is [num_layers, batch_size, hidden_size]

        # print(f"z: {z.shape}")

        # decode latent space
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(device)

        # initialize hidden state
        hidden = h_.contiguous() # just for effieenciy - stored in same memory
        x_hat, hidden = self.dec(z, hidden) # runs decoder GRU

        return x_hat, mu, logvar

# Partition into train set and test set
def partition_data(light_curves, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Partition a list of light curves into train, validation, and test sets.

    Parameters:
        light_curves (list): List of light curve DataFrames.
        test_size (float): Proportion of data to use for the test set.
        val_size (float): Proportion of train data to use for the validation set.
        random_seed (int): Random seed for reproducibility.

    Returns:
        train_set (list): List of light curves for training.
        val_set (list): List of light curves for validation (if val_size > 0).
        test_set (list): List of light curves for testing.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Split into train+val and test sets
    train_val_set, test_set = train_test_split(light_curves, test_size=test_size, random_state=random_seed)

    if val_size > 0:
        # Split train_val into train and validation sets
        train_size = 1 - val_size
        train_set, val_set = train_test_split(train_val_set, test_size=val_size, random_state=random_seed)
        return train_set, val_set, test_set
    else:
        # If no validation set is needed, return only train and test sets
        return train_val_set, test_set

# Set up DataLoader
def collate_fn_err(batch):
    rate = [torch.tensor(lc['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr = [torch.tensor(lc['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr = [torch.tensor(lc['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    sequences = [torch.stack([r, le, ue], dim=-1) for r, le, ue in zip(rate, lowErr, upErr)]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)

    # Pad sequences
    x = pad_sequence(sequences, batch_first=True)

    return x, lengths


print('start to load light curves')

fits_files = load_all_fits_files()

test_dataset = load_n_light_curves(1024, fits_files, band = "med")
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_err, shuffle=True)

print('finish loading lcs')

model = RNN_VAE(input_size=3, hidden_size=256, latent_size=22, output_size=1).to(device)
model_str = 'more_bands_16384_10k'
model.load_state_dict(torch.load('./models/' + model_str + '.h5'))

plot_dir = "/home/pdong/Astro UROP/plots/" + model_str
# Ensure the directory exists
os.makedirs(plot_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))
    axes = axes.flatten()

    # Determine the maximum sequence length in the validation dataset
    max_length = max(len(sample['RATE']) for sample in test_dataset)

    for i in range(25):
        idx = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[idx]
        x, lengths = collate_fn_err([sample])
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
pca = PCA(n_components=2)
latent_2d_pca = pca.fit_transform(latent_vectors)

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
cluster_labels_pca = hdbscan_pca.fit_predict(latent_2d_pca)

# Cluster using t-SNE representation
print("Clustering t-SNE representation...")
hdbscan_tsne = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
cluster_labels_tsne = hdbscan_tsne.fit_predict(latent_2d_tsne)

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

# Plot clustered t-SNE results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1],
                     c=cluster_labels_tsne, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('HDBSCAN Clustering on t-SNE Representation')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig(os.path.join(plot_dir, f"hdbscan_tsne_{model_str}.png"))
plt.close()

# Print clustering statistics
print("\nClustering Statistics:")
print("PCA-based clustering:")
print(f"Number of clusters: {len(set(cluster_labels_pca)) - (1 if -1 in cluster_labels_pca else 0)}")
print(f"Number of noise points: {sum(1 for label in cluster_labels_pca if label == -1)}")

print("\nt-SNE-based clustering:")
print(f"Number of clusters: {len(set(cluster_labels_tsne)) - (1 if -1 in cluster_labels_tsne else 0)}")
print(f"Number of noise points: {sum(1 for label in cluster_labels_tsne if label == -1)}")
