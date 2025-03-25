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

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="cont_multi_model",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-5,
    "architecture": "VAE",
    "dataset": "3 LCs",
    "epochs": 2000,
    "files": 10000
    }
)

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
    KLD = -0.003 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def Poisson_NLL(x_hat, x, mu, logvar):
    """
    Poisson Negative Log-Likelihood (NLL) Loss with KL divergence regularization.

    Args:
    - x_hat (Tensor): Predicted log-rates
    - x (Tensor): Observed photon counts
    - mu (Tensor): Mean from the VAE's latent space
    - logvar (Tensor): Log-variance from the VAE's latent space

    Returns:
    - loss (Tensor): Computed Poisson NLL loss with regularization
    """
    # Compute predicted rates with numerical stability
    log_rate = x_hat
    rate = torch.exp(log_rate)

    # Poisson NLL (using log-space for numerical stability)
    # log P(x|λ) = x log(λ) - λ - log(x!)
    # We can ignore log(x!) as it's constant with respect to λ
    poisson_term = -torch.mean(x * log_rate - rate)

    # Total loss
    loss = poisson_term

    return loss

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


        # NOTE: Here we use the pytorch functions pack_padded_sequence and pad_packed_sequence, which let us pad function
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

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
        else:
            output, hidden = self.gru(x, hidden)
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


print('start to load light curves')

fits_files = load_all_fits_files()
print('loaded files')

lc_low, lc_med, lc_high = load_n_light_curves(30000, fits_files, band = "all")

light_curves_sample = list(zip(lc_low, lc_med, lc_high))
print('finished loading lcs')
print(len(light_curves_sample))

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
train_dataset, val_dataset, test_dataset = partition_data(light_curves_sample)
# train_dataset = light_curves_sample

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

def collate_fn_err_mult(batch):

    rate_low = [torch.tensor(lc[0]['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr_low = [torch.tensor(lc[0]['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr_low = [torch.tensor(lc[0]['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    rate_med = [torch.tensor(lc[1]['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr_med = [torch.tensor(lc[1]['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr_med = [torch.tensor(lc[1]['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    rate_hi = [torch.tensor(lc[2]['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr_hi= [torch.tensor(lc[2]['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr_hi = [torch.tensor(lc[2]['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    sequences = [torch.stack([rl, lel, uel, rm, lem, uem, rh, leh, ueh], dim=-1) for rl, lel, uel, rm, lem, uem, rh, leh, ueh in zip(rate_low, lowErr_low, upErr_low, rate_med, lowErr_med, upErr_med, rate_hi, lowErr_hi, upErr_hi)]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)

    # Pad sequences
    x = pad_sequence(sequences, batch_first=True)

    return x, lengths

# Set up DataLoader
# TODO: Truncate Dtat
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_err_mult, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn_err_mult)


# Initialize the model and optimizer
model = RNN_VAE(input_size=9, hidden_size=512, latent_size=22, output_size=1).to(device)

model_str = 'RNN_Large_3bands'
new_model_str = 'RNN_Large_3bands_Cont_5000ep'
model_dir = "/home/pdong/Astro UROP/models"

plot_dir = "/home/pdong/Astro UROP/plots/" + new_model_str
os.makedirs(os.path.join(model_dir, new_model_str), exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

model.load_state_dict(torch.load('./models/' + model_str + '.h5'))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Number of epochs
nepochs = 5000

# Initialize lists to store training and validation losses
validation_losses = []
training_losses = []
benchmark_reconstructions = []
benchmark_idx = random.randint(0, len(train_dataset) - 1)

# Assuming `train_loader` and `val_loader` are PyTorch DataLoader objects
# The data loaders should yield batches with:
# - x: tensor of shape (batch_size, seq_length, 3)
# - lengths: tensor of sequence lengths
print("Beginning training...")
for epoch in range(nepochs):
    # Training phase
    model.train()  # Set the model to training mode
    train_loss = 0
    for batch in train_loader:
        x, lengths = batch
        x = x.to(device)
        lengths = lengths.cpu().to(torch.int64)  # Move lengths to CPU and ensure it is int64

        # Forward pass
        x_hat, mu, logvar = model(x, lengths) # Check these and see if they're the same across diff LCs
        # print(f"x_hat: {x_hat} | mu: {mu} | logvar: {logvar}")

        # Extract the rate component from x_hat
        x_hat_rate = x_hat[..., 0]

        # Compute loss (use x[..., 0] as target or modify if necessary)
        loss = ELBO(x_hat_rate, x[..., 0], mu, logvar)
        train_loss += loss.item()
        training_losses.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # print(1)

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, lengths = batch
            x = x.to(device)
            lengths = lengths.cpu().to(torch.int64)  # Move lengths to CPU and ensure it is int64

            # Forward pass
            x_hat, mu, logvar = model(x, lengths)

            # Extract the rate component from x_hat
            x_hat_rate = x_hat[..., 0]

            # Compute validation loss (focus on RATE)
            valid_loss += ELBO(x_hat_rate, x[..., 0], mu, logvar).item()  # Compare only with RATE
        # print('1')

    # Average losses
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(val_loader.dataset)
    if epoch % 100 == 0:
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

        # Generate a plot of the reconstruction
        benchmark_sample = train_dataset[benchmark_idx]
        x_benchmark, lengths_benchmark = collate_fn_err_mult([benchmark_sample])
        x_benchmark = x_benchmark.to(device)
        lengths_benchmark = lengths_benchmark.cpu().to(torch.int64)

        model.eval()
        with torch.no_grad():
            x_hat_benchmark, _, _ = model(x_benchmark, lengths_benchmark)

        x_hat_benchmark_rate = x_hat_benchmark[..., 0].detach().cpu().numpy()  # Convert to numpy
        x_original_rate = x_benchmark[..., 0].detach().cpu().numpy()

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_original_rate[0], label="Original RATE", color="blue", linestyle="dashed")
        ax.plot(x_hat_benchmark_rate[0], label="Reconstructed RATE", color="red", alpha=0.7)
        ax.legend()
        ax.set_title(f"Benchmark Reconstruction at Epoch {epoch + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Rate")

        # Log the figure to wandb
        wandb.log({"Reconstruction": wandb.Image(fig)})

        # Close the figure to prevent memory leaks
        plt.close(fig)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}')

        # Save the reconstructed benchmark curve at this epoch
        benchmark_sample = train_dataset[benchmark_idx]
        x_benchmark, lengths_benchmark = collate_fn_err_mult([benchmark_sample])
        x_benchmark = x_benchmark.to(device)
        lengths_benchmark = lengths_benchmark.cpu().to(torch.int64)
        x_hat_benchmark, _, _ = model(x_benchmark, lengths_benchmark)
        x_hat_benchmark_rate = x_hat_benchmark[..., 0].detach().cpu().numpy()  # Detach before converting to numpy
        benchmark_reconstructions.append((epoch + 1, x_hat_benchmark_rate[0]))



# Save the model
torch.save(model.state_dict(), os.path.join(model_dir, new_model_str + '.h5'))
print(f'Saved Model at {os.path.join(model_dir, new_model_str + ".h5")}')


model.eval()
with torch.no_grad():
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()

    # Determine the maximum sequence length in the validation dataset
    max_length = max(len(sample[1]['RATE']) for sample in train_dataset)

    for i in range(10):
        idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[idx]
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
    plt.savefig(os.path.join(plot_dir,'errorplot.png'))
    plt.show()

# Plot the benchmark curve reconstruction at different epochs
plt.figure(figsize=(10, 6))
for epoch, x_hat_benchmark_rate in benchmark_reconstructions:
    plt.plot(x_hat_benchmark_rate, label=f'Epoch {epoch}')
plt.legend()
plt.title('Benchmark Curve Reconstruction at Different Epochs')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.savefig(os.path.join(plot_dir,'benchmark_reconstruction.png'))
plt.show()


filename = os.path.join(plot_dir,"loss.txt")
with open(filename, "w") as file:
    for value in training_losses:
        file.write(f"{value}\n")

print('losses saved!')

plt.figure(figsize=(16, 10))
plt.plot(training_losses, marker="o", linestyle="-", color="b", label="Loss")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Line Plot of Losses")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(plot_dir,'loss.png'))

wandb.finish()

print("done!")
