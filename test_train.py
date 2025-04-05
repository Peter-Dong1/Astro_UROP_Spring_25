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

import wandb
import json

new_model_str = "30000_files_RNN"
learning_rate = 1e-5
data_size = 30000
num_epochs = 3000
latent_size = 32
KLD_coef = 0.0035
hidden_size = 512
input_size = 9
output_size = 1
batch_size = 32

# Define a dictionary with all hyperparameters
hyperparams = {
    "model_name": new_model_str,
    "learning_rate": learning_rate,
    "data_size": data_size,
    "num_epochs": num_epochs,
    "latent_size": latent_size,
    "KLD_coef": KLD_coef,
    "hidden_size": hidden_size,
    "input_size": input_size,
    "output_size": output_size,
    "batch_size": batch_size
}

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

wandb.init(
    # set the wandb project where this run will be logged
    project="RNN_BIG",

    # track hyperparameters and run metadata
    config=hyperparams
)

# set the device we're using for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Torch Diagnostics')
print(f'Available: {torch.cuda.is_available()}')
print(f'Version: {torch.__version__}, {torch.version.cuda}')

print('Start Loading Files ...')
fits_files = load_all_fits_files()
print('Loaded Files Complete')

print('Start Loading Light Curves ...')
lc_low, lc_med, lc_high = load_n_light_curves(data_size, fits_files, band = "all")
light_curves_sample = list(zip(lc_low, lc_med, lc_high))
print(f'Loading Light Curves Complete. Length: {len(light_curves_sample)}')

print('Start Partitioning Data ...')
train_dataset, val_dataset, test_dataset = partition_data(light_curves_sample)
print('Partitioning Complete')

# Set up DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_err_mult, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_err_mult)


# Initialize the model and optimizer
model = RNN_VAE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, output_size=output_size, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#TODO: Here

validation_losses = []
training_losses = []
benchmark_reconstructions = []
benchmark_idx = [random.randint(0, len(train_dataset) - 1) for i in range(5)]

# Assuming `train_loader` and `val_loader` are PyTorch DataLoader objects
# The data loaders should yield batches with:
# - x: tensor of shape (batch_size, seq_length, 3)
# - lengths: tensor of sequence lengths
print("Beginning training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    train_loss = 0
    for batch in train_loader:
        x, lengths = batch
        x = x.to(device)
        lengths = lengths.cpu().to(torch.int64)  # Move lengths to CPU and ensure it is int64

        # Forward pass
        x_hat, mu, logvar = model(x, lengths) # Check these and see if they're the same across diff LCs

        # Extract the rate component from x_hat
        x_hat_rate = x_hat[..., 0]

        # Compute loss (use x[..., 0] as target or modify if necessary)
        loss = ELBO(x_hat_rate, x[..., 0], mu, logvar)
        train_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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

    # Average losses
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(val_loader.dataset)
    training_losses.append(train_loss)

    if epoch % 50 == 0:
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})
        wandb.log({"epoch": epoch + 1, "val_loss": valid_loss})

        # Generate a plot of the reconstruction
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, bm_inx in enumerate(benchmark_idx[:4]):
            benchmark_sample = train_dataset[bm_inx]
            x_benchmark, lengths_benchmark = collate_fn_err_mult([benchmark_sample])
            x_benchmark = x_benchmark.to(device)
            lengths_benchmark = lengths_benchmark.cpu().to(torch.int64)

            model.eval()
            with torch.no_grad():
                x_hat_benchmark, _, _ = model(x_benchmark, lengths_benchmark)

            x_hat_benchmark_rate = x_hat_benchmark[..., 0].detach().cpu().numpy()  # Convert to numpy
            x_original_rate = x_benchmark[..., 0].detach().cpu().numpy()


            axes[i].plot(x_original_rate[0], label="Original RATE", color="blue", linestyle="dashed")
            axes[i].plot(x_hat_benchmark_rate[0], label="Reconstructed RATE", color="red", alpha=0.7)
            axes[i].legend()
            axes[i].set_title(f"Benchmark {i+1} at Epoch {epoch + 1}")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Rate")

        plt.tight_layout()

        # Log the combined figure to wandb
        wandb.log({f"Reconstruction curves": wandb.Image(fig)})

        # Close the figure to prevent memory leaks
        plt.close(fig)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.2f} | Valid Loss: {valid_loss:.2f}')

        # Save the reconstructed benchmark curve at this epoch
        benchmark_sample = train_dataset[benchmark_idx[0]]
        x_benchmark, lengths_benchmark = collate_fn_err_mult([benchmark_sample])
        x_benchmark = x_benchmark.to(device)
        lengths_benchmark = lengths_benchmark.cpu().to(torch.int64)
        x_hat_benchmark, _, _ = model(x_benchmark, lengths_benchmark)
        x_hat_benchmark_rate = x_hat_benchmark[..., 0].detach().cpu().numpy()  # Detach before converting to numpy
        benchmark_reconstructions.append((epoch + 1, x_hat_benchmark_rate[0]))


save_dir = "/home/pdong/Astro UROP/models/RNN Models"
# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, new_model_str + ".h5"))
print(f'Saved Model at {os.path.join(save_dir, new_model_str + ".h5")}')

plot_dir = "/home/pdong/Astro UROP/training_plots/RNN\ Models/" + new_model_str
os.makedirs(plot_dir, exist_ok=True)

# Save hyperparameters to config file in the plotting folder
config_path = os.path.join(plot_dir, "config.json")
save_config(hyperparams, config_path)

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
    plt.savefig(os.path.join(plot_dir, 'errorplot.png'))
    plt.show()



# Plot the benchmark curve reconstruction at different epochs
plt.figure(figsize=(10, 6))
for epoch, x_hat_benchmark_rate in benchmark_reconstructions:
    plt.plot(x_hat_benchmark_rate, label=f'Epoch {epoch}')
plt.legend()
plt.title('Benchmark Curve Reconstruction at Different Epochs')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.savefig(os.path.join(plot_dir, 'benchmark_reconstruction.png'))
plt.show()


filename = os.path.join(plot_dir, "loss.txt")
with open(filename, "w") as file:
    for value in training_losses:
        file.write(f"{value}\n")

print('losses saved!')

plt.figure(figsize=(8, 5))
plt.plot(training_losses, marker="o", linestyle="-", color="b", label="Loss")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Line Plot of Losses")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(plot_dir, 'loss.png'))

wandb.finish()

print('Done!')
