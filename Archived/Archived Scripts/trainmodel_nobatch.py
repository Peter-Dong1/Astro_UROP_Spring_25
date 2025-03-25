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
    KLD = -0.3 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def Poisson_NLL(x_hat, x, mu, logvar):
    """
    Poisson Negative Log-Likelihood Loss.

    Args:
    - y_pred (Tensor): Predicted Poisson rate (must be positive).
    - y_true (Tensor): Observed photon counts.
    - mu (Tensor): Mean from the VAE's latent space.
    - logvar (Tensor): Log-variance from the VAE's latent space.

    Returns:
    - loss (Tensor): Computed Poisson NLL loss.
    """
    # Ensure positive predicted rates using exp
    lambda_pred = torch.exp(x_hat)  # Transform outputs to positive values

    # Poisson negative log likelihood loss
    poisson_nll = lambda_pred - x * torch.log(lambda_pred + 1e-8)  # Add epsilon for stability

    # KL Divergence to regularize latent space
    KLD = -0.05 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return poisson_nll.mean() + KLD  # Combine both losses
# our encoder class
class Encoder(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=4, dropout=0.2):
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

        print(f"EF Input shape: {x.shape}")               # (num_layers, batch_size, hidden_size)

        # NOTE: Here we use the pytorch functions pack_padded_sequence and pad_packed_sequence, which
        # allow us to
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        print(f"EF Packed Output shape: {packed_output.data.shape}")  # Packed sequences
        print(f"EF Output shape: {output.shape}")                    # (batch_size, seq_length, hidden_size)
        print(f"EF Hidden shape: {hidden.shape}")
        return output, hidden

# our decoder class
class Decoder(torch.nn.Module):
    def __init__(
        self, input_size=3, hidden_size=64, output_size=4, num_layers=4, dropout=0.2 # change hidden size to 128 later
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
        print(f"DF Input shape: {x.shape}")
        if lengths is not None:
            # unpad the light curves so that our latent representations learn only from real data
            packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed_x, hidden)

            # re-pad the light curves so that they can be processed elsewhere
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            print(f"DF1 Packed Output shape: {packed_output.data.shape}")  # Packed sequences
            print(f"DF1 Output shape: {output.shape}")                    # (batch_size, seq_length, hidden_size)
            print(f"DF1 Hidden shape: {hidden.shape}")
        else:
            output, hidden = self.gru(x, hidden)
            print(f"DF2 Output shape: {output.shape}")                    # (batch_size, seq_length, hidden_size)
            print(f"DF2 Hidden shape: {hidden.shape}")
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
        self.num_layers = 4
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

        # extract latent variable z
        mu = self.fc21(enc_h)
        logvar = self.fc22(enc_h)
        z = self.reparameterize(mu, logvar)

        # initialize hidden state
        h_ = self.fc3(z)
        h_ = h_.unsqueeze(0)  # Add an extra dimension for num_layers
        # Repeat the hidden state for each layer
        h_ = h_.repeat(self.dec.num_layers, 1, 1)  # Now h_ is [num_layers, batch_size, hidden_size]

        # decode latent space
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(device)

        # initialize hidden state
        hidden = h_.contiguous()
        x_hat, hidden = self.dec(z, hidden)

        return x_hat, mu, logvar


def create_dataframe_of_light_curves(n, fits_files, band = 'all'):
    """
    Create a dataframe of dataframes where each row corresponds to a light curve
    and each column represents a specific energy band.

    Parameters:
        fits_files (list): List of paths to FITS files.

    Returns:
        pd.DataFrame: A dataframe of dataframes.
    """

    data = []

    for file_path in fits_files:
        light_curve_low = load_light_curve(file_path, band=0)  # Low band
        light_curve_med = load_light_curve(file_path, band=1)  # Medium band
        light_curve_high = load_light_curve(file_path, band=2)  # High band

        data.append({
            'file_name': os.path.basename(file_path),
            'low_band': light_curve_low,
            'medium_band': light_curve_med,
            'high_band': light_curve_high
        })

    # Convert the list of dictionaries into a dataframe
    df_of_dfs = pd.DataFrame(data)
    return df_of_dfs

# Path to the directory containing the FITS files
data_dir = '/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned'

def read_inaccessible_lightcurves():
    """
    Read the list of inaccessible light curves from the text file in the notebook directory.

    Returns:
        list: List of file paths that were inaccessible
    """
    file_path = os.path.join(os.getcwd(), "inaccessible_lightcurves.txt")

    try:
        with open(file_path, 'r') as f:
            # Read all lines and remove any trailing whitespace
            inaccessible_files = [line.strip() for line in f.readlines()]
        return inaccessible_files
    except FileNotFoundError:
        print(f"No inaccessible light curves file found at {file_path}")
        return []

# Function to load a single FITS file and return as a Pandas DataFrame
def load_light_curve(file_path, band=1, trunc = 20):
    """
    Load light curve data from a FITS file and return a Pandas DataFrame including asymmetric errors (ERRM and ERRP).

    Parameters:
        file_path (str): Path to the FITS file.
        band (int): Energy band index to load data for (default: 1).

    Returns:
        pd.DataFrame or None: DataFrame with light curve data, or None if file is skipped.
    """
    with fits.open(file_path) as hdul:
        data = hdul[1].data  # Assuming light curve data is in the second HDU
        try:
            light_curve = pd.DataFrame({
                'TIME': data['TIME'],
                'TIMEDEL': data['TIMEDEL'],
                'RATE': data['RATE'][:, band],  # Light curve intensity
                'ERRM': data['RATE_ERRM'][:, band],  # Negative error
                'ERRP': data['RATE_ERRP'][:, band],  # Positive error
                'SYM_ERR': (data['RATE_ERRM'][:, band] + data['RATE_ERRP'][:, band]) / 2,  # Symmetric error approximation
            })

            # Truncate to a maximum of 20 data points
            if len(light_curve) > trunc:
                light_curve = light_curve.iloc[:trunc]
            # Attach metadata as attributes
            light_curve.attrs['FILE_NAME'] = os.path.basename(file_path)
            light_curve.attrs['OUTLIER'] = False
            return light_curve
        except KeyError:
            print(f"Skipping file {file_path}: some key not found")
            return None

def load_all_fits_files(data_dir = '/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned'):
    """
    Loads all fits files

    Parameters:
        data_dir (str): The filepath where the data is located

    Returns:
        fits_files (list): A list of all the fits files
    """

    return glob.glob(os.path.join(data_dir, "*.fits"))

def load_n_light_curves(n, fits_files, band = 'all'):
    """
    Loads a specified amount of light curves to analyze.

    Parameters:
        n (int): Number of light curves to load.
        fits_files (list): A list of all the fits files

    Returns:
        light_curves_1 (list): A list of n light curves in 0.2-0.6 keV,
        light_curves_2 (list): A list of n light curves in 0.6-2.3keV
        light_curves_3 (list): A list of n light curves in 2.3-5.0keV
    """

    inaccess_files = read_inaccessible_lightcurves()

    # Randomly select n files
    fits_files = random.sample(fits_files, n)

    temp = []
    for lc in  fits_files:
        if lc not in inaccess_files:
            temp.append(lc)
    fits_files = temp

    # Load all bands of the light curves into a list of DataFrames
    if band == 'all':
        light_curves_1 = [df for df in (load_light_curve(file, band = 0) for file in fits_files) if df is not None]
        light_curves_2 = [df for df in (load_light_curve(file, band = 1) for file in fits_files) if df is not None]
        light_curves_3 = [df for df in (load_light_curve(file, band = 2) for file in fits_files) if df is not None]

        return light_curves_1, light_curves_2, light_curves_3
    elif band == 'low':
        light_curves_1 = [df for df in (load_light_curve(file, band = 0) for file in fits_files) if df is not None]
        return light_curves_1
    elif band == 'med':
        total_files = len(fits_files)
        tenths = total_files // 10

        light_curves_2 = []
        for i, file in enumerate(fits_files):
            df = load_light_curve(file, band=1)
            if df is not None:
                light_curves_2.append(df)

            if (i + 1) % tenths == 0:
                print(f"Processed {(i + 1) / total_files:.0%} of files")

        return light_curves_2
    elif band == 'high':
        light_curves_3 = [df for df in (load_light_curve(file, band = 2) for file in fits_files) if df is not None]
        return light_curves_3
    else:
        raise KeyError("Input for Band is not valid")



def load_all_light_curves(fits_files):
    """
    Loads a specified amount of light curves to analyze.

    Parameters:
        n (int): Number of light curves to load.
        fits_files (list): A list of all the fits files

    Returns:
        light_curves_1 (list): A list of n light curves in 0.2-0.6 keV,
        light_curves_2 (list): A list of n light curves in 0.6-2.3keV
        light_curves_3 (list): A list of n light curves in 2.3-5.0keV
    """

    inaccess_files = read_inaccessible_lightcurves()

    temp = []
    for lc in  fits_files:
        if lc not in inaccess_files:
            temp.append(lc)
    fits_files = temp

    if band == 'all':
        light_curves_1 = [df for df in (load_light_curve(file, band = 0) for file in fits_files) if df is not None]
        light_curves_2 = [df for df in (load_light_curve(file, band = 1) for file in fits_files) if df is not None]
        light_curves_3 = [df for df in (load_light_curve(file, band = 2) for file in fits_files) if df is not None]

        return light_curves_1, light_curves_2, light_curves_3
    elif band == 'low':
        light_curves_1 = [df for df in (load_light_curve(file, band = 0) for file in fits_files) if df is not None]
        return light_curves_1
    elif band == 'med':
        light_curves_2 = [df for df in (load_light_curve(file, band = 1) for file in fits_files) if df is not None]
        return light_curves_2
    elif band == 'high':
        light_curves_3 = [df for df in (load_light_curve(file, band = 2) for file in fits_files) if df is not None]
        return light_curves_3
    else:
        raise KeyError("Input for Band is not valid")

print('start to load light curves')

fits_files = load_all_fits_files()
print('loaded files')

light_curves_sample = load_n_light_curves(1000, fits_files, band = "med")
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


# Initialize the model and optimizer
model = RNN_VAE(input_size=3, hidden_size=64, latent_size=50, output_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Number of epochs
nepochs = 20

# Initialize lists to store training and validation losses
validation_losses = []
training_losses = []
benchmark_reconstructions = []

# Beginning training without batching
print("Beginning training without batching...")
for epoch in range(nepochs):
    # Training phase
    model.train()  # Set the model to training mode
    train_loss = 0
    for sample in train_dataset:
        x, lengths = collate_fn_err([sample])  # Process a single light curve
        x = x.to(device)
        lengths = lengths.cpu().to(torch.int64)

        # Forward pass
        x_hat, mu, logvar = model(x, lengths)

        # Extract the rate component from x_hat
        x_hat_rate = x_hat[..., 0]

        # Compute loss
        loss = Poisson_NLL(x_hat_rate, x[..., 0], mu, logvar)
        train_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation phase
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for sample in val_dataset:
            x, lengths = collate_fn_err([sample])
            x = x.to(device)
            lengths = lengths.cpu().to(torch.int64)

            # Forward pass
            x_hat, mu, logvar = model(x, lengths)

            # Extract the rate component from x_hat
            x_hat_rate = x_hat[..., 0]

            # Compute validation loss
            valid_loss += Poisson_NLL(x_hat_rate, x[..., 0], mu, logvar).item()

    # Average losses
    train_loss /= len(train_dataset)
    valid_loss /= len(val_dataset)

    print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.2f} | Valid Loss: {valid_loss:.2f}')

    # Save benchmark reconstruction at this epoch
    benchmark_idx = 3
    benchmark_sample = val_dataset[benchmark_idx]
    x_benchmark, lengths_benchmark = collate_fn_err([benchmark_sample])
    x_benchmark = x_benchmark.to(device)
    lengths_benchmark = lengths_benchmark.cpu().to(torch.int64)
    x_hat_benchmark, _, _ = model(x_benchmark, lengths_benchmark)
    x_hat_benchmark_rate = x_hat_benchmark[..., 0].detach().cpu().numpy()
    benchmark_reconstructions.append((epoch + 1, x_hat_benchmark_rate[0]))

# Save the model
torch.save(model.state_dict(), './RNN_VAE_Err_100Epochs_NoBatching.h5')
