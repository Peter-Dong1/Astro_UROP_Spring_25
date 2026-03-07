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


# the evidence lower bound loss for training autoencoders
# TODO: pass in another variable call mask
# the evidence lower bound loss for training autoencoders
def ELBO(x_hat, x, mu, logvar, KLD_coef = 0.001):
    # the reconstruction loss
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)

    # the KL-divergence between the latent distribution and a multivariate normal
    KLD = -KLD_coef * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
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
        self, input_size=3, hidden_size=64, latent_size=50, dropout=0.2, output_size=1, device = "cpu"
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
            return mu + torch.randn(mu.shape).to(self.device)*torch.exp(0.5*logvar)
        else:
            return mu

    def forward(self, x, lengths):
        batch_size, seq_len, feature_dim = x.shape

        # encode input space
        enc_output, enc_hidden = self.enc(x, lengths)

        # Correctly accessing the hidden state of the last layer
        enc_h = enc_hidden[-1].to(self.device)  # This is now [batch_size, hidden_size]
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
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)

        # initialize hidden state
        hidden = h_.contiguous() # just for effieenciy - stored in same memory
        x_hat, hidden = self.dec(z, hidden) # runs decoder GRU

        return x_hat, mu, logvar

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
