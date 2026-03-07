import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from helper import load_light_curve, load_n_light_curves, load_all_fits_files, partition_data

import wandb
import json

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for temporal information in light curves
    """
    def __init__(self, num_freqs=6, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.include_input = include_input
        self.code_size = 2 * num_freqs + 1 * (self.include_input)

        # Register frequencies and phases as buffers
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, 1, -1)
        )
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, 1, -1))

    def forward(self, t):
        """
        Apply positional encoding to temporal data
        Args:
            t: tensor of shape (batch_size, seq_len, 1) containing time values
        Returns:
            Encoded tensor of shape (batch_size, seq_len, code_size)
        """
        B, n, _ = t.shape
        coded_t = t.repeat(1, 1, self.num_freqs * 2)
        coded_t = torch.sin(torch.addcmul(self._phases, coded_t, self._freqs))

        if self.include_input:
            coded_t = torch.cat((t, coded_t), dim=-1)
        return coded_t

class ResnetBlock(nn.Module):
    """
    Fully connected ResNet Block with pre-activation
    """
    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()

        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.activation = nn.ReLU()

        # Shortcut connection if input/output dimensions differ
        self.shortcut = None if size_in == size_out else nn.Linear(size_in, size_out)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc_0, self.fc_1]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        if self.shortcut is not None:
            nn.init.kaiming_normal_(self.shortcut.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        identity = x

        out = self.activation(x)
        out = self.fc_0(out)
        out = self.activation(out)
        out = self.fc_1(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return out + identity

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing light curve sequences
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()

        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Initialize weights
        nn.init.kaiming_normal_(self.input_linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_linear.bias, 0)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Padding mask (batch_size, seq_len)
        """
        x = self.input_linear(x)
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        return x

class Decoder(nn.Module):
    """
    Decoder with residual connections
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super().__init__()

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_linear = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        nn.init.kaiming_normal_(self.input_linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_linear.bias, 0)
        nn.init.kaiming_normal_(self.output_linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_linear.bias, 0)

    def forward(self, x):
        x = self.input_linear(x)
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)

# Loss functions
def ELBO(x_hat, x, mu, logvar):
    # Reconstruction loss (MSE)
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)
    # KL divergence
    # KLD = -0.0001 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE
    # return MSE + KLD

def Poisson_NLL(x_hat, x, mu, logvar):
    """
    Poisson Negative Log-Likelihood Loss with KL Divergence.
    """
    # Ensure positive predicted rates
    lambda_pred = torch.exp(x_hat) + 1e-4
    # Poisson NLL
    poisson_nll = lambda_pred - x * torch.log(lambda_pred + 1e-8)
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return poisson_nll.mean() + 0.01 * KLD

class TransformerVAE(nn.Module):
    """
    Transformer-based Variational Autoencoder for light curve analysis
    """
    def __init__(
        self,
        input_size=3,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        latent_size=50,
        hidden_size=64,
        num_decoder_blocks=3,
        dropout=0.1
    ):
        super().__init__()

        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(num_freqs=6)
        pos_encoded_dim = self.pos_encoder.code_size + input_size - 1  # -1 because time dimension is encoded

        # Transformer encoder
        self.encoder = TransformerEncoder(
            input_dim=pos_encoded_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        # Latent space projections
        self.fc_mu = nn.Linear(d_model, latent_size)
        self.fc_var = nn.Linear(d_model, latent_size)

        # Decoder
        self.decoder = Decoder(
            input_dim=latent_size,
            hidden_dim=hidden_size,
            output_dim=input_size,
            num_blocks=num_decoder_blocks
        )

        # Initialize latent space projections
        for fc in [self.fc_mu, self.fc_var]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(fc.bias, 0)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape

        # Create attention mask for padding
        if lengths is not None:
            mask = torch.arange(seq_len)[None, :] >= lengths[:, None]
            mask = mask.to(x.device)
        else:
            mask = None

        # Split time and features
        time = x[:, :, 0:1]
        features = x[:, :, 1:]

        # Apply positional encoding to time
        pos_encoded_time = self.pos_encoder(time)

        # Combine encoded time with features
        encoder_input = torch.cat([pos_encoded_time, features], dim=-1)

        # Encode
        encoded = self.encoder(encoder_input, mask)

        # Get sequence representation (use mean pooling over non-padded elements)
        # if lengths is not None:
        #     # Create mask for mean pooling
        #     mask_expanded = mask.unsqueeze(-1)
        #     # Mean over non-padded elements
        #     encoded_mean = (encoded * ~mask_expanded).sum(dim=1) / (~mask_expanded).sum(dim=1)
        #     z_seq = self.reparameterize(self.fc_mu(encoded), self.fc_var(encoded))
        # else:
        #     encoded_mean = encoded.mean(dim=1)

        z_seq = self.reparameterize(self.fc_mu(encoded), self.fc_var(encoded))

        # Project to latent space
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode
        decoded = self.decoder(z_seq)

        # Expand decoded output to match sequence length
        # decoded = decoded.unsqueeze(1).expand(-1, seq_len, -1)

        return decoded, mu, logvar

    def encode(self, x, lengths=None):
        """
        Encode input to latent space without sampling
        """
        batch_size, seq_len, _ = x.shape

        # Create attention mask for padding
        if lengths is not None:
            mask = torch.arange(seq_len)[None, :] >= lengths[:, None]
            mask = mask.to(x.device)
        else:
            mask = None

        # Split time and features
        time = x[:, :, 0:1]
        features = x[:, :, 1:]

        # Apply positional encoding to time
        pos_encoded_time = self.pos_encoder(time)

        # Combine encoded time with features
        encoder_input = torch.cat([pos_encoded_time, features], dim=-1)

        # Encode
        encoded = self.encoder(encoder_input, mask)

        # Get sequence representation
        # if lengths is not None:
        #     mask_expanded = mask.unsqueeze(-1).float()
        #     encoded_mean = (encoded * ~mask_expanded).sum(dim=1) / (~mask_expanded).sum(dim=1)
        # else:
        #     encoded_mean = encoded.mean(dim=1)

        z_seq = self.reparameterize(self.fc_mu(encoded), self.fc_var(encoded))

        # Project to latent space
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)

        return mu, logvar

    def decode(self, z, seq_len=None):
        """
        Decode latent vectors to output space
        """
        decoded = self.decoder(z)
        # if seq_len is not None:
        #     decoded = decoded.unsqueeze(1).expand(-1, seq_len, -1)
        return decoded

def collate_fn_err(batch):
    """Collate function for DataLoader that handles errors."""

    # time = [torch.tensor(lc[0]['TIME'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    # for lc in time:
    #     for i in range(len(lc)):
    #         lc[i] = lc[i] - lc[0] + 1
    #     print(lc)

    rate_low = [torch.tensor(lc[0]['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr_low = [torch.tensor(lc[0]['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr_low = [torch.tensor(lc[0]['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    rate_med = [torch.tensor(lc[1]['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr_med = [torch.tensor(lc[1]['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr_med = [torch.tensor(lc[1]['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    rate_hi = [torch.tensor(lc[2]['RATE'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    lowErr_hi = [torch.tensor(lc[2]['ERRM'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]
    upErr_hi = [torch.tensor(lc[2]['ERRP'].values.byteswap().newbyteorder(), dtype=torch.float32) for lc in batch]

    sequences = [torch.stack([rl, lel, uel, rm, lem, uem, rh, leh, ueh], dim=-1)
                for rl, lel, uel, rm, lem, uem, rh, leh, ueh in
                zip(rate_low, lowErr_low, upErr_low, rate_med, lowErr_med, upErr_med, rate_hi, lowErr_hi, upErr_hi)]
    # sequences = [torch.stack([t, rl, lel, uel, rm, lem, uem, rh, leh, ueh], dim=-1)
    #             for t, rl, lel, uel, rm, lem, uem, rh, leh, ueh in
                # zip(time, rate_low, lowErr_low, upErr_low, rate_med, lowErr_med, upErr_med, rate_hi, lowErr_hi, upErr_hi)]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)

    x = pad_sequence(sequences, batch_first=True)
    return x, lengths

def plot_reconstructions(model, test_loader, plot_dir, num_samples=25):
    """Plot original vs reconstructed light curves."""
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(5, 5, figsize=(15, 20))
        axes = axes.flatten()

        max_length = max(len(sample[1]['RATE']) for sample in test_loader.dataset)

        for i in range(num_samples):
            idx = random.randint(0, len(test_loader.dataset) - 1)
            sample = test_loader.dataset[idx]
            x, lengths = collate_fn_err([sample])
            x = x.to(device)
            lengths = lengths.to(torch.int64)

            x_hat, _, _ = model(x, lengths)
            x_hat_rate = x_hat[..., 0].cpu().numpy()

            axes[i].plot(x.cpu().numpy()[0, :, 0], label='Original RATE')
            axes[i].plot(x_hat_rate[0], label='Reconstructed RATE')
            axes[i].legend()
            axes[i].set_title(f'Light Curve {i+1}')
            axes[i].set_xlim(0, max_length)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"transformer_reconstructions.png"))
        plt.close()

def compute_losses(model, test_loader):
    """Compute reconstruction losses for all samples."""
    file_losses = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, lengths) in enumerate(test_loader):
            x = x.to(device)
            x_hat, mu, logvar = model(x, lengths)
            x_hat_rate = x_hat[..., 0]

            for i in range(len(lengths)):
                original = x[i, :lengths[i], 0]
                predicted = x_hat_rate[i, :lengths[i]]
                epsilon = 1e-8
                relative_error = torch.abs(original - predicted) / (torch.abs(original) + epsilon)
                sample_loss = torch.mean(relative_error).item()
                dataset_idx = batch_idx * test_loader.batch_size + i
                if dataset_idx >= len(test_loader.dataset):
                    continue
                file_losses.append((f"Sample_{dataset_idx}", sample_loss))

    return file_losses

if __name__ == '__main__':
    # Initialize wandb

    new_model_str = "140000_files_trans"
    learning_rate = 1e-5
    data_size = 100000
    num_epochs = 1500
    latent_size = 40
    d_model = 256
    nhead = 4
    num_encoder_layers = 2
    num_decoder_blocks = 3
    dropout = 0.1
    KLD_coef = 0.0035
    hidden_size = 256
    input_size = 9
    output_size = 1
    batch_size = 32


    wandb.init(
        project=new_model_str,
        config={
            "learning_rate": learning_rate,
            "architecture": "TransformerVAE",
            "dataset": "3 LCs",
            'data_size': data_size,
            "epochs": num_epochs,
            "latent_size": latent_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "hidden_size": hidden_size,
            "num_decoder_blocks": num_decoder_blocks,
            "dropout": dropout,
            "KLD_coef": KLD_coef
        }
    )

    # Define a dictionary with all hyperparameters
    hyperparams = {
        "learning_rate": learning_rate,
        "architecture": "TransformerVAE",
        "dataset": "3 LCs",
        'data_size': data_size,
        "epochs": num_epochs,
        "latent_size": latent_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "hidden_size": hidden_size,
        "num_decoder_blocks": num_decoder_blocks,
        "dropout": dropout,
        "KLD_coef": KLD_coef,
        "input_size": input_size,
        "output_size": output_size,
        "batch_size": batch_size
    }

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



    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load data
    print('Loading light curves...')
    fits_files = load_all_fits_files()
    print('finished getting all files')
    lc_low, lc_med, lc_high = load_n_light_curves(data_size, fits_files, band="all")
    # lc_low, lc_med, lc_high = load_n_light_curves(10, fits_files, band="all")
    light_curves_sample = list(zip(lc_low, lc_med, lc_high))
    print(f'Loaded {len(light_curves_sample)} light curves')

    # Split data
    train_set, val_set, test_set = partition_data(light_curves_sample)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=32, collate_fn=collate_fn_err, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, collate_fn=collate_fn_err)
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn_err)

    # Initialize model
    model = TransformerVAE(
        input_size=input_size,  # 1 time +  3 bands x 3 values (RATE, ERRM, ERRP)
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        latent_size=latent_size,
        hidden_size=hidden_size,
        num_decoder_blocks=num_decoder_blocks,
        dropout=dropout
    ).to(device)

    # Training parameters
    num_epochs = num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Create directories for saving
    model_dir = "./models"
    plot_dir = "./plots/Transformer plots/" + new_model_str
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Make directories for saving model and plots
    save_dir = "/home/pdong/Astro UROP/models/Transformer Models"
    os.makedirs(save_dir, exist_ok=True)

    plot_dir = "/home/pdong/Astro UROP/training_plots/Transformer Models/" + new_model_str
    os.makedirs(plot_dir, exist_ok=True)

    config_path = os.path.join(plot_dir, "config.json")
    save_config(hyperparams, config_path)

    # Select a benchmark sample for consistent visualization
    benchmark_idx = random.randint(0, len(train_set) - 1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, lengths) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, lengths)
            loss = ELBO(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # if batch_idx % 100 == 0:
            #     print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, lengths in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x, lengths)
                val_loss += ELBO(x_hat, x, mu, logvar).item()

        val_loss /= len(val_loader)

        if epoch % 50 == 0:
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            # Generate and log reconstruction plot
            benchmark_sample = train_set[benchmark_idx]
            x_benchmark, lengths_benchmark = collate_fn_err([benchmark_sample])
            x_benchmark = x_benchmark.to(device)
            lengths_benchmark = lengths_benchmark.to(torch.int64)

            with torch.no_grad():
                x_hat_benchmark, _, _ = model(x_benchmark, lengths_benchmark)

            x_hat_benchmark_rate = x_hat_benchmark[..., 0].cpu().numpy()
            x_original_rate = x_benchmark[..., 0].cpu().numpy()

            # Create reconstruction plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_original_rate[0], label="Original RATE", color="blue", linestyle="dashed")
            ax.plot(x_hat_benchmark_rate[0], label="Reconstructed RATE", color="red", alpha=0.7)
            ax.legend()
            ax.set_title(f"Benchmark Reconstruction at Epoch {epoch + 1}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Rate")

            # Log plot to wandb
            wandb.log({"Reconstruction": wandb.Image(fig)})
            plt.close(fig)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, new_model_str + '.h5'))

    # Final evaluation
    print('Computing final losses and generating plots...')
    file_losses = compute_losses(model, test_loader)

    # Plot final reconstructions
    plot_reconstructions(model, test_loader, plot_dir)

    # Save losses
    sorted_losses = sorted(file_losses, key=lambda x: x[1], reverse=True)
    loss_file_path = os.path.join(plot_dir, "transformer_reconstruction_losses.txt")
    with open(loss_file_path, 'w') as f:
        f.write("File Name\tLoss\n")
        for file_name, loss in sorted_losses:
            f.write(f"{file_name}\t{loss:.6f}\n")

    # Finish wandb run
    wandb.finish()
    print(f'Training completed. Model at {model_dir + new_model_str} saved and evaluation plots generated at {plot_dir + new_model_str}.')
