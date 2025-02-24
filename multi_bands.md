# Multi-Band Light Curve Training Pipeline

## Model Architecture Overview

The training pipeline implements a Recurrent Variational Autoencoder (RNN-VAE) architecture specifically designed for processing multi-band astronomical light curves from eROSITA.

### Neural Network Components

1. **Encoder (RNN-based)**
   - Input: Light curves from 3 energy bands (0.2-0.6, 0.6-2.3, 2.3-5.0 keV)
   - Architecture:
     - GRU layers with configurable hidden size
     - Dropout for regularization
     - Processes variable-length sequences using packed sequences
   - Output: Hidden state representations

2. **Latent Space**
   - Dimensionality: 40 (configured in wandb)
   - Implements VAE reparameterization trick
   - Two linear layers for:
     - Mean (mu) computation
     - Log-variance (logvar) computation

3. **Decoder (RNN-based)**
   - Input: Sampled latent vectors
   - Architecture:
     - GRU layers matching encoder
     - Final linear layer for reconstruction
     - Exponential activation for positive Poisson rates
   - Output: Reconstructed light curves

## Training Pipeline

### 1. Data Preparation
```python
fits_files = load_all_fits_files()
lc_low, lc_med, lc_high = load_n_light_curves(16384, fits_files, band="all")
light_curves_sample = list(zip(lc_low, lc_med, lc_high))
```

### 2. Data Partitioning
- Training set: 85% of data
- Validation set: 5% of data
- Test set: 10% of data
```python
train_dataset, val_dataset, test_dataset = partition_data(
    light_curves_sample,
    test_size=0.1,
    val_size=0.05
)
```

### 3. Data Loading
- Batch size: 32
- Custom collate function for handling variable-length sequences
- Includes error measurements in the data loading process
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=collate_fn_err_mult,
    shuffle=True
)
```

### 4. Loss Function: ELBO (Evidence Lower BOund)

The model uses the ELBO loss function, which consists of two components:
- **Reconstruction Loss**: Mean Squared Error (MSE) between input and reconstructed output
  ```python
  MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)
  ```
- **KL Divergence**: Regularizes the latent space distribution
  ```python
  KLD = -0.001 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  ```
- Total loss is the sum of MSE and KLD
- KL coefficient: 0.001 (hardcoded in ELBO function)

### 5. Training Configuration
```python
config = {
    "learning_rate": 1e-5,
    "architecture": "VAE",
    "dataset": "16384 LCs",
    "epochs": 12000,
    "latent_size": 40,
    "KLD_coef": 0.0035
}
```

### 6. Model Parameters
- Input size: 9 (three energy bands × 3 values per band: rate, lower error, upper error)
- Hidden size: 64 (encoder and decoder)
- Latent size: 40 (dimension of the latent space)
- Number of GRU layers: 2 (both in encoder and decoder)
- Dropout rate: 0.2 (for regularization)

### 7. Training Process
- Uses CUDA if available (automatically detected)
- Saves model checkpoints in .h5 format
- Logs metrics using Weights & Biases (wandb) including:
  - Training loss
  - Model parameters
  - Training progress
  - Benchmark graphs every 50 epochs
- Generates training plots in `/home/pdong/Astro UROP/training_plots/`

### 8. Output and Visualization
- Model saves:
  - Checkpoints stored as .h5 files
  - Training plots saved to `/home/pdong/Astro UROP/training_plots/`
- Wandb logging:
  - Training loss
  - Model parameters
  - Training progress
  - Benchmark graphs at each 50 epochs

## Detailed Technical Implementation

1. **Variable Length Processing**
   - Implements PyTorch's `pack_padded_sequence` and `pad_packed_sequence` in both encoder and decoder
   - GRU layers process sequences of varying lengths efficiently by:
     ```python
     packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
     packed_output, hidden = self.gru(packed_x)
     output, _ = pad_packed_sequence(packed_output, batch_first=True)
     ```
   - Maintains temporal information through the GRU architecture

2. **Multi-band Integration**
   - Processes all three energy bands (0.2-0.6, 0.6-2.3, 2.3-5.0 keV) simultaneously
   - Custom collate function `collate_fn_err_mult` combines data from all bands:
     ```python
     rate_low, lowErr_low, upErr_low  # Low band data
     rate_med, lowErr_med, upErr_med  # Medium band data
     rate_hi, lowErr_hi, upErr_hi     # High band data
     ```
   - Preserves correlations between bands by processing them together in the same network

3. **Error Handling**
   - Incorporates asymmetric error measurements (ERRM and ERRP) for each band
   - Custom collate function processes both lower and upper error bounds
   - Error measurements are included in the input tensor, allowing the model to learn uncertainty-aware representations
   - Input features per time step: [rate_low, errm_low, errp_low, rate_med, errm_med, errp_med, rate_hi, errm_hi, errp_hi]

4. **Monitoring and Visualization**
   - Real-time training monitoring through Weights & Biases:
     - Project name: 'allbands_16384_11k'
     - Tracks learning rate, architecture, dataset size, epochs
     - Records loss metrics and model performance
   - Generates benchmark visualizations every 50 epochs
   - Saves training plots for analysis and debugging
