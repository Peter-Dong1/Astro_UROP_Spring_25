# eROSITA Light Curve Analysis Project

## Overview

This project focuses on analyzing light curves from the eROSITA X-ray telescope, employing various machine learning techniques to extract features, detect anomalies, and classify astronomical sources. The analysis pipeline includes data preprocessing, feature extraction, dimensionality reduction, and both supervised and unsupervised learning approaches.

## Data Source

- **Source**: eROSITA X-ray telescope data from eRASS1 (first all-sky survey)
- **Format**: FITS files containing light curve data
- **Location**: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`
- **Energy Bands**:
  - Low: 0.2-0.6 keV
  - Medium: 0.6-2.3 keV
  - High: 2.3-5.0 keV

## Core Components

### Data Processing

- **`helper.py`**: Core utility functions for loading and processing light curve data
  - `load_light_curve()`: Loads individual light curves from FITS files
  - `load_n_light_curves()`: Loads multiple light curves across different energy bands
  - `partition_data()`: Splits data into training, validation, and test sets

- **`light_curves.py`**: Visualization and analysis of raw light curve data
  - Functions for plotting single and multiple light curves
  - Comparison of light curves across different energy bands
  - Grid visualization of multiple light curves

### Feature Extraction

- **`feature_extraction.py`**: Statistical feature extraction and anomaly detection
  - Extracts features like weighted mean, variance, median, IQR, etc.
  - Implements outlier detection using multiple methods:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - IQR-based detection
    - Z-score-based detection
  - Clustering methods:
    - HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    - UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction

### Deep Learning Models

#### Transformer-based Variational Autoencoder

- **`test_trans.py`**: Implementation and evaluation of a Transformer-based VAE
  - Processes multi-band light curve data (9 features per time step)
  - Encodes light curves into a latent space representation
  - Reconstructs light curves from the latent space
  - Provides visualization of the latent space using PCA and t-SNE
  - Implements outlier detection in the latent space using Isolation Forest

#### RNN-based Variational Autoencoder

- **`test_train.py`**: Training and evaluation of an RNN-based VAE
  - Processes multi-band light curve data
  - Uses Weights & Biases (wandb) for experiment tracking
  - Implements custom loss functions (ELBO, Poisson NLL)
  - Provides visualization of reconstruction quality

## Project Structure

```
/home/pdong/Astro UROP/
├── Markdown Files/        # Documentation
├── models/                # Saved model weights
│   ├── RNN Models/        # RNN-based VAE models
│   └── Transformer Models/# Transformer-based VAE models
├── plots/                 # Visualization outputs
│   ├── RNN plots/         # Plots from RNN models
│   ├── Transformer plots/ # Plots from Transformer models
│   └── feature_extraction_plots/ # Clustering and feature analysis plots
├── helper.py              # Core utility functions
├── light_curves.py        # Light curve visualization
├── feature_extraction.py  # Feature extraction and clustering
├── test_trans.py          # Transformer VAE implementation
└── test_train.py          # RNN VAE training
```

## Key Techniques

1. **Data Preprocessing**:
   - Handling asymmetric errors in light curve measurements
   - Normalization and scaling of features
   - Sequence padding for variable-length light curves

2. **Feature Engineering**:
   - Statistical features (weighted mean, variance, median, etc.)
   - Error-aware feature extraction
   - Dimensionality reduction (PCA, UMAP)

3. **Deep Learning**:
   - Variational Autoencoders (VAEs) for unsupervised learning
   - Transformer architecture for sequence modeling
   - RNN-based sequence modeling
   - Custom loss functions for probabilistic modeling

4. **Anomaly Detection**:
   - Isolation Forest for outlier detection
   - Local Outlier Factor for density-based outlier detection
   - Latent space anomaly detection

5. **Clustering**:
   - HDBSCAN for density-based clustering
   - DBSCAN with UMAP embeddings

## Visualization

The project includes extensive visualization capabilities:
- Light curve plots with error bars
- Comparison of light curves across energy bands
- Latent space visualization using PCA and t-SNE
- Cluster visualization
- Reconstruction quality assessment
- Grid plots of outliers and cluster representatives

## Current Status

The project has implemented:
1. Data loading and preprocessing pipeline
2. Feature extraction from light curves
3. Multiple clustering and anomaly detection methods
4. Two deep learning architectures (Transformer VAE and RNN VAE)
5. Comprehensive visualization tools

## Usage

The main workflows are:

1. **Feature-based analysis**:
   ```python
   from feature_extraction import run_hdbscan_clustering

   # Run HDBSCAN clustering on light curves
   cluster_labels, feature_matrix, pca_result = run_hdbscan_clustering(
       light_curves, min_cluster_size=5, min_samples=None
   )
   ```

2. **Deep learning analysis**:
   ```python
   # Load a trained Transformer VAE model
   model = TransformerVAE(
       input_size=9,
       d_model=256,
       nhead=4,
       num_encoder_layers=2,
       latent_size=40,
       hidden_size=256,
       num_decoder_blocks=3,
       dropout=0.05
   ).to(device)

   model.load_state_dict(torch.load('./models/trans_100k_1500.h5'))

   # Extract latent representations and detect outliers
   outlier_indices, latent_vectors = detect_outliers_isolation_forest(
       model, test_loader, plot_dir
   )
   ```
