# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for analyzing X-ray light curves from the eROSITA telescope (eRASS1 survey). The project uses machine learning and deep learning techniques to extract features, detect anomalies, and classify astronomical sources. The pipeline includes traditional statistical feature extraction and two variational autoencoder architectures (RNN-based and Transformer-based).

## Data Source and Structure

**Data Location**: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`

**Energy Bands**:
- Low: 0.2-0.6 keV (band=0)
- Medium: 0.6-2.3 keV (band=1)
- High: 2.3-5.0 keV (band=2)

**Data Format**: FITS files containing:
- `TIME`: Observation timestamps
- `TIMEDEL`: Time delta
- `RATE`: Light curve intensity (3 bands)
- `RATE_ERRM`: Negative error measurements (3 bands)
- `RATE_ERRP`: Positive error measurements (3 bands)

Light curves are truncated to 20 data points maximum by default.

## Running on SLURM Cluster

All SLURM scripts use the `myenv` conda environment. Jobs are submitted via:

```bash
sbatch <script>.slurm
```

**Deep learning SLURM scripts** (in repo root):
- `rnn.slurm` - Train RNN VAE model
- `trans.slurm` - Train Transformer VAE model
- `plot_rnn.slurm` - Generate RNN model visualizations

**Feature extraction SLURM scripts** (all in `pipeline/`):
- `get_features.slurm` - Path A single-node extraction (not currently used — missing `ampl_sig`)
- `split_curves.slurm` - Path B step 1: partition all light curves
- `run_chunks.slurm` - Path B step 2: array job (0–28), extract features per partition
- `consol_feat.slurm` - Path B step 3: merge per-curve pickles
- `append_feat.slurm` - Path B step 4: add `ampl_sig` feature
- `analyze_features.slurm` - Final analysis (both paths)

**SLURM Configuration**:
- Partition: `mit_normal_gpu` (GPU jobs) or `mit_normal` (CPU feature extraction)
- Standard location: `/home/pdong/Astro\ UROP/`
- Logs stored in `logs/` directory

## Key Architecture Components

### Data Loading Pipeline (`helper.py`)

Core functions for loading FITS files:
- `load_light_curve(file_path, band=1, trunc=20)` - Load single light curve
- `load_n_light_curves(n, fits_files, band='all', trunc=20)` - Load multiple light curves
- `load_all_fits_files(data_dir)` - Get all FITS file paths
- `partition_data(light_curves, test_size=0.2, val_size=0.1)` - Split train/val/test sets

Returns pandas DataFrames with columns: `TIME`, `TIMEDEL`, `RATE`, `ERRM`, `ERRP`, `SYM_ERR`

**Important**: Some light curves are inaccessible or empty. The file `inaccessible_lightcurves.txt` tracks these and they are filtered during loading.

### Feature Extraction Pipeline

**Root Directory Pipeline** (`feature_extraction.py`):
- Statistical feature extraction using weighted means/variance
- Outlier detection: Isolation Forest, Local Outlier Factor, IQR, Z-score
- Clustering: HDBSCAN with UMAP dimensionality reduction
- Uses `light_curve` package for advanced features

**New Pipeline** (`pipeline/`) — **currently in use**:
- `config.py` - Configuration with paths, parameters, and `SELECTED_FEATURES_FOR_CLUSTERING`
- `helper.py` - Pipeline-specific FITS loader (adds FRACEXP, COUNTS, BACKCOUNTS, BACKRATIO)
- `run_feature_extraction.py` - Path A single-node extraction (**10 features, not current**)
- `batch_feature_extraction.py` - Path B per-partition extraction (10 features)
- `split_curves.py` - Path B step 1: split all curves into 28 partitions
- `consolidate_features.py` - Path B step 3: merge per-curve pickles
- `append_new_features.py` - Path B step 4: add `ampl_sig` (makes feature set complete)
- `run_pipeline_on_features.py` - Full analysis pipeline with HDBSCAN/UMAP clustering
- `bexvar_ero.py` - Bayesian excess variance via ultranest nested sampling

**Two execution paths**:
- **Path B (current)**: split → batch extract → consolidate → append `ampl_sig` → analyze. Produces **11 features**. Required because `ampl_sig` is in `SELECTED_FEATURES_FOR_CLUSTERING`.
- **Path A (not used)**: single-node extraction. Only produces **10 features** — missing `ampl_sig`, so output is incomplete for clustering.

**All 11 features extracted**:
- `weighted_mean`, `weighted_variance` — error-weighted statistics
- `lag1_autocorr` — lag-1 autocorrelation
- `hurst_exp` — Hurst exponent (persistence vs. mean-reversion)
- `mean_rise_fall_ratio` — rises/falls ratio
- `beyond1std`, `stetson_k`, `mean_var` — from `light_curve` package
- `excess_var` — normalized excess variance (NEV)
- `bexvar` — Bayesian excess variance (nested sampling, slow)
- `ampl_sig` — amplitude significance (**Path B only**, added by `append_new_features.py`)

**9 features used for HDBSCAN clustering** (`SELECTED_FEATURES_FOR_CLUSTERING` in `config.py`):
`weighted_mean`, `weighted_variance`, `lag1_autocorr`, `hurst_exp`, `mean_rise_fall_ratio`, `stetson_k`, `bexvar`, `mean_var`, `ampl_sig`

Features are saved as pickled DataFrames in `data/all/amp_max_features/features.pkl` (= `FEATURES_FILE`).

### Deep Learning Models

**RNN-Based VAE** (`RNN_9_model.py`, `RNN_train.py`, `train_rnn.py`):
- Encoder/Decoder using GRU layers
- Input: 9 features per timestep (3 bands × 3 values: RATE, ERRM, ERRP)
- Uses packed sequences to handle variable-length inputs
- Loss functions: ELBO (Evidence Lower Bound) or Poisson NLL
- Experiment tracking via Weights & Biases (wandb)

**Transformer-Based VAE** (`trans_model.py`, `test_trans.py`):
- Transformer encoder with multi-head attention
- Positional encoding for temporal information
- ResNet blocks in decoder
- Input: Same 9-feature format as RNN
- Latent space visualization via PCA and t-SNE
- Outlier detection in latent space using Isolation Forest

Both models save checkpoints to `models/` directory and plots to `plots/`.

## Common Development Commands

### Loading Data
```python
from helper import load_all_fits_files, load_n_light_curves

# Load all FITS file paths
fits_files = load_all_fits_files()

# Load 1000 light curves from medium energy band
light_curves = load_n_light_curves(1000, fits_files, band='med', trunc=20)

# Load all three bands
lc_low, lc_med, lc_high = load_n_light_curves(1000, fits_files, band='all')
```

### Feature Extraction
```python
from feature_extraction import run_hdbscan_clustering

# Run clustering pipeline
cluster_labels, feature_matrix, pca_result = run_hdbscan_clustering(
    light_curves,
    min_cluster_size=5,
    min_samples=None
)
```

### Training Models
```python
# RNN VAE training is in train_rnn.py
# Transformer VAE training is in test_trans.py
# Both use DataLoader with custom Dataset classes for handling variable-length sequences
```

### Using New Feature Pipeline (Path B — current workflow)
```bash
# Step 1: Split all light curves into 28 partitions
sbatch "pipeline/split_curves.slurm"

# Step 2: Extract features for each partition (array job 0–28)
sbatch "pipeline/run_chunks.slurm"

# Step 3: Merge per-curve pickles into one DataFrame
sbatch "pipeline/consol_feat.slurm"

# Step 4: Append ampl_sig feature (required for clustering)
sbatch "pipeline/append_feat.slurm"

# Step 5: Run clustering, outlier detection, and all visualizations
sbatch "pipeline/analyze_features.slurm"
```

## Directory Structure

```
/
├── helper.py                          # Core data loading utilities
├── light_curves.py                    # Light curve visualization
├── feature_extraction.py              # Statistical features & clustering
├── RNN_9_model.py                     # RNN VAE model definition
├── RNN_train.py / train_rnn.py        # RNN training scripts
├── trans_model.py                     # Transformer VAE model
├── test_trans.py                      # Transformer training/evaluation
├── test_rnn.py                        # RNN evaluation
├── plotmodel.py / plotmodelerror.py   # Visualization utilities
├── *.slurm                            # Deep learning SLURM scripts (rnn, trans, plot_rnn)
├── Markdown Files/                    # Documentation
│   ├── README.md                      # Project overview
│   └── Data.md                        # Data structure documentation
├── models/                            # Saved model checkpoints
│   ├── RNN Models/
│   └── Transformer Models/
├── plots/                             # All visualizations
│   ├── RNN plots/
│   ├── Transformer plots/
│   └── feature_extraction_plots/
└── pipeline/ # New modular pipeline (current)
    ├── config.py                      # Configuration (paths, SELECTED_FEATURES_FOR_CLUSTERING)
    ├── helper.py                      # Pipeline-specific FITS loader
    ├── run_feature_extraction.py      # Path A: single-node (10 features, not current)
    ├── batch_feature_extraction.py    # Path B step 2: per-partition extraction
    ├── split_curves.py                # Path B step 1: partition all curves
    ├── consolidate_features.py        # Path B step 3: merge pickles
    ├── append_new_features.py         # Path B step 4: add ampl_sig
    ├── run_pipeline_on_features.py    # Final analysis (clustering, plots)
    ├── bexvar_ero.py                  # Bayesian excess variance (ultranest)
    ├── *.slurm                        # All feature extraction SLURM scripts
    └── data/all/amp_max_features/     # Final features.pkl (FEATURES_FILE)
```

## Important Notes

- **Two helper.py files**: One in root, one in `pipeline/`. The pipeline version has additional utilities for batch processing.
- **Hardcoded paths**: Some scripts reference `/home/pdong/Astro\ UROP/` - adjust when running in different environments.
- **GPU required**: Deep learning models require CUDA-enabled GPU.
- **Conda environment**: Activate `myenv` before running scripts.
- **Variable-length sequences**: All models handle variable-length light curves via padding/masking.
- **Known interesting sources**: Three specific light curves are flagged in config files for special analysis.

## Data Processing Notes

- Light curves with missing data or inaccessible files are tracked in `inaccessible_lightcurves.txt`
- Asymmetric errors (ERRM/ERRP) are handled; symmetric approximation available as `SYM_ERR = (ERRM + ERRP)/2`
- Feature extraction uses weighted statistics to account for measurement uncertainties
- HDBSCAN clustering is density-based and produces noise labels (-1) for outliers
