# eROSITA Light Curve Analysis - Project Status

**Last Updated**: 2026-03-06
**Project**: eROSITA X-ray Light Curve Event Detection & Classification
**Data Source**: eRASS1 (eROSITA All-Sky Survey 1) light curves

---

## Executive Summary

This project explored multiple approaches for detecting anomalous events and classifying X-ray light curves from the eROSITA telescope. After initial experiments with deep learning approaches (RNNs, Transformers, LSTM autoencoders), the main work focused on **statistical feature extraction combined with unsupervised clustering** using HDBSCAN, UMAP, and Isolation Forest.

**Key Results**:
- Extracted statistical features from light curves across 3 energy bands
- Performed HDBSCAN clustering to identify similar light curve behaviors
- Computed cosine similarity between sources to find related objects
- Generated cluster visualizations and feature histograms
- Created web-ready HDF5 outputs for interactive visualization

---

## Project Timeline & Approaches

### Phase 1: Deep Learning Experiments (Root Directory - Early Work)

**Status**: Exploratory, not used in final analysis

**Notebooks**:
- `.ipynb_checkpoints/LSTM_AutoEncoder-checkpoint.ipynb` - LSTM-based variational autoencoder experiments
- `.ipynb_checkpoints/Raw Data Clustering-checkpoint.ipynb` - Initial clustering on raw light curve data
- `.ipynb_checkpoints/Statistical Clustering-checkpoint.ipynb` - Statistical approach experiments
- `.ipynb_checkpoints/ML Approach Standard-checkpoint.ipynb` - Standard ML methods
- `.ipynb_checkpoints/light_curves-checkpoint.ipynb` - Light curve visualization and exploration

**Python Scripts** (Root):
- `RNN_9_model.py`, `RNN_train.py`, `train_rnn.py` - RNN-based VAE implementation
  - Uses GRU encoder/decoder
  - 9-feature input (3 bands × 3 values: RATE, ERRM, ERRP)
  - Loss: ELBO or Poisson NLL
  - Weights & Biases (wandb) integration

- `trans_model.py`, `test_trans.py` - Transformer-based VAE
  - Multi-head attention encoder
  - Positional encoding for time
  - ResNet decoder blocks
  - Latent space outlier detection (Isolation Forest)

- `feature_extraction.py` - Early feature extraction pipeline
- `light_curves.py` - Visualization utilities
- `plotmodel.py`, `plotmodelerror.py` - Model evaluation plots
- `test_model_9.py`, `test_rnn.py` - Testing scripts

**SLURM Scripts** (for cluster execution):
- `rnn.slurm` - RNN VAE training
- `trans.slurm` - Transformer VAE training
- `feature.slurm` - Feature extraction
- `helper.slurm` - Helper jobs
- `plot_rnn.slurm` - Plot generation

**Findings**: Deep learning approaches were exploratory but not the final method used.

---

### Phase 2: Feature-Based Analysis (z New Feature Extraction Pipeline/ - Main Work)

**Status**: Primary analysis pipeline - ACTIVE RESULTS

This is where the **actual experiments and results** live.

#### Core Pipeline Scripts

1. **`run_feature_extraction.py`** - Main feature extraction
   - Parallel processing support (`--job-id`, `--num-jobs`)
   - Extracts statistical features from FITS light curves
   - Saves features as pickled DataFrames

2. **`batch_feature_extraction.py`** - Batch processing
   - Processes light curves in chunks
   - Handles parallel SLURM array jobs

3. **`consolidate_features.py`** - Merge parallel results
   - Combines features from multiple jobs
   - Creates unified feature DataFrame

4. **`run_pipeline_on_features.py`** - **MAIN ANALYSIS SCRIPT**
   - **HDBSCAN clustering** with configurable parameters
   - **UMAP dimensionality reduction** for visualization
   - **Isolation Forest** outlier detection
   - Saves cluster assignments to CSV
   - Generates visualizations
   - Exports HDF5 files for web viewing

5. **`analyze_similar_curves.py`** - **COSINE SIMILARITY ANALYSIS**
   - Finds top 100 most similar curves for known sources
   - Computes cosine similarity between feature vectors
   - Generates feature histograms
   - Tracks cluster assignments for similar sources
   - **This likely contains the "lists of sources with cosine similarity" your supervisors want**

#### Supporting Scripts

- **`config.py`** - Configuration file
  - Data paths
  - Feature selection for clustering
  - Known interesting light curves
  - HDBSCAN/UMAP parameters

- **`helper.py`** - Pipeline utilities
  - Data loading functions
  - File path resolution
  - Index mapping for fast lookups

- **`bexvar_ero.py`** - Bayesian excess variance calculation
  - Computes bexvar statistic for variability
  - eROSITA-specific implementation

- **`bexvar_histograms.py`** - Histogram generation for bexvar

- **`check_nans.py`** - Data quality checks

- **`inspect_features.py`** - Feature DataFrame inspection

- **`plot_cluster_samples.py`** - Visualize sample light curves from clusters
  - Takes cluster ID as input
  - Plots N sample light curves
  - Saves to PNG

- **`sample_clusters.py`** - Extract cluster samples

- **`split_curves.py`** - Split light curves into time segments

- **`append_new_features.py`** - Add new features to existing DataFrame

#### SLURM Scripts (Pipeline)

- `analyze_features.slurm` - Run analysis pipeline
- `append_feat.slurm` - Append features
- `batch_get_features.slurm` - Batch feature extraction
- `consol_feat.slurm` - Consolidate features
- `get_features.slurm` - Get features
- `run_chunks.slurm` - Run chunked processing
- `split_curves.slurm` - Split curves

---

## Extracted Features

The pipeline extracts these statistical features from each light curve:

**Selected for Clustering** (from `config.py`):
- `weighted_mean` - Mean flux weighted by errors
- `weighted_variance` - Variance weighted by errors
- `lag1_autocorr` - Lag-1 autocorrelation (temporal correlation)
- `hurst_exp` - Hurst exponent (persistence vs mean-reversion)
- `mean_rise_fall_ratio` - Ratio of rising to falling segments
- `stetson_k` - Stetson K statistic (variability measure)
- `bexvar` - Bayesian excess variance
- `mean_var` - Mean variance
- `ampl_sig` - Amplitude significance

**Additional Features** (likely extracted but not used for clustering):
- `weighted_median`
- `weighted_iqr`
- `beyond1std` - Fraction of points beyond 1 standard deviation
- `excess_var` - Excess variance

---

## Key Outputs & Results

### 1. Feature Histograms (`640features/`)

Contains histograms for all extracted features:
- `bexvar_hist.png`
- `beyond1std_hist.png`
- `excess_var_hist.png`
- `hurst_exp_hist.png`
- `lag1_autocorr_hist.png`
- `mean_rise_fall_ratio_hist.png`
- `mean_var_hist.png`
- `stetson_k_hist.png`
- `weighted_mean_hist.png`
- `weighted_variance_hist.png`

### 2. Cluster Analysis Results (Expected Locations)

Based on code analysis, results are saved in structure like:
```
z New Feature Extraction Pipeline/data/all/{run_number}/
├── hdbscan_data/
│   ├── cluster_assignments.csv          # Cluster labels per source
│   ├── cluster_probabilities.csv        # Membership probabilities
│   └── outlier_scores.csv              # HDBSCAN outlier scores
├── umap_data/
│   ├── umap_embedding.csv              # 2D UMAP coordinates
│   └── cluster_assignments.csv         # Clusters with UMAP coords
├── web_data/
│   ├── features.h5                     # HDF5 features for web
│   └── light_curves.h5                 # HDF5 light curves for web
└── sample_cluster_plots/               # Grid plots of cluster samples
    └── cluster_{id}_samples_{timestamp}.png
```

Also:
```
z New Feature Extraction Pipeline/data/all/analysis_results/
├── cluster_assignments_237_real_clippedxvar/    # Cluster assignments for run 237
├── feature_histograms_237_real_clippedxvar/     # Feature histograms
└── sig_nev_analysis/                            # SIG/NEV source analysis
```

### 3. Cosine Similarity Results

**Key Output**: `analyze_similar_curves.py` generates:
- Top 100 most similar curves for each known source
- Cluster assignments for similar sources
- **This is what your supervisors asked about: "lists of sources with cosine similarity"**

Expected outputs:
- `SIG_NEV_mappings.pkl` - Mappings between SIG and NEV identifiers
- CSV files with source names and similarity scores
- Plots showing similar curves

### 4. Known Interesting Sources

From `config.py`, these are the specific sources tracked:
```python
KNOWN_LIGHT_CURVES = [
    "em01_211120_020_LightCurve_00007_c010_rebinned.fits",
    "em01_039135_020_LightCurve_00058_c010_rebinned.fits",
    "em01_038099_020_LightCurve_00005_c010_rebinned.fits"
]
```

### 5. Features File

Main features file (from `config.py`):
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl
```

This is a pickled pandas DataFrame with all extracted features.

---

## Clustering Methods Used

### HDBSCAN (Primary Method)

**Configuration** (from `run_pipeline_on_features.py`):
- `min_cluster_size = 24` (minimum points per cluster)
- `min_samples = 12` (minimum samples for core points)
- `cluster_selection_epsilon = 0.11`
- `cluster_selection_method = 'leaf'`

**Purpose**: Density-based clustering that can find clusters of varying shapes and sizes, and identifies noise points.

### UMAP (Dimensionality Reduction)

**Configuration** (from `config.py`):
- `n_neighbors = 15`
- `min_dist = 0.1`
- `n_components = 2` (for visualization)

**Purpose**: Reduce high-dimensional feature space to 2D for visualization while preserving local and global structure.

### Isolation Forest (Outlier Detection)

**Configuration**:
- `contamination = 0.05` (expected fraction of outliers)

**Purpose**: Identify anomalous light curves that don't fit typical patterns.

---

## Data Structure

**Source Data**:
- Location: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`
- Format: FITS files
- Energy bands: Low (0.2-0.6 keV), Medium (0.6-2.3 keV), High (2.3-5.0 keV)
- Light curves truncated to 20 time points maximum

**Data Loading**:
- `helper.py` (root and pipeline versions) handles FITS loading
- Inaccessible/empty light curves tracked in `inaccessible_lightcurves.txt`
- DataFrames include: `TIME`, `RATE`, `ERRM`, `ERRP`, `SYM_ERR`

---

## What Your Supervisors Requested

Based on: *"Dan and I were wondering if you could share with us the code/results repositories, and other useful files (I am interesting in these lists of sources with cosine similarity)"*

### 1. **Code Repository**
✅ Entire `z New Feature Extraction Pipeline/` directory

**Key files to highlight**:
- `run_pipeline_on_features.py` - Main analysis
- `analyze_similar_curves.py` - Cosine similarity analysis
- `config.py` - All parameters
- `run_feature_extraction.py` - Feature extraction

### 2. **Results Repository**

**Priority outputs to share**:
- Feature histograms: `640features/*.png`
- Cluster assignments CSVs (need to locate on cluster)
- UMAP embeddings (need to locate on cluster)
- HDF5 web files (if generated)

### 3. **Lists of Sources with Cosine Similarity** ⭐

**Most important**: Outputs from `analyze_similar_curves.py`

This script:
- Loads features from `features.pkl`
- Computes cosine similarity between all sources
- For each known source, finds top 100 most similar
- Saves results with cluster labels

**Action needed**: Locate and organize these outputs from cluster runs.

---

## Outstanding Tasks

### 1. Locate Actual Results on Cluster
Results are likely on the SLURM cluster at:
- `/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/`
- `/home/pdong/Astro UROP/z New Feature Extraction Pipeline/plots/`

**Need to**:
- Download all result CSVs
- Download cluster assignment files
- Download similarity analysis outputs
- Download feature histograms (beyond the 10 in `640features/`)

### 2. Generate Missing Outputs
If not already done:
- Run `analyze_similar_curves.py` for all known sources
- Export final cluster assignments to clean CSVs
- Create summary statistics

### 3. Clean Documentation
- Create README for results directory
- Document what each CSV contains
- Create data dictionary for features

### 4. Organize for Handoff
See `ORGANIZATION_PLAN.md` (to be created next)

---

## Repository Structure Summary

```
Astro_UROP_Spring_25/
│
├── [ROOT] - Old/Exploratory Work (Deep Learning Experiments)
│   ├── RNN VAE models (RNN_9_model.py, train_rnn.py, etc.)
│   ├── Transformer VAE models (trans_model.py, test_trans.py)
│   ├── Early feature extraction (feature_extraction.py)
│   ├── Notebooks (.ipynb_checkpoints/)
│   ├── SLURM scripts (*.slurm)
│   ├── helper.py - Core data loading
│   └── light_curves.py - Visualization
│
├── z New Feature Extraction Pipeline/ - MAIN WORK ⭐
│   ├── config.py - Configuration
│   ├── run_feature_extraction.py - Extract features
│   ├── batch_feature_extraction.py - Batch processing
│   ├── consolidate_features.py - Merge results
│   ├── run_pipeline_on_features.py - Main analysis (HDBSCAN/UMAP)
│   ├── analyze_similar_curves.py - Cosine similarity ⭐
│   ├── plot_cluster_samples.py - Visualizations
│   ├── bexvar_ero.py - Bayesian variance
│   ├── helper.py - Pipeline utilities
│   ├── *.slurm - SLURM job scripts
│   ├── 640features/ - Feature histograms
│   ├── libs/bexvar/ - External library
│   └── data/all/ - Results (ON CLUSTER)
│
├── Markdown Files/ - Documentation
│   ├── README.md - Project overview
│   ├── Data.md - Data structure
│   └── multi_bands.md
│
├── CLAUDE.md - Guide for Claude Code
├── PROJECT_STATUS.md - This file
└── .gitignore
```

---

## Next Steps

1. **Immediate**: Create organization plan (see `ORGANIZATION_PLAN.md`)
2. **Access cluster**: Retrieve all results from `/home/pdong/Astro UROP/`
3. **Package results**: Create organized results repository
4. **Document outputs**: Add metadata to all result files
5. **Prepare handoff**: Create summary for supervisors

---

## Technical Environment

**Cluster**: MIT Supercloud (Engaging)
**Partition**: `mit_normal_gpu`
**Conda Environment**: `myenv`
**Python Dependencies**:
- numpy, pandas, scipy
- scikit-learn
- astropy (FITS file handling)
- hdbscan, umap-learn
- matplotlib, seaborn
- pytorch (for deep learning experiments)
- h5py (HDF5 output)
- light_curve package (feature extraction)

**Data Scale**: eRASS1 full sky survey (hundreds of thousands of light curves)
