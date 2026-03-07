# Pipeline Code

This directory contains all code used for the feature extraction, clustering, and similarity analysis.

---

## Core Configuration & Utilities

### config.py
**Purpose**: Central configuration file for all pipeline parameters

**Key contents**:
- `FEATURES_FILE` - Path to main features pickle file
- `KNOWN_LIGHT_CURVES` - List of 3 interesting sources for similarity analysis
- `SELECTED_FEATURES_FOR_CLUSTERING` - Which features to use in HDBSCAN
- UMAP parameters (`n_neighbors`, `min_dist`)
- HDBSCAN default parameters
- Data paths and directories

**Usage**: All scripts import from this file to maintain consistency

### helper.py
**Purpose**: Core data loading and utility functions

**Key functions**:
- `load_light_curve(file_path, band, trunc)` - Load single FITS light curve
- `load_n_light_curves(n, fits_files, band, trunc)` - Load multiple light curves
- `load_all_fits_files(data_dir)` - Get all FITS file paths
- `partition_data(light_curves, test_size, val_size)` - Split data
- Index mapping for fast lookups

**Data format**:
Returns pandas DataFrames with columns: `TIME`, `TIMEDEL`, `RATE`, `ERRM`, `ERRP`, `SYM_ERR`

### bexvar_ero.py
**Purpose**: Bayesian excess variance calculation (eROSITA-specific)

**Key function**:
- `compute_bexvar(time, rate, errm, errp)` - Computes Bayesian excess variance statistic

**Reference**: Adapted from external bexvar library

---

## Pipeline Scripts (pipeline/)

### 1. Feature Extraction

#### run_feature_extraction.py ⭐
**Purpose**: Main feature extraction script

**Usage**:
```bash
# Single job
python run_feature_extraction.py

# Parallel processing (SLURM array job)
python run_feature_extraction.py --job-id 0 --num-jobs 100
```

**What it does**:
- Loads FITS light curves from data directory
- Extracts 10 statistical features per light curve
- Saves features as pickled DataFrame
- Supports parallel processing for large datasets

**Output**: `features.pkl` (DataFrame with all features)

#### batch_feature_extraction.py
**Purpose**: Batch processing wrapper for feature extraction

**What it does**:
- Processes light curves in chunks
- Handles parallel SLURM array jobs
- Useful for processing ~200k light curves

#### consolidate_features.py
**Purpose**: Merge features from parallel jobs

**Usage**:
```bash
python consolidate_features.py
```

**What it does**:
- Combines feature DataFrames from multiple jobs
- Creates single unified `features.pkl`
- Removes duplicates and validates

---

### 2. Clustering & Analysis

#### run_pipeline_on_features.py ⭐⭐
**Purpose**: **MAIN ANALYSIS SCRIPT** - HDBSCAN clustering + UMAP

**Usage**:
```bash
python run_pipeline_on_features.py \
  --min-cluster 7 \
  --epsilon 0.2 \
  --min-samples 5 \
  --run-number 237
```

**What it does**:
1. Loads features from `features.pkl`
2. Normalizes features (RobustScaler)
3. Runs HDBSCAN clustering
4. Computes UMAP dimensionality reduction
5. Optionally runs Isolation Forest outlier detection
6. Saves results:
   - `data/all/{run_number}/hdbscan_data/cluster_assignments.csv`
   - `data/all/{run_number}/umap_data/umap_embedding.csv`
   - `data/all/{run_number}/web_data/features.h5` (optional)

**Parameters**:
- `--min-cluster`: Minimum cluster size (default: 24)
- `--epsilon`: Cluster selection epsilon (default: 0.11)
- `--min-samples`: Minimum samples for core points (default: 12)
- `--cluster-method`: 'leaf' or 'eom' (default: 'leaf')

**Run 237 parameters**: `--min-cluster 7 --epsilon 0.2 --min-samples 5 --cluster-method leaf`

#### analyze_similar_curves.py ⭐⭐⭐
**Purpose**: **COSINE SIMILARITY ANALYSIS** (key deliverable for supervisors)

**Usage**:
```bash
python analyze_similar_curves.py
```

**What it does**:
1. Loads features from `features.pkl`
2. Loads cluster assignments from Run 237
3. Computes cosine similarity between all sources
4. For each of 3 known sources, finds top 100 most similar
5. Merges with cluster labels
6. Saves to CSV files:
   - `analysis_results/.../em01_211120_020_similar.csv`
   - `analysis_results/.../em01_039135_020_similar.csv`
   - `analysis_results/.../em01_038099_020_similar.csv`

**Known sources** (from config.py):
- `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
- `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
- `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

---

### 3. Visualization

#### plot_cluster_samples.py
**Purpose**: Plot sample light curves from a single cluster

**Usage**:
```bash
python plot_cluster_samples.py --cluster 5 --num-samples 25
```

**What it does**:
- Loads light curves from cluster
- Plots N random samples in grid layout
- Saves to PNG

#### sample_clusters.py ⭐
**Purpose**: Generate sample plots for ALL clusters in a run

**Usage**:
```bash
python sample_clusters.py --run 237 --samples 25
```

**What it does**:
- Iterates through all clusters in specified run
- Creates 5×5 grid plots (25 samples per cluster)
- Saves to `data/all/{run}/sample_cluster_plots/`

**Used for**: Run 266 cluster visualizations

---

### 4. Utilities

#### bexvar_histograms.py
**Purpose**: Generate feature distribution histograms

**Usage**:
```bash
python bexvar_histograms.py "features.pkl" --outdir "output_dir"
```

**What it does**:
- Loads features DataFrame
- Generates histogram for each feature
- Saves PNG files

**Used for**: Creating the 640features/ histograms

#### append_new_features.py
**Purpose**: Add new features to existing DataFrame

**Usage**:
```bash
python append_new_features.py
```

**What it does**:
- Loads existing `features.pkl`
- Computes additional features
- Saves updated DataFrame

#### split_curves.py
**Purpose**: Split light curves into time segments

**What it does**:
- Splits long light curves into shorter segments
- Used for testing temporal evolution

#### check_nans.py
**Purpose**: Data quality checks

**What it does**:
- Scans features DataFrame for NaN values
- Identifies problematic sources

#### inspect_features.py
**Purpose**: Interactive feature DataFrame inspection

**What it does**:
- Loads and displays feature statistics
- Useful for debugging

---

## SLURM Scripts (slurm_scripts/)

### analyze_features.slurm
Runs `run_pipeline_on_features.py` on cluster

### batch_get_features.slurm
Batch feature extraction with SLURM array jobs

### get_features.slurm
Single-job feature extraction

### consol_feat.slurm
Consolidate features from parallel jobs

### append_feat.slurm
Append new features to existing DataFrame

### run_chunks.slurm
Process data in chunks

### split_curves.slurm
Split light curves into segments

**SLURM Configuration**:
- Partition: `mit_normal_gpu` or `sched_mit_hill`
- Conda environment: `myenv`
- Typical resources: 4-8 CPUs, 32-64GB RAM

---

## Reproducing the Analysis

### Full Pipeline from Scratch:

```bash
# 1. Extract features (parallel processing)
sbatch --array=0-99 get_features.slurm  # On cluster

# 2. Consolidate features
python consolidate_features.py

# 3. Run clustering (Run 237 parameters)
python run_pipeline_on_features.py \
  --min-cluster 7 \
  --epsilon 0.2 \
  --min-samples 5 \
  --cluster-method leaf \
  --run-number 237

# 4. Compute cosine similarity
python analyze_similar_curves.py

# 5. Generate visualizations
python sample_clusters.py --run 237 --samples 25

# 6. Generate histograms (optional, on subset)
python bexvar_histograms.py "features.pkl" --outdir "histograms"
```

---

## Dependencies

**Python packages**:
- `pandas`, `numpy`, `scipy` - Data handling
- `scikit-learn` - Feature scaling, Isolation Forest
- `hdbscan` - HDBSCAN clustering
- `umap-learn` - UMAP dimensionality reduction
- `astropy` - FITS file I/O
- `light_curve` - Advanced time series features
- `matplotlib`, `seaborn` - Visualization
- `h5py` - HDF5 output (optional)

**Install**:
```bash
conda create -n myenv python=3.9
conda activate myenv
pip install pandas numpy scipy scikit-learn hdbscan umap-learn astropy matplotlib seaborn h5py
pip install light_curve  # For advanced features
```

---

## Key Workflow Diagram

```
FITS files → run_feature_extraction.py → features.pkl
                                              ↓
                                    run_pipeline_on_features.py
                                    ↙                         ↘
                    cluster_assignments.csv            umap_embedding.csv
                                    ↓
                        analyze_similar_curves.py
                                    ↓
                        em01_*_similar.csv ⭐⭐⭐
```

---

## Notes

- All paths in scripts may reference cluster directories (`/home/pdong/Astro UROP/...`)
- Adjust paths in `config.py` when running locally
- `helper.py` tracks inaccessible light curves in `inaccessible_lightcurves.txt`
- Feature extraction handles variable-length light curves (truncated to 20 points max)
- HDBSCAN handles high-dimensional feature space (9-10 features)
