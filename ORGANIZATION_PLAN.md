# Project Organization & Handoff Plan

**Purpose**: Organize the eROSITA light curve analysis project for handoff to supervisors (Dan & team)

**Target Audience**: Dan, other supervisors, future researchers continuing this work

**Key Deliverable**: "Lists of sources with cosine similarity" + comprehensive code/results repository

---

## Current State Assessment

### ✅ What We Have
- Complete feature extraction pipeline code
- Feature histograms (10 features in `640features/`)
- Clustering and analysis scripts
- Configuration files with all parameters
- Documentation (README, Data.md, CLAUDE.md, PROJECT_STATUS.md)

### ⚠️ What Needs to be Retrieved from Cluster
- Cluster assignment CSVs
- Cosine similarity results
- Full feature extraction outputs (`features.pkl`)
- UMAP embeddings
- HDF5 web visualization files
- Additional plots and analysis outputs
- Run-specific results (run 217, 237, 501, etc.)

### ❌ What Needs to be Created
- Clean, documented results repository
- Summary statistics document
- Data dictionary for features
- README for results
- Comprehensive cosine similarity lists (formatted for easy use)

---

## Organization Plan - Phase by Phase

## Phase 1: Retrieve Results from Cluster ⭐ IMMEDIATE

### 1.1 Connect to Cluster & List Results

```bash
# SSH into cluster
ssh pdong@engaging.mit.edu  # or whatever the login is

# Navigate to project
cd /home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/

# List all result directories
ls -lh data/all/
ls -lh plots/
```

### 1.2 Identify Key Output Files

**Priority files to locate**:
```
data/all/
├── amp_max_features/features.pkl           # Main features file ⭐
├── {run_number}/
│   ├── hdbscan_data/
│   │   ├── cluster_assignments.csv        # Cluster labels ⭐
│   │   ├── cluster_probabilities.csv
│   │   └── outlier_scores.csv
│   ├── umap_data/
│   │   ├── umap_embedding.csv             # UMAP coordinates ⭐
│   │   └── cluster_assignments.csv
│   ├── web_data/
│   │   ├── features.h5
│   │   └── light_curves.h5
│   └── sample_cluster_plots/*.png
└── analysis_results/
    ├── cluster_assignments_237_real_clippedxvar/
    ├── feature_histograms_237_real_clippedxvar/
    └── sig_nev_analysis/
        ├── SIG_NEV_mappings.pkl           # Source mappings ⭐
        └── similar_sources_*.csv          # Cosine similarity ⭐⭐⭐
```

### 1.3 Download Results

```bash
# From local machine
# Create results directory locally
mkdir -p results_from_cluster

# Download features file
scp pdong@engaging.mit.edu:"/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/amp_max_features/features.pkl" results_from_cluster/

# Download all analysis results
scp -r pdong@engaging.mit.edu:"/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/analysis_results" results_from_cluster/

# Download cluster assignments for each run
scp -r pdong@engaging.mit.edu:"/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/*/hdbscan_data" results_from_cluster/

# Download UMAP embeddings
scp -r pdong@engaging.mit.edu:"/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/*/umap_data" results_from_cluster/

# Download plots
scp -r pdong@engaging.mit.edu:"/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/plots" results_from_cluster/
```

---

## Phase 2: Create Organized Results Repository

### 2.1 Directory Structure

Create this structure locally:

```
eROSITA_Results_Handoff/
│
├── README.md                              # Overview of results
├── DATA_DICTIONARY.md                     # Feature definitions
├── METHODS.md                             # Brief methods description
│
├── 1_extracted_features/
│   ├── README.md                          # What's in this folder
│   ├── features.pkl                       # Full feature DataFrame ⭐
│   ├── features_summary_statistics.csv    # Summary stats
│   └── feature_histograms/                # All feature distributions
│       ├── bexvar_hist.png
│       ├── hurst_exp_hist.png
│       └── ... (all 10+ features)
│
├── 2_cluster_results/                     # HDBSCAN clustering outputs
│   ├── README.md                          # Clustering parameters
│   ├── run_217/                          # If run 217 exists
│   │   ├── cluster_assignments.csv
│   │   ├── cluster_probabilities.csv
│   │   ├── outlier_scores.csv
│   │   ├── umap_embedding.csv
│   │   └── cluster_summary.txt           # Cluster sizes, etc.
│   ├── run_237/                          # Main run
│   │   └── ... (same structure)
│   └── run_501/                          # Latest run?
│       └── ... (same structure)
│
├── 3_similarity_analysis/                 # ⭐⭐⭐ KEY FOR SUPERVISORS
│   ├── README.md                          # How cosine similarity was computed
│   ├── top_100_similar_sources/          # For each known source
│   │   ├── em01_211120_020_LightCurve_00007_similar.csv
│   │   ├── em01_039135_020_LightCurve_00058_similar.csv
│   │   └── em01_038099_020_LightCurve_00005_similar.csv
│   ├── similarity_matrix.csv             # Full pairwise if available
│   ├── SIG_NEV_mappings.pkl              # ID mappings
│   └── all_sources_ranked_similarity.csv # All sources by similarity
│
├── 4_visualizations/
│   ├── cluster_sample_plots/             # Light curve grids by cluster
│   ├── umap_plots/                       # 2D UMAP embeddings colored by cluster
│   ├── feature_correlations/             # Feature correlation matrices
│   └── known_sources/                    # Plots of the 3 known sources
│
├── 5_web_data/                           # For interactive visualization
│   ├── features.h5
│   └── light_curves.h5
│
└── 6_code/                               # Clean, documented code
    ├── README.md                         # How to run the pipeline
    ├── environment.yml                   # Conda environment
    ├── requirements.txt                  # Pip requirements
    ├── config.py
    ├── run_full_pipeline.sh             # Master script
    ├── feature_extraction/
    │   ├── run_feature_extraction.py
    │   ├── batch_feature_extraction.py
    │   └── consolidate_features.py
    ├── analysis/
    │   ├── run_pipeline_on_features.py
    │   ├── analyze_similar_curves.py
    │   └── plot_cluster_samples.py
    ├── utilities/
    │   ├── helper.py
    │   └── bexvar_ero.py
    └── slurm_scripts/
        └── *.slurm
```

### 2.2 Essential Documents to Create

#### `README.md` (Root)
```markdown
# eROSITA Light Curve Analysis Results

## Overview
Statistical analysis and clustering of eRASS1 X-ray light curves.

## Key Findings
- Total light curves analyzed: [NUMBER]
- Features extracted: 10 statistical features
- Clusters identified: [NUMBER] via HDBSCAN
- Outliers detected: [NUMBER]

## What's Inside
- **1_extracted_features/**: Statistical features from all light curves
- **2_cluster_results/**: HDBSCAN clustering outputs
- **3_similarity_analysis/**: Cosine similarity between sources ⭐
- **4_visualizations/**: Plots and figures
- **5_web_data/**: HDF5 files for interactive visualization
- **6_code/**: Complete pipeline code

## Quick Start
See `2_cluster_results/run_XXX/cluster_assignments.csv` for cluster labels.
See `3_similarity_analysis/top_100_similar_sources/` for similar source lists.

## Contact
[Your contact info]
```

#### `DATA_DICTIONARY.md`
Document all features:
```markdown
# Feature Definitions

## Extracted Features

### weighted_mean
- **Description**: Mean flux weighted by measurement uncertainties
- **Units**: counts/s
- **Range**: [min, max]
- **Purpose**: Central tendency accounting for error bars

### weighted_variance
- **Description**: Variance of flux weighted by uncertainties
- **Units**: (counts/s)²
- **Purpose**: Measure of variability

### lag1_autocorr
- **Description**: Lag-1 autocorrelation coefficient
- **Range**: [-1, 1]
- **Interpretation**:
  - >0: Positive correlation (persistent)
  - <0: Negative correlation (oscillating)
  - ~0: No temporal correlation

...continue for all features...

## Cluster Labels
- **-1**: Noise/outliers (HDBSCAN)
- **0, 1, 2, ...**: Cluster IDs

## Similarity Scores
- **cosine_similarity**: Cosine of angle between feature vectors
- **Range**: [0, 1]
- **Interpretation**: 1 = identical, 0 = orthogonal
```

#### `3_similarity_analysis/README.md` ⭐
```markdown
# Cosine Similarity Analysis

## Overview
This directory contains lists of sources ranked by cosine similarity to known interesting sources.

## Known Sources Analyzed
1. `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
2. `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
3. `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

## File Descriptions

### `top_100_similar_sources/`
Contains CSV files with the 100 most similar sources for each known source.

**Columns**:
- `rank`: Similarity rank (1 = most similar)
- `file_path`: Path to FITS file
- `file_name`: Filename
- `cosine_similarity`: Similarity score [0, 1]
- `cluster_label`: HDBSCAN cluster ID
- `weighted_mean`: Mean flux
- `bexvar`: Bayesian excess variance
- ...(other features)

### `similarity_matrix.csv`
Full pairwise similarity matrix (if computed).

## How to Use
1. Open `em01_211120_020_LightCurve_00007_similar.csv`
2. Top row = most similar source
3. Use `file_path` to locate FITS file
4. Check `cluster_label` to see if similar sources cluster together

## Methodology
- Features used: 9 statistical features (see config)
- Features normalized using RobustScaler
- Similarity computed via cosine similarity
- Based on HDBSCAN run [XXX]
```

---

## Phase 3: Generate Missing Outputs

### 3.1 If Cosine Similarity Files Don't Exist

Run on cluster or locally (if you have features.pkl):

```python
# In z New Feature Extraction Pipeline/
python analyze_similar_curves.py
```

This should generate the similarity files automatically.

### 3.2 Create Summary Statistics

```python
# Create this script: generate_summaries.py
import pandas as pd
import numpy as np

# Load features
features_df = pd.read_pickle('features.pkl')

# Summary statistics
summary = features_df.describe()
summary.to_csv('features_summary_statistics.csv')

# Cluster summary
clusters = pd.read_csv('cluster_assignments.csv')
cluster_counts = clusters['cluster_label'].value_counts().sort_index()
cluster_counts.to_csv('cluster_sizes.csv')

print("Summaries generated!")
```

### 3.3 Export Clean CSVs

If you only have pickle files, convert to CSV:

```python
import pandas as pd

# Features
features_df = pd.read_pickle('features.pkl')

# Drop the actual light curve column (too nested for CSV)
features_export = features_df.drop(columns=['light_curve'], errors='ignore')
features_export.to_csv('features_table.csv', index=False)

print(f"Exported {len(features_export)} sources with {len(features_export.columns)} features")
```

---

## Phase 4: Clean and Document Code

### 4.1 Create Master Script

**`run_full_pipeline.sh`**:
```bash
#!/bin/bash
# Master script to run the full analysis pipeline

# Step 1: Extract features
echo "Step 1: Extracting features..."
python run_feature_extraction.py --job-id 0 --num-jobs 1

# Step 2: Run clustering analysis
echo "Step 2: Running HDBSCAN clustering and UMAP..."
python run_pipeline_on_features.py

# Step 3: Compute cosine similarity
echo "Step 3: Computing cosine similarity for known sources..."
python analyze_similar_curves.py

# Step 4: Plot cluster samples
echo "Step 4: Generating cluster visualizations..."
python sample_clusters.py --run 501 --samples 25

echo "Pipeline complete! Check results in data/all/"
```

### 4.2 Document Each Script

Add docstrings to all major scripts. Example:

```python
"""
run_pipeline_on_features.py

Main analysis script for eROSITA light curve clustering.

This script:
1. Loads extracted features from features.pkl
2. Performs HDBSCAN clustering
3. Applies UMAP dimensionality reduction
4. Identifies outliers using Isolation Forest
5. Saves cluster assignments, UMAP embeddings, and visualizations

Usage:
    python run_pipeline_on_features.py

Configuration:
    Edit config.py to change:
    - HDBSCAN parameters (min_cluster_size, min_samples, etc.)
    - UMAP parameters (n_neighbors, min_dist)
    - Features used for clustering (SELECTED_FEATURES_FOR_CLUSTERING)

Outputs:
    data/all/{number}/hdbscan_data/cluster_assignments.csv
    data/all/{number}/umap_data/umap_embedding.csv
    plots/{number}/cluster_plot.png

Author: Peter Dong
Date: 2025
"""
```

### 4.3 Create Environment Files

**`environment.yml`**:
```yaml
name: erosita_analysis
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - astropy
  - h5py
  - pip
  - pip:
    - hdbscan
    - umap-learn
    - light_curve
```

**`requirements.txt`**:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
astropy>=5.0
h5py>=3.0
hdbscan>=0.8.28
umap-learn>=0.5.3
light_curve>=0.8.0
```

---

## Phase 5: Package for Handoff

### 5.1 Create Archive

```bash
# Create organized directory
mkdir eROSITA_Results_Handoff
cd eROSITA_Results_Handoff

# Copy organized files (following structure from Phase 2.1)
# ... populate directories ...

# Create archive
cd ..
tar -czf eROSITA_Results_$(date +%Y%m%d).tar.gz eROSITA_Results_Handoff/

# Or create zip for Windows compatibility
zip -r eROSITA_Results_$(date +%Y%m%d).zip eROSITA_Results_Handoff/
```

### 5.2 Create Handoff Email/Document

**Template**:
```
Subject: eROSITA Light Curve Analysis - Results & Code Package

Hi Dan,

Attached/linked is the complete results package for the eROSITA light curve analysis project.

KEY DELIVERABLES:
1. **Cosine Similarity Lists** (3_similarity_analysis/top_100_similar_sources/)
   - Top 100 most similar sources for each of the 3 known interesting sources
   - Includes similarity scores, cluster labels, and feature values

2. **Cluster Assignments** (2_cluster_results/)
   - HDBSCAN clustering results
   - ~[N] clusters identified from [M] light curves
   - Outlier scores and cluster probabilities included

3. **Extracted Features** (1_extracted_features/)
   - 10 statistical features extracted from each light curve
   - Feature histograms showing distributions
   - Full feature table (features.pkl and features_table.csv)

4. **Complete Code** (6_code/)
   - Feature extraction pipeline
   - Clustering and similarity analysis
   - All scripts documented and ready to run

QUICK START:
- See eROSITA_Results_Handoff/README.md for overview
- Similarity lists: 3_similarity_analysis/top_100_similar_sources/*.csv
- Cluster results: 2_cluster_results/run_XXX/cluster_assignments.csv
- Code documentation: 6_code/README.md

DATA DICTIONARY:
- See DATA_DICTIONARY.md for feature definitions
- See METHODS.md for brief methodology

The archive is [SIZE] and includes [N] light curves analyzed.

Please let me know if you need any clarification or additional outputs!

Best,
Peter
```

---

## Phase 6: Final Checklist

Before handoff, verify:

### Data Completeness
- [ ] features.pkl retrieved from cluster
- [ ] All cluster_assignments.csv files retrieved
- [ ] UMAP embeddings retrieved
- [ ] Cosine similarity CSVs exist for all 3 known sources
- [ ] Feature histograms (all 10+) included
- [ ] Cluster sample plots generated

### Documentation
- [ ] Root README.md created
- [ ] DATA_DICTIONARY.md complete with all features
- [ ] METHODS.md describes methodology
- [ ] Each subdirectory has its own README
- [ ] Code has docstrings
- [ ] config.py is well-commented

### Code
- [ ] All scripts run without errors
- [ ] Paths in config.py documented
- [ ] environment.yml and requirements.txt created
- [ ] Master pipeline script created (run_full_pipeline.sh)
- [ ] SLURM scripts documented

### Key Deliverables
- [ ] ⭐ Top 100 similar sources CSV for each known source
- [ ] ⭐ Cluster assignments CSV
- [ ] ⭐ Feature summary statistics
- [ ] ⭐ UMAP embedding coordinates
- [ ] Sample cluster plots
- [ ] SIG_NEV_mappings.pkl

### Metadata
- [ ] Number of light curves analyzed documented
- [ ] Number of clusters documented
- [ ] Clustering parameters documented
- [ ] Date of analysis included
- [ ] Contact info added

---

## Quick Win: Minimal Handoff Package

If time is limited, create minimal package with essentials:

```
eROSITA_Results_Minimal/
├── README.md                              # Quick overview
├── features_table.csv                     # All features
├── cluster_assignments.csv                # Cluster labels
├── top_100_similar_sources/              # ⭐ KEY: Similarity lists
│   ├── em01_211120_020_LightCurve_00007_similar.csv
│   ├── em01_039135_020_LightCurve_00058_similar.csv
│   └── em01_038099_020_LightCurve_00005_similar.csv
├── feature_histograms/                   # 10 histogram PNGs
└── code/                                 # Main analysis scripts
    ├── run_pipeline_on_features.py
    ├── analyze_similar_curves.py
    └── config.py
```

**This minimal package gives supervisors what they need to explore results immediately.**

---

## Timeline Estimate

- **Phase 1** (Retrieve from cluster): 1-2 hours
- **Phase 2** (Organize structure): 2-3 hours
- **Phase 3** (Generate missing outputs): 1-2 hours
- **Phase 4** (Document code): 2-3 hours
- **Phase 5** (Package): 30 minutes
- **Phase 6** (Verify): 1 hour

**Total**: 7-11 hours for full package
**Minimal package**: 3-4 hours

---

## Next Steps After Handoff

Potential follow-up work:
1. Interactive web visualization (using HDF5 files)
2. Physical interpretation of clusters
3. Cross-matching with external catalogs
4. Time-domain analysis of specific clusters
5. Publication preparation

---

## Contact & Questions

If supervisors have questions, they should be able to:
1. Reproduce the analysis using the code
2. Understand each feature via the data dictionary
3. Explore similar sources via the CSV files
4. Visualize clusters via the plots

Make sure documentation is clear enough that someone new can understand the project!
