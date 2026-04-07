# Final Repository Reorganization Plan
**Based on: Meeting Notes Analysis + Current Codebase State**

**Goal**: Organize this repository for handoff to Dan and supervisors with clear documentation of all experiments, results, and deliverables (especially "lists of sources with cosine similarity").

---

## Understanding Your Experimental Timeline

### Phase 1: Deep Learning Exploration (Root Directory - Early Semester)
**Period**: January - February
**Approaches Tried**:
- RNN-based VAE (GRU encoder/decoder)
- Transformer-based VAE (with positional encoding)
- LSTM Autoencoder experiments (in notebooks)

**Key Files**:
- `RNN_9_model.py`, `train_rnn.py`, `test_rnn.py`
- `trans_model.py`, `test_trans.py`
- `.ipynb_checkpoints/LSTM_AutoEncoder-checkpoint.ipynb`
- `.ipynb_checkpoints/Raw Data Clustering-checkpoint.ipynb`

**Outcome**: Exploratory - moved away from deep learning to statistical approaches

---

### Phase 2: Statistical Feature Extraction (Main Work - March onwards)
**Period**: March - May
**Location**: `z New Feature Extraction Pipeline/`

**This is where your actual results live!**

---

## Decoding Your Run Numbers

From your meeting notes, here's what each run represents:

### Early Experiments (100s series)
```
Run 100-108: Initial HDBSCAN parameter exploration
- Testing min_cluster_size: 3 or 5
- Testing epsilon: 0, 1, 3, 5
- All with very small min_samples
```
**Status**: Exploratory - likely superseded

### Fine-tuning (200s series)
```
Run 200-223: Systematic parameter sweeps
- min_cluster: 3
- epsilon: varied from 0 to 0.5
- EOM: testing 'eom' vs 'leaf'
- min_samples: 3

Key runs:
- Run 217: ⭐ RERUN OF 215 (min_cluster=3, epsilon=0.1, leaf, min_samples=3)
- Run 223: ✂️ CLIPPED VERSION (removing excess variance outliers)
```
**Status**: Run 217 and 223 likely important

### Refinement (230s series)
```
Run 231-243: Testing slightly larger min_cluster sizes

Key parameters tested:
- Run 231: min_cluster=5, epsilon=0.13, eom='eom', min_samples=3
- Run 232: min_cluster=7, epsilon=0.11, eom='eom', min_samples=5
- Run 233: Same as 232 but 'leaf'
- Run 237: ⭐⭐ 233 + leaf + epsilon=0.2
- Run 243: Same as 237
```
**Status**: **Run 237 is THE IMPORTANT ONE**
- Mentioned in folder name: `cluster_assignments_237_real_clippedxvar`
- This appears to be your "final" clustering run

### Larger Clusters (260s series)
```
Run 260-267: Testing much larger min_cluster_size
- 260: min_cluster=20
- 265: min_cluster=25, min_samples=12
- 266: min_cluster=50, min_samples=25 ⭐
- 267: min_cluster=100, min_samples=50
```
**Status**: Run 266 used for generating sample plots (from meeting notes)

### Final Experiments (300s series)
```
Run 301-311: Unknown parameter variations
Numbers listed in notes: 99.7, 95, 68, 40, 30, 20
- 310: 30 without exvar
- 311: 20 w/o exvar, min_cluster=50, min_samples=25
```
**Status**: These might be describing cluster counts, not run numbers?

### The 640 Features Run
```
From meeting notes page 1:
python bexvar_histograms.py \
  "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/640/processedbatch/feature.pkl" \
  --outdir "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/640features"
```
**This is where your local `640features/` histograms came from!**
- Run on 640 light curves
- Generated the 10 feature histograms you have locally

---

## What You Currently Have (Local Repository)

### In `640features/` Directory
```
✅ 10 feature histograms (PNG files):
- bexvar_hist.png
- beyond1std_hist.png
- excess_var_hist.png
- hurst_exp_hist.png
- lag1_autocorr_hist.png
- mean_rise_fall_ratio_hist.png
- mean_var_hist.png
- stetson_k_hist.png
- weighted_mean_hist.png
- weighted_variance_hist.png
```
**Source**: Feature extraction run on 640 light curves
**Status**: ✅ Keep - these are good summary visualizations

### What You DON'T Have Locally (Still on Cluster)
```
❌ Main features file: features.pkl
❌ Cluster assignments for runs 217, 237, 266, etc.
❌ UMAP embeddings
❌ Cosine similarity results ⭐⭐⭐ (WHAT SUPERVISORS WANT)
❌ HDF5 web visualization files
❌ Sample cluster plots
```

**These are all on the cluster at**: `/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/`

---

## Step-by-Step Reorganization Plan

## STEP 1: Retrieve Results from Cluster 🔥 PRIORITY

### 1.1 Connect to Cluster
```bash
ssh pdong@engaging.mit.edu
# OR
ssh pdong@orcd-login001.mit.edu
```

### 1.2 Navigate and Survey
```bash
cd "/home/pdong/Astro UROP/z New Feature Extraction Pipeline"

# List all data directories
ls -lh data/
ls -lh data/640/
ls -lh data/all/

# Find run directories
find data/all -type d -name "[0-9]*" | sort
```

### 1.3 Identify Key Files to Download

**Priority 1 - Cosine Similarity Results** ⭐⭐⭐
```bash
# These are what your supervisors specifically asked for
find data/all -name "*similar*" -o -name "*cosine*" -o -name "SIG_NEV*"
```

**Priority 2 - Run 237 Results** (Your main clustering run)
```bash
ls -R data/all/237/
# OR
ls -R data/all/analysis_results/cluster_assignments_237_real_clippedxvar/
```

**Priority 3 - Features File**
```bash
# Main features file
ls -lh data/all/amp_max_features/features.pkl
# OR
ls -lh data/640/processedbatch/feature.pkl
```

**Priority 4 - Run 217 and 266 Results**
```bash
ls -R data/all/217/
ls -R data/all/266/
ls -R plots/all266/CLUSTERS/
```

### 1.4 Download Everything

**From your LOCAL machine** (not on cluster):

```bash
# Create local directory
mkdir -p ~/Desktop/erosita_cluster_results
cd ~/Desktop/erosita_cluster_results

# Download features file
scp "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/amp_max_features/features.pkl" ./

# Download Run 237 results (THE MAIN RUN)
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/237" ./

# Download analysis_results folder (COSINE SIMILARITY)
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/analysis_results" ./

# Download Run 217 results
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/217" ./

# Download Run 266 results and plots
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/266" ./
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/plots/all266" ./

# Download 640 features data if different from what you have
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/640" ./640_dataset
```

### 1.5 Download Any Additional Runs
```bash
# If you find other important run numbers on the cluster:
scp -r "pdong@engaging.mit.edu:/home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/data/all/[RUN_NUMBER]" ./
```

---

## STEP 2: Organize Local Repository Structure

### 2.1 Create Clean Directory Structure

```bash
# In your project root: /Users/peterd/RealAstroStuff/Astro_UROP_Spring_25/

# Create organized structure
mkdir -p ORGANIZED_RESULTS/{1_features,2_clustering_runs,3_similarity_analysis,4_visualizations,5_code,6_documentation}
```

### 2.2 Move Files to Organized Structure

**A. Features**
```bash
# Move features file
mv ~/Desktop/erosita_cluster_results/features.pkl \
   ORGANIZED_RESULTS/1_features/

# Move existing histograms
mv "z New Feature Extraction Pipeline/640features" \
   ORGANIZED_RESULTS/1_features/feature_histograms_640/
```

**B. Clustering Runs**
```bash
# Move clustering results
mv ~/Desktop/erosita_cluster_results/217 \
   ORGANIZED_RESULTS/2_clustering_runs/run_217_important/

mv ~/Desktop/erosita_cluster_results/237 \
   ORGANIZED_RESULTS/2_clustering_runs/run_237_main/

mv ~/Desktop/erosita_cluster_results/266 \
   ORGANIZED_RESULTS/2_clustering_runs/run_266_large_clusters/
```

**C. Similarity Analysis** ⭐⭐⭐
```bash
# THE KEY DELIVERABLE FOR SUPERVISORS
mv ~/Desktop/erosita_cluster_results/analysis_results \
   ORGANIZED_RESULTS/3_similarity_analysis/
```

**D. Visualizations**
```bash
# Move plots
mv ~/Desktop/erosita_cluster_results/all266 \
   ORGANIZED_RESULTS/4_visualizations/run_266_cluster_samples/
```

**E. Code**
```bash
# Copy (don't move) the pipeline code
cp -r "z New Feature Extraction Pipeline"/*.py \
   ORGANIZED_RESULTS/5_code/pipeline/

cp -r "z New Feature Extraction Pipeline"/*.slurm \
   ORGANIZED_RESULTS/5_code/slurm_scripts/

cp -r "z New Feature Extraction Pipeline"/config.py \
   ORGANIZED_RESULTS/5_code/

cp -r "z New Feature Extraction Pipeline"/helper.py \
   ORGANIZED_RESULTS/5_code/

cp -r "z New Feature Extraction Pipeline"/bexvar_ero.py \
   ORGANIZED_RESULTS/5_code/
```

---

## STEP 3: Create Documentation for Each Directory

### 3.1 Document Features Directory
```bash
cd ORGANIZED_RESULTS/1_features
```

Create `README.md`:
```markdown
# Extracted Features

## features.pkl
- **Source**: Feature extraction from all eRASS1 light curves
- **Location**: Originally at `/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl`
- **Size**: [CHECK SIZE]
- **Format**: Pickled pandas DataFrame
- **Columns**:
  - file_path
  - file_name
  - light_curve (nested DataFrame)
  - weighted_mean
  - weighted_variance
  - lag1_autocorr
  - hurst_exp
  - mean_rise_fall_ratio
  - stetson_k
  - bexvar
  - mean_var
  - ampl_sig
  - (possibly: excess_var, beyond1std, etc.)

## feature_histograms_640/
Feature distribution histograms generated from 640 light curves sample.

**Command used**:
```bash
python bexvar_histograms.py \
  "data/640/processedbatch/feature.pkl" \
  --outdir "640features"
```

**Files**:
- bexvar_hist.png - Bayesian excess variance distribution
- beyond1std_hist.png - Fraction of points beyond 1σ
- excess_var_hist.png - Excess variance measure
- hurst_exp_hist.png - Hurst exponent (persistence measure)
- lag1_autocorr_hist.png - Lag-1 autocorrelation
- mean_rise_fall_ratio_hist.png - Rise/fall asymmetry
- mean_var_hist.png - Mean variance
- stetson_k_hist.png - Stetson K variability index
- weighted_mean_hist.png - Error-weighted mean flux
- weighted_variance_hist.png - Error-weighted variance
```

### 3.2 Document Clustering Runs
```bash
cd ORGANIZED_RESULTS/2_clustering_runs
```

Create `README.md`:
```markdown
# Clustering Runs

All runs use HDBSCAN clustering algorithm with UMAP dimensionality reduction.

## Run 217 (Important Reference Run)
**Date**: Mid-semester
**Parameters**:
- min_cluster_size: 3
- cluster_selection_epsilon: 0.1
- cluster_selection_method: 'leaf'
- min_samples: 3

**Notes**: This was a rerun of Run 215 mentioned specifically in meeting notes.

**Expected files**:
- hdbscan_data/cluster_assignments.csv
- hdbscan_data/cluster_probabilities.csv
- hdbscan_data/outlier_scores.csv
- umap_data/umap_embedding.csv

## Run 237 (MAIN FINAL RUN) ⭐
**Date**: Late semester
**Parameters**:
- min_cluster_size: 7 (from run 233)
- cluster_selection_epsilon: 0.2
- cluster_selection_method: 'leaf'
- min_samples: 5

**Notes**:
- This is the "real_clippedxvar" run mentioned in analysis_results folders
- Excess variance outliers were clipped before clustering
- **This appears to be the primary clustering result to share with supervisors**

**Expected files**:
- hdbscan_data/cluster_assignments.csv ⭐
- hdbscan_data/cluster_probabilities.csv
- hdbscan_data/outlier_scores.csv
- umap_data/umap_embedding.csv ⭐
- web_data/features.h5
- web_data/light_curves.h5

## Run 266 (Large Cluster Analysis)
**Date**: Late semester
**Parameters**:
- min_cluster_size: 50
- min_samples: 25

**Notes**:
- Designed to find larger, more significant clusters
- Sample plots generated with: `python sample_clusters.py --run 266 --samples 25`

**Expected files**:
- hdbscan_data/cluster_assignments.csv
- Sample plots in ../4_visualizations/run_266_cluster_samples/
```

### 3.3 Document Similarity Analysis ⭐⭐⭐
```bash
cd ORGANIZED_RESULTS/3_similarity_analysis
```

Create `README.md`:
```markdown
# Cosine Similarity Analysis

**THIS IS WHAT DAN AND YOUR SUPERVISORS REQUESTED**

## Overview
For each of the 3 known interesting light curves, this analysis finds the top 100 most similar sources using cosine similarity on feature vectors.

## Known Source Light Curves
From config.py:
1. `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
2. `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
3. `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

## Directory Structure

### analysis_results/
Contains similarity analysis outputs from various runs.

#### cluster_assignments_237_real_clippedxvar/
Results linked to Run 237 (main clustering run).

**Expected contents**:
- Top 100 similar sources for each known source (CSV files)
- Cluster labels for similar sources
- Similarity scores

#### feature_histograms_237_real_clippedxvar/
Feature histograms for Run 237 analysis.

#### sig_nev_analysis/
Source ID mappings and analysis.

**Expected files**:
- `SIG_NEV_mappings.pkl` - ID cross-reference
- Similar source lists

## How Similarity Was Computed
From meeting notes:
1. Features normalized to unit vectors (RobustScaler)
2. Cosine similarity computed between all sources
3. Euclidean distance used in HDBSCAN (equivalent: ||x-y||² = 2(1-cos(x,y)))
4. Top 100 most similar sources ranked for each known source

## Key Deliverables for Supervisors
- [ ] Top 100 similar sources CSV for each of 3 known sources
- [ ] Cluster assignments showing which clusters similar sources belong to
- [ ] Similarity scores (0-1 scale)
```

---

## STEP 4: Generate Missing Outputs (If Needed)

### 4.1 Check What You Have

After downloading from cluster, check:

```bash
# Check if you have the main similarity CSVs
find ORGANIZED_RESULTS/3_similarity_analysis -name "*similar*" -o -name "*em01_*"

# Check cluster assignments
ls ORGANIZED_RESULTS/2_clustering_runs/run_237_main/hdbscan_data/
```

### 4.2 If Cosine Similarity Files Don't Exist, Regenerate

**On cluster** (since features.pkl is large):

```bash
ssh pdong@engaging.mit.edu
cd "/home/pdong/Astro UROP/z New Feature Extraction Pipeline"

# Run similarity analysis
python analyze_similar_curves.py

# This should generate files in data/all/analysis_results/
```

Then download the newly generated files.

### 4.3 Generate Summary Statistics

Create `generate_summary_stats.py` in your local repo:

```python
#!/usr/bin/env python3
"""Generate summary statistics for the handoff."""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path("ORGANIZED_RESULTS")
FEATURES_FILE = RESULTS_DIR / "1_features" / "features.pkl"
RUN_237_CLUSTERS = RESULTS_DIR / "2_clustering_runs" / "run_237_main" / "hdbscan_data" / "cluster_assignments.csv"

def generate_feature_summary():
    """Generate feature summary statistics."""
    print("Loading features...")
    features_df = pd.read_pickle(FEATURES_FILE)

    # Drop light_curve column for summary
    features_only = features_df.drop(columns=['light_curve'], errors='ignore')

    # Summary stats
    summary = features_only.describe()
    summary.to_csv(RESULTS_DIR / "1_features" / "feature_summary_statistics.csv")

    print(f"✅ Feature summary saved")
    print(f"   Total sources: {len(features_df)}")
    print(f"   Features extracted: {len(features_only.columns) - 2}")  # Exclude file_path, file_name

    return features_df

def generate_cluster_summary():
    """Generate cluster summary."""
    print("\nLoading cluster assignments...")
    clusters_df = pd.read_csv(RUN_237_CLUSTERS)

    # Cluster counts
    cluster_counts = clusters_df['cluster_label'].value_counts().sort_index()
    cluster_counts.to_csv(RESULTS_DIR / "2_clustering_runs" / "run_237_main" / "cluster_summary.csv")

    print(f"✅ Cluster summary saved")
    print(f"   Total sources clustered: {len(clusters_df)}")
    print(f"   Number of clusters: {(cluster_counts.index >= 0).sum()}")
    print(f"   Noise points (cluster -1): {cluster_counts.get(-1, 0)}")

    return clusters_df

if __name__ == "__main__":
    features_df = generate_feature_summary()
    clusters_df = generate_cluster_summary()
    print("\n✅ All summaries generated!")
```

Run it:
```bash
python generate_summary_stats.py
```

---

## STEP 5: Create Handoff Package

### 5.1 Final Directory Structure

```
ORGANIZED_RESULTS/
├── 1_features/
│   ├── README.md
│   ├── features.pkl                    # Main features file
│   ├── feature_summary_statistics.csv  # Generated summary
│   └── feature_histograms_640/         # 10 histogram PNGs
│
├── 2_clustering_runs/
│   ├── README.md
│   ├── run_217_important/
│   │   ├── hdbscan_data/
│   │   │   ├── cluster_assignments.csv
│   │   │   ├── cluster_probabilities.csv
│   │   │   └── outlier_scores.csv
│   │   └── umap_data/
│   │       └── umap_embedding.csv
│   ├── run_237_main/                  # ⭐ PRIMARY RUN
│   │   ├── cluster_summary.csv        # Generated summary
│   │   ├── hdbscan_data/
│   │   │   ├── cluster_assignments.csv
│   │   │   ├── cluster_probabilities.csv
│   │   │   └── outlier_scores.csv
│   │   ├── umap_data/
│   │   │   └── umap_embedding.csv
│   │   └── web_data/
│   │       ├── features.h5
│   │       └── light_curves.h5
│   └── run_266_large_clusters/
│       └── hdbscan_data/
│           └── cluster_assignments.csv
│
├── 3_similarity_analysis/              # ⭐⭐⭐ KEY DELIVERABLE
│   ├── README.md
│   └── analysis_results/
│       ├── cluster_assignments_237_real_clippedxvar/
│       │   ├── em01_211120_020_similar.csv          # Top 100 similar
│       │   ├── em01_039135_020_similar.csv          # Top 100 similar
│       │   └── em01_038099_020_similar.csv          # Top 100 similar
│       ├── feature_histograms_237_real_clippedxvar/
│       └── sig_nev_analysis/
│           └── SIG_NEV_mappings.pkl
│
├── 4_visualizations/
│   ├── README.md
│   └── run_266_cluster_samples/
│       └── cluster_*_samples_*.png
│
├── 5_code/
│   ├── README.md
│   ├── pipeline/
│   │   ├── run_feature_extraction.py
│   │   ├── batch_feature_extraction.py
│   │   ├── consolidate_features.py
│   │   ├── run_pipeline_on_features.py
│   │   ├── analyze_similar_curves.py
│   │   ├── plot_cluster_samples.py
│   │   └── sample_clusters.py
│   ├── slurm_scripts/
│   │   └── *.slurm
│   ├── config.py
│   ├── helper.py
│   └── bexvar_ero.py
│
└── 6_documentation/
    ├── HANDOFF_README.md              # Overview for supervisors
    ├── RUN_HISTORY.md                 # Complete run history
    ├── DATA_DICTIONARY.md             # Feature definitions
    └── METHODS.md                     # Methodology description
```

### 5.2 Create Handoff README

Create `ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`:

```markdown
# eROSITA Light Curve Analysis - Results Package

**For**: Dan & Supervisors
**Date**: March 2026
**Student**: Peter Dong

## Quick Start - What You Asked For

### "Lists of sources with cosine similarity"

**Location**: `3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/`

**Files**:
1. `em01_211120_020_similar.csv` - Top 100 sources similar to first known source
2. `em01_039135_020_similar.csv` - Top 100 sources similar to second known source
3. `em01_038099_020_similar.csv` - Top 100 sources similar to third known source

**Columns in each CSV**:
- `rank` - Similarity rank (1 = most similar)
- `file_name` - FITS file name
- `file_path` - Full path to light curve
- `cosine_similarity` - Similarity score [0, 1]
- `cluster_label` - Which HDBSCAN cluster this source belongs to
- Feature values (weighted_mean, bexvar, etc.)

---

## Project Summary

### What Was Done
Analyzed ~200,000 X-ray light curves from eRASS1 using:
1. **Feature Extraction**: 10 statistical features per light curve
2. **HDBSCAN Clustering**: Identified groups of similar sources
3. **Cosine Similarity**: Ranked all sources by similarity to 3 known interesting sources

### Key Results
- **Run 237** is the main clustering result (parameters: min_cluster=7, epsilon=0.2)
- Found [N] clusters (see cluster_summary.csv)
- Generated top 100 similarity lists for 3 known sources

---

## Directory Guide

### 1_features/
- `features.pkl` - All extracted features (load with pandas)
- Feature histograms showing distributions

### 2_clustering_runs/
- `run_237_main/` - **Primary clustering result**
  - `cluster_assignments.csv` - Cluster label for each source
  - `umap_embedding.csv` - 2D visualization coordinates

### 3_similarity_analysis/ ⭐
- **The lists you requested**
- Top 100 similar sources for each known light curve
- Includes cluster assignments and similarity scores

### 4_visualizations/
- Sample light curve plots for each cluster

### 5_code/
- Complete pipeline code
- Can reproduce analysis with same parameters

---

## How to Use the Results

### Load cluster assignments:
```python
import pandas as pd

clusters = pd.read_csv('2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')
print(f"Found {clusters['cluster_label'].nunique()} clusters")
```

### Load similarity results:
```python
similar_to_source1 = pd.read_csv(
    '3_similarity_analysis/.../em01_211120_020_similar.csv'
)
print(f"Top similar source: {similar_to_source1.iloc[0]['file_name']}")
print(f"Similarity score: {similar_to_source1.iloc[0]['cosine_similarity']}")
```

### Load features:
```python
features = pd.read_pickle('1_features/features.pkl')
print(f"Analyzed {len(features)} sources")
print(f"Features: {[c for c in features.columns if c not in ['file_path', 'file_name', 'light_curve']]}")
```

---

## Questions?

See detailed documentation in:
- `RUN_HISTORY.md` - Complete experimental timeline
- `DATA_DICTIONARY.md` - What each feature means
- `METHODS.md` - Methodology details

Original project notes: See PROJECT_STATUS.md and ORGANIZATION_PLAN.md in parent directory.
```

### 5.3 Create Run History Document

Create `ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md`:

```markdown
# Complete Run History

## Experimental Timeline

### Phase 1: Deep Learning Exploration (Jan-Feb)
**Location**: Root directory
**Status**: Exploratory, not used in final results

- RNN-based VAE experiments
- Transformer-based VAE experiments
- LSTM Autoencoder in notebooks

**Outcome**: Moved to statistical approaches for better interpretability

---

### Phase 2: Statistical Feature Extraction (Mar-May)
**Location**: `z New Feature Extraction Pipeline/`
**Status**: Main work, results in this package

#### 100s Series (Early Exploration)
- Runs 100-108: Initial HDBSCAN parameter exploration
- Very small min_cluster_size (3-5)
- Testing epsilon values

#### 200s Series (Systematic Tuning)
- Runs 200-223: Parameter sweeps
- **Run 217**: Rerun of 215, important reference
  - min_cluster=3, epsilon=0.1, leaf, min_samples=3
- **Run 223**: Clipped version (removed excess_var outliers)

#### 230s Series (Refinement)
- Runs 231-243: Testing slightly larger clusters
- **Run 237** ⭐⭐⭐: MAIN FINAL RUN
  - Parameters: min_cluster=7, epsilon=0.2, leaf, min_samples=5
  - Based on run 233 with epsilon adjustment
  - This is the "clippedxvar" run referenced in analysis

#### 260s Series (Large Cluster Analysis)
- Runs 260-267: Much larger min_cluster_size
- **Run 266**: min_cluster=50, min_samples=25
  - Used for sample cluster visualizations
  - Command: `python sample_clusters.py --run 266 --samples 25`

#### 640 Dataset
- Feature extraction on 640 light curve sample
- Generated feature histograms (in `1_features/feature_histograms_640/`)

---

## Key Decisions Made

### Why Statistical Features?
- Deep learning struggled with small number of time points (~10-20 per curve)
- Features more interpretable for astronomers
- Can handle variable-length sequences naturally

### Features Selected for Clustering
From `config.py`:
- weighted_mean, weighted_variance
- lag1_autocorr (temporal correlation)
- hurst_exp (persistence measure)
- mean_rise_fall_ratio
- stetson_k (variability index)
- bexvar (Bayesian excess variance)
- mean_var
- ampl_sig

### Why HDBSCAN?
- Density-based clustering
- Can find clusters of varying shapes/sizes
- Identifies noise points (unlike K-means)
- No need to specify number of clusters

### Cosine Similarity Implementation
- Normalized features to unit vectors
- Used euclidean distance in HDBSCAN: ||x-y||² = 2(1-cos(x,y))
- Ranked top 100 most similar sources for each known source

---

## Run Parameters Summary

| Run | min_cluster | epsilon | method | min_samples | Notes |
|-----|-------------|---------|--------|-------------|-------|
| 217 | 3 | 0.1 | leaf | 3 | Important reference |
| 237 | 7 | 0.2 | leaf | 5 | **MAIN RUN** |
| 266 | 50 | ? | ? | 25 | Large clusters |

---

## Known Issues Encountered

1. **Permissions errors**: Some FITS files inaccessible (tracked in `inaccessible_lightcurves.txt`)
2. **Skew/kurtosis features**: Caused NaN errors, removed from pipeline
3. **Posterior collapse**: Issue with early VAE models (Phase 1)
4. **GPU allocation**: Limited GPU availability affected deep learning experiments

---

## What Would Be Done Next

From meeting notes:
- Random forest to see feature importance for cluster assignment
- Web visualization interface (HDF5 files prepared)
- Cross-matching with external catalogs
- Physical interpretation of clusters
```

---

## STEP 6: Archive Old/Exploratory Code

### 6.1 Create Archive Directory

```bash
mkdir -p ARCHIVED_CODE/
mkdir -p ARCHIVED_CODE/deep_learning_experiments
mkdir -p ARCHIVED_CODE/notebooks
```

### 6.2 Move Old Code

```bash
# Move root directory deep learning code
mv RNN_9_model.py RNN_train.py train_rnn.py test_rnn.py test_model_9.py \
   ARCHIVED_CODE/deep_learning_experiments/

mv trans_model.py test_trans.py \
   ARCHIVED_CODE/deep_learning_experiments/

mv plotmodel.py plotmodelerror.py \
   ARCHIVED_CODE/deep_learning_experiments/

mv cont_9.py \
   ARCHIVED_CODE/deep_learning_experiments/

# Move notebooks
mv .ipynb_checkpoints/ \
   ARCHIVED_CODE/notebooks/

# Move old helper/extraction if duplicates
# (Keep main ones in ORGANIZED_RESULTS/5_code/)
```

### 6.3 Create Archive README

Create `ARCHIVED_CODE/README.md`:

```markdown
# Archived Exploratory Code

This directory contains code from early experimental phases that are not part of the final results.

## deep_learning_experiments/
RNN and Transformer VAE implementations from January-February.

**Why archived**:
- Struggled with small number of time points per light curve
- Results not as interpretable as statistical features
- Moved to statistical approach for final analysis

## notebooks/
Jupyter notebook checkpoints from interactive analysis sessions.

These contain early exploration and visualization work.

---

**Note**: The final pipeline code is in `ORGANIZED_RESULTS/5_code/`
```

---

## STEP 7: Update Root README

Update the main repository `README.md`:

```markdown
# eROSITA Light Curve Analysis Project

**Project**: Detection and clustering of X-ray transients in eRASS1 data
**Institution**: MIT Kavli Institute
**Period**: January - May 2026
**Student**: Peter Dong

---

## 🎯 For Supervisors (Dan & Team)

### What You Asked For: "Lists of sources with cosine similarity"

**Location**: `ORGANIZED_RESULTS/3_similarity_analysis/`

Top 100 most similar sources for each of the 3 known interesting light curves, including:
- Similarity scores
- Cluster assignments
- Feature values

**Start here**: `ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`

---

## Repository Structure

```
├── ORGANIZED_RESULTS/           ⭐ MAIN DELIVERABLES
│   ├── 1_features/             # Extracted features from all light curves
│   ├── 2_clustering_runs/      # HDBSCAN clustering results
│   ├── 3_similarity_analysis/  # Cosine similarity lists ⭐⭐⭐
│   ├── 4_visualizations/       # Cluster sample plots
│   ├── 5_code/                 # Complete pipeline code
│   └── 6_documentation/        # Detailed documentation
│
├── z New Feature Extraction Pipeline/  # Original working directory
│   └── (pipeline code - copied to ORGANIZED_RESULTS)
│
├── ARCHIVED_CODE/              # Exploratory/unused code
│   ├── deep_learning_experiments/
│   └── notebooks/
│
├── Markdown Files/             # Original project documentation
├── PROJECT_STATUS.md           # Complete project history
├── ORGANIZATION_PLAN.md        # How this was organized
├── FINAL_REORGANIZATION_PLAN.md  # This file
└── CLAUDE.md                   # Guide for Claude Code
```

---

## Quick Start

### Load Results
```python
import pandas as pd

# Load features
features = pd.read_pickle('ORGANIZED_RESULTS/1_features/features.pkl')

# Load main clustering (Run 237)
clusters = pd.read_csv('ORGANIZED_RESULTS/2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')

# Load similarity results for first known source
similar = pd.read_csv('ORGANIZED_RESULTS/3_similarity_analysis/analysis_results/.../em01_211120_020_similar.csv')
```

---

## Documentation

- **For supervisors**: `ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`
- **Complete history**: `ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md`
- **Feature definitions**: `ORGANIZED_RESULTS/6_documentation/DATA_DICTIONARY.md`
- **Methodology**: `ORGANIZED_RESULTS/6_documentation/METHODS.md`
- **Original status**: `PROJECT_STATUS.md`

---

## Reproducing Results

See code in `ORGANIZED_RESULTS/5_code/pipeline/`

Main workflow:
1. Feature extraction: `run_feature_extraction.py`
2. Clustering: `run_pipeline_on_features.py`
3. Similarity analysis: `analyze_similar_curves.py`
4. Visualization: `plot_cluster_samples.py`

Configuration: `config.py`

---

## Data Location (Cluster)

Original data and additional results still on cluster:
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/
```

---

## Contact

Peter Dong - [email]
Supervisors: Dan & Riccardo (MIT Kavli Institute)
```

---

## STEP 8: Create Final Checklist

Create `HANDOFF_CHECKLIST.md`:

```markdown
# Handoff Checklist

Use this to verify everything is ready for supervisors.

## Data Retrieved from Cluster

- [ ] features.pkl downloaded
- [ ] Run 237 results downloaded (main clustering)
- [ ] Run 217 results downloaded (reference)
- [ ] Run 266 results downloaded (large clusters)
- [ ] Cosine similarity analysis results downloaded ⭐⭐⭐
- [ ] SIG_NEV_mappings.pkl downloaded
- [ ] Sample cluster plots downloaded

## Organization Complete

- [ ] Files moved to ORGANIZED_RESULTS structure
- [ ] Old code archived to ARCHIVED_CODE
- [ ] Directory READMEs created
- [ ] Handoff documentation created

## Key Deliverables Present

### Priority 1: Cosine Similarity Lists ⭐⭐⭐
- [ ] em01_211120_020_similar.csv exists
- [ ] em01_039135_020_similar.csv exists
- [ ] em01_038099_020_similar.csv exists
- [ ] Each has top 100 entries with similarity scores
- [ ] Cluster labels included

### Priority 2: Cluster Assignments
- [ ] Run 237 cluster_assignments.csv exists
- [ ] File has columns: file_path, cluster_label
- [ ] Cluster summary statistics generated

### Priority 3: Features
- [ ] features.pkl loads successfully
- [ ] Feature summary statistics generated
- [ ] Feature histograms present

### Priority 4: Visualizations
- [ ] Cluster sample plots exist
- [ ] At least 5-10 clusters visualized

## Documentation Complete

- [ ] HANDOFF_README.md created
- [ ] RUN_HISTORY.md created
- [ ] DATA_DICTIONARY.md created (see next step)
- [ ] METHODS.md created (see next step)
- [ ] Root README.md updated

## Code Organized

- [ ] Pipeline scripts copied to 5_code/pipeline/
- [ ] SLURM scripts copied to 5_code/slurm_scripts/
- [ ] config.py and helper.py copied
- [ ] Code README created

## Final Verification

- [ ] Test loading features.pkl
- [ ] Test loading cluster assignments
- [ ] Test loading similarity CSVs
- [ ] All CSVs have proper headers
- [ ] No broken file paths in documentation

## Ready for Handoff

- [ ] Create .zip archive of ORGANIZED_RESULTS
- [ ] Draft email to supervisors (template below)
- [ ] Include link/attachment to results

---

## Email Template

```
Subject: eROSITA Analysis Results - Cosine Similarity Lists & Clustering

Hi Dan,

I've completed the organization of my eROSITA light curve analysis results. Here's what I'm sharing:

**Key Deliverable - Cosine Similarity Lists** (what you requested):
- Location: ORGANIZED_RESULTS/3_similarity_analysis/
- Top 100 most similar sources for each of the 3 known interesting sources
- Includes similarity scores, cluster assignments, and feature values

**Primary Clustering Result**:
- Run 237: Main HDBSCAN clustering with parameters: min_cluster=7, epsilon=0.2
- Cluster assignments for all ~200k sources
- UMAP 2D embeddings for visualization

**Features & Documentation**:
- Complete feature extraction results (features.pkl)
- 10 statistical features per light curve
- Feature distribution histograms
- Full methodology documentation

**Quick Start**: See ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md

Everything is packaged in the attached/linked ORGANIZED_RESULTS directory with complete documentation.

Let me know if you need anything clarified or additional outputs!

Best,
Peter
```
```

---

## STEP 9: Create Remaining Documentation

Still need to create:

### DATA_DICTIONARY.md
(See ORGANIZATION_PLAN.md for template - expand with your specific features)

### METHODS.md
Create `ORGANIZED_RESULTS/6_documentation/METHODS.md`:

```markdown
# Methodology

## Overview

Statistical feature extraction + HDBSCAN clustering + cosine similarity ranking.

## Data Source

- eRASS1 (eROSITA All-Sky Survey 1)
- ~200,000 X-ray light curves
- 3 energy bands: Low (0.2-0.6 keV), Medium (0.6-2.3 keV), High (2.3-5.0 keV)
- ~10-20 time points per source

## Pipeline

### 1. Feature Extraction

**Script**: `run_feature_extraction.py`

For each light curve, extract 10 statistical features accounting for measurement uncertainties.

**Features**:
1. **weighted_mean**: Mean flux weighted by inverse variance
2. **weighted_variance**: Variance weighted by inverse variance
3. **lag1_autocorr**: Correlation between consecutive time points
4. **hurst_exp**: Measure of persistence (>0.5) vs mean-reversion (<0.5)
5. **mean_rise_fall_ratio**: Asymmetry of rising vs falling flux
6. **stetson_k**: Kurtosis-based variability measure
7. **bexvar**: Bayesian excess variance
8. **mean_var**: Ratio of mean flux to variance
9. **ampl_sig**: Amplitude significance relative to errors
10. **excess_var**: Classical excess variance measure

**Implementation**: Uses `light_curve` Python package + custom functions

### 2. Feature Normalization

- Method: RobustScaler (robust to outliers)
- Scales each feature to similar range for clustering

### 3. HDBSCAN Clustering

**Script**: `run_pipeline_on_features.py`

**Algorithm**: Hierarchical Density-Based Spatial Clustering of Applications with Noise

**Key Parameters** (Run 237):
- `min_cluster_size`: 7 (minimum points to form a cluster)
- `cluster_selection_epsilon`: 0.2 (distance threshold)
- `cluster_selection_method`: 'leaf' (bottom-up cluster selection)
- `min_samples`: 5 (neighborhood size for core points)

**Why HDBSCAN?**:
- Finds clusters of varying density
- Identifies noise/outliers (label = -1)
- No need to specify number of clusters
- More robust than DBSCAN

### 4. UMAP Dimensionality Reduction

**Purpose**: Visualization of high-dimensional feature space

**Parameters**:
- `n_neighbors`: 15
- `min_dist`: 0.1
- `n_components`: 2 (for 2D visualization)

Preserves both local and global structure better than PCA.

### 5. Cosine Similarity Analysis

**Script**: `analyze_similar_curves.py`

**Method**:
1. Normalize feature vectors to unit length
2. Compute cosine similarity between all pairs: cos(θ) = (A·B)/(||A|| ||B||)
3. For each known source, rank all others by similarity
4. Extract top 100 most similar

**Note on Distance Metric**:
HDBSCAN used euclidean distance, which relates to cosine similarity:
||x-y||² = 2(1 - cos(x,y))

### 6. Visualization

- Cluster sample plots: 25 random light curves per cluster
- UMAP scatter plots colored by cluster
- Feature histograms

## Clustering Performance

From Run 237:
- Number of clusters: [TO BE FILLED]
- Noise points: [TO BE FILLED]
- Cluster sizes: See cluster_summary.csv

## Validation

- Known interesting sources checked against cluster assignments
- Visual inspection of cluster samples
- Feature importance analysis (via statistical comparison)

## Software

- Python 3.9
- Key packages: pandas, numpy, scipy, scikit-learn, hdbscan, umap-learn
- Astronomy: astropy, light_curve package
- Visualization: matplotlib, seaborn

## Computational Resources

- MIT Engaging cluster
- SLURM job scheduler
- CPU-based computation (feature extraction + clustering)
```

---

## Summary - Next Actions for You

### Immediate (Today):
1. ✅ Connect to cluster and survey what data exists
2. ✅ Download all results from cluster (follow Step 1)
3. ✅ Move files into ORGANIZED_RESULTS structure (Step 2)

### Short-term (This Week):
4. ✅ Create all README files for each directory (Step 3)
5. ✅ Generate summary statistics (Step 4)
6. ✅ Create documentation files (Step 9)
7. ✅ Complete DATA_DICTIONARY.md with your feature definitions
8. ✅ Fill in specific numbers (cluster counts, etc.) where marked [TO BE FILLED]

### Final (Before Handoff):
9. ✅ Archive old code (Step 6)
10. ✅ Update root README (Step 7)
11. ✅ Go through HANDOFF_CHECKLIST.md (Step 8)
12. ✅ Create .zip of ORGANIZED_RESULTS directory
13. ✅ Send to supervisors with email template

---

## If You Need to Regenerate Anything

### Cosine Similarity Lists Missing?
```bash
# On cluster
cd "/home/pdong/Astro UROP/z New Feature Extraction Pipeline"
python analyze_similar_curves.py
```

### Feature Histograms Missing?
```bash
python bexvar_histograms.py "data/all/amp_max_features/features.pkl" --outdir "histograms_output"
```

### Cluster Sample Plots Missing?
```bash
python plot_cluster_samples.py --run 237 --num-samples 25
```

---

## Questions to Answer as You Organize

When you download the results, fill in these details:

1. **How many total sources?** (from features.pkl)
2. **How many clusters in Run 237?** (from cluster_assignments.csv)
3. **How many noise points?** (cluster_label = -1)
4. **Do the similarity CSVs exist?** (key deliverable check)
5. **Are HDF5 web files present?** (features.h5, light_curves.h5)
6. **Which runs actually have results on cluster?** (not all may have been saved)

---

## Final Note

This plan assumes:
- ✅ Run 237 is your main clustering result (from meeting notes)
- ✅ 640 features refer to a 640-source sample (from meeting notes)
- ✅ Cosine similarity analysis has been run (from meeting notes - you implemented it)

If any of these assumptions are wrong after you check the cluster, adjust accordingly!

The most critical item for your supervisors is the cosine similarity lists. Make sure those exist or can be regenerated.

Good luck! 🚀
```
