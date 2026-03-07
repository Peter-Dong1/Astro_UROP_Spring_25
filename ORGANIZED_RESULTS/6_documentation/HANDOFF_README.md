# eROSITA Light Curve Analysis - Results Package

**For**: Dan & Supervisors
**Date**: March 2026
**Student**: Peter Dong
**Project**: eRASS1 X-ray Light Curve Event Detection & Classification

---

## 🎯 Quick Start - What You Asked For

### "Lists of sources with cosine similarity"

**Location**: `../3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/`

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

**Files** (Expected):
1. `em01_211120_020_similar.csv` - Top 100 sources similar to first known source
2. `em01_039135_020_similar.csv` - Top 100 sources similar to second known source
3. `em01_038099_020_similar.csv` - Top 100 sources similar to third known source

**Columns in each CSV**:
- `rank` - Similarity rank (1 = most similar)
- `file_name` - FITS filename
- `file_path` - Full path to light curve
- `cosine_similarity` - Similarity score [0, 1] where 1 = identical
- `cluster_label` - Which HDBSCAN cluster this source belongs to (from Run 237)
- Feature values (weighted_mean, weighted_variance, bexvar, etc.)

**How to use**:
```python
import pandas as pd

# Load top 100 similar sources for first known source
similar = pd.read_csv('3_similarity_analysis/.../em01_211120_020_similar.csv')

print(f"Top similar source: {similar.iloc[0]['file_name']}")
print(f"Similarity score: {similar.iloc[0]['cosine_similarity']:.4f}")
print(f"Belongs to cluster: {similar.iloc[0]['cluster_label']}")

# Show top 10
print(similar[['rank', 'file_name', 'cosine_similarity', 'cluster_label']].head(10))
```

---

## Project Summary

### What Was Done

Analyzed ~200,000 X-ray light curves from the eROSITA All-Sky Survey (eRASS1) using:

1. **Statistical Feature Extraction**
   - Extracted 10 features per light curve
   - Features capture flux level, variability, temporal patterns, and shape
   - See `DATA_DICTIONARY.md` for feature definitions

2. **HDBSCAN Clustering**
   - Unsupervised clustering in 9-dimensional feature space
   - Identified groups of sources with similar X-ray behavior
   - Run 237 is the main result (parameters: min_cluster=7, epsilon=0.2)

3. **Cosine Similarity Analysis** ⭐
   - For 3 known interesting sources, ranked all ~200k sources by similarity
   - Top 100 most similar sources per known source
   - **This is the key deliverable you requested**

### Key Results

- **Run 237** is the main clustering result
  - Parameters: min_cluster=7, epsilon=0.2, leaf method, min_samples=5
  - Found [N] clusters (see `cluster_summary.csv` after downloading)
  - Used for cosine similarity analysis

- **Top 100 similarity lists** generated for 3 known sources

- **Feature histograms** showing distributions across 640-source sample

---

## Repository Structure

```
ORGANIZED_RESULTS/
├── 1_features/                    # Extracted features
│   ├── README.md
│   ├── features.pkl               # ⚠️ TO DOWNLOAD - Main features file
│   └── feature_histograms_640/    # ✅ Present - 10 histogram PNGs
│
├── 2_clustering_runs/             # HDBSCAN clustering results
│   ├── README.md
│   ├── run_217_important/         # ⚠️ TO DOWNLOAD - Reference run
│   ├── run_237_main/              # ⚠️ TO DOWNLOAD - PRIMARY RESULT
│   └── run_266_large_clusters/    # ⚠️ TO DOWNLOAD - Large clusters only
│
├── 3_similarity_analysis/         # ⭐⭐⭐ COSINE SIMILARITY LISTS
│   └── README.md
│   └── analysis_results/          # ⚠️ TO DOWNLOAD
│       └── cluster_assignments_237_real_clippedxvar/
│           ├── em01_211120_020_similar.csv
│           ├── em01_039135_020_similar.csv
│           └── em01_038099_020_similar.csv
│
├── 4_visualizations/              # Plots and figures
│   ├── README.md
│   └── run_266_cluster_samples/   # ⚠️ TO DOWNLOAD - Cluster sample plots
│
├── 5_code/                        # Complete pipeline code
│   ├── README.md
│   ├── config.py, helper.py, bexvar_ero.py
│   ├── pipeline/                  # 13 Python scripts
│   │   ├── run_feature_extraction.py
│   │   ├── run_pipeline_on_features.py
│   │   ├── analyze_similar_curves.py  # ⭐ Cosine similarity
│   │   └── ...
│   └── slurm_scripts/             # 7 SLURM job scripts
│
└── 6_documentation/               # Detailed documentation
    ├── HANDOFF_README.md          # This file
    ├── RUN_HISTORY.md             # Complete experimental timeline
    ├── DATA_DICTIONARY.md         # Feature & column definitions
    └── METHODS.md                 # Methodology details
```

**Legend**:
- ✅ Present - Files are in repository
- ⚠️ TO DOWNLOAD - Need to download from cluster

---

## Directory Guide

### 1_features/
- `features.pkl` - All extracted features (to download; load with pandas)
- Feature histograms showing distributions across 640-source sample

**Key file**: `features.pkl` is a pandas DataFrame with ~200k rows × 15 columns

### 2_clustering_runs/
- **`run_237_main/`** ⭐ - **Primary clustering result**
  - `cluster_assignments.csv` - Cluster label for each source
  - `umap_embedding.csv` - 2D visualization coordinates
  - `cluster_probabilities.csv` - Membership confidence
  - `outlier_scores.csv` - GLOSH outlier scores

- `run_217_important/` - Important reference run (rerun of 215)
- `run_266_large_clusters/` - Large clusters only (used for sample plots)

### 3_similarity_analysis/ ⭐⭐⭐
- **The lists you specifically requested**
- Top 100 similar sources for each of 3 known light curves
- Includes cluster assignments and similarity scores
- **This is the key deliverable**

### 4_visualizations/
- Sample light curve plots for clusters
- Grid layouts showing 25 random samples per cluster
- Can generate additional visualizations from clustering results

### 5_code/
- Complete pipeline code
- Can reproduce entire analysis with same parameters
- See `5_code/README.md` for usage instructions

### 6_documentation/
- **`RUN_HISTORY.md`**: Complete timeline of all experimental runs
- **`DATA_DICTIONARY.md`**: Definitions of all features and columns
- **`METHODS.md`**: Detailed methodology
- **`HANDOFF_README.md`**: This file

---

## How to Use the Results

### Load cluster assignments:
```python
import pandas as pd

# Load Run 237 clustering (main result)
clusters = pd.read_csv('2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')

print(f"Total sources: {len(clusters)}")
print(f"Number of clusters: {(clusters['cluster_label'] >= 0).sum()}")
print(f"Noise points: {(clusters['cluster_label'] == -1).sum()}")

# Cluster sizes
print(clusters['cluster_label'].value_counts())
```

### Load cosine similarity results:
```python
# Load top 100 similar to first known source
similar_1 = pd.read_csv(
    '3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/em01_211120_020_similar.csv'
)

# Show top 10
print(similar_1[['rank', 'file_name', 'cosine_similarity', 'cluster_label']].head(10))

# What clusters do similar sources belong to?
print("Cluster distribution of similar sources:")
print(similar_1['cluster_label'].value_counts())
```

### Load features:
```python
features = pd.read_pickle('1_features/features.pkl')

print(f"Analyzed {len(features)} sources")
print(f"Features: {[c for c in features.columns if c not in ['file_path', 'file_name', 'light_curve']]}")

# Access light curve data
first_lc = features.iloc[0]['light_curve']
print(f"Time points in first light curve: {len(first_lc)}")
```

### Merge features with clusters:
```python
# Combine features and cluster assignments
data = features.merge(clusters, on='file_path')

# Get all sources in cluster 5
cluster_5 = data[data['cluster_label'] == 5]
print(f"Cluster 5 has {len(cluster_5)} sources")

# Analyze features of cluster 5
print("Mean bexvar in cluster 5:", cluster_5['bexvar'].mean())
```

### Visualize UMAP:
```python
import matplotlib.pyplot as plt

umap = pd.read_csv('2_clustering_runs/run_237_main/umap_data/umap_embedding.csv')
data = clusters.merge(umap, on='file_path')

plt.figure(figsize=(12, 10))
scatter = plt.scatter(data['umap_x'], data['umap_y'],
                     c=data['cluster_label'], cmap='tab20',
                     s=1, alpha=0.5)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('HDBSCAN Clusters - Run 237')
plt.show()
```

---

## Key Findings (To Be Updated)

After downloading cluster results, update this section with:
- Total number of clusters found
- Cluster size distribution
- Most anomalous sources (from outlier_scores.csv)
- Cluster assignments of 3 known sources
- Patterns in similarity results

---

## What's Still on the Cluster

These files need to be downloaded from the MIT Engaging cluster:

**Priority 1** (Cosine similarity - what you asked for):
```bash
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/analysis_results/
```

**Priority 2** (Main clustering - Run 237):
```bash
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/237/
```

**Priority 3** (Features file):
```bash
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl
```

**Priority 4** (Additional runs & visualizations):
```bash
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/217/
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/266/
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/plots/all266/CLUSTERS/
```

---

## Questions?

**Detailed documentation**:
- `RUN_HISTORY.md` - Complete experimental timeline with all run numbers
- `DATA_DICTIONARY.md` - What each feature means
- `METHODS.md` - Methodology details (HDBSCAN, UMAP, cosine similarity)

**Original project notes**:
- `../PROJECT_STATUS.md` - Project status document
- `../FINAL_REORGANIZATION_PLAN.md` - How this was organized

**Code usage**:
- `5_code/README.md` - How to use all scripts
- `5_code/pipeline/analyze_similar_curves.py` - Cosine similarity implementation

---

## Reproducing the Analysis

See `5_code/README.md` for complete workflow.

**Quick version**:
```bash
# 1. Extract features (on cluster, parallel)
python run_feature_extraction.py --job-id 0 --num-jobs 100

# 2. Consolidate features
python consolidate_features.py

# 3. Run clustering (Run 237 parameters)
python run_pipeline_on_features.py \
  --min-cluster 7 --epsilon 0.2 --min-samples 5 \
  --cluster-method leaf --run-number 237

# 4. Compute cosine similarity
python analyze_similar_curves.py

# 5. Generate visualizations
python sample_clusters.py --run 237 --samples 25
```

**Configuration**: All parameters in `5_code/config.py`

---

## Contact

**Student**: Peter Dong
**Supervisors**: Dan & Riccardo (MIT Kavli Institute)
**Project Period**: January - May 2026

---

## Checklist for Completeness

Before considering this handoff complete:

- [ ] `features.pkl` downloaded from cluster
- [ ] Run 237 results downloaded (cluster_assignments.csv, umap_embedding.csv)
- [ ] **Cosine similarity CSVs downloaded** ⭐⭐⭐ (em01_*_similar.csv files)
- [ ] Run 217 and 266 results downloaded
- [ ] Cluster sample plots downloaded
- [ ] All CSVs verified to have proper headers and data
- [ ] Summary statistics generated (cluster_summary.csv)
- [ ] HANDOFF_README.md updated with actual numbers

**Once complete**: This package contains everything needed to understand and reproduce the analysis.
