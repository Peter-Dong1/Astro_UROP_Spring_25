# eROSITA Light Curve Analysis Project

**Project**: Detection and clustering of X-ray transients in eRASS1 data
**Institution**: MIT Kavli Institute
**Period**: January - May 2026
**Student**: Peter Dong

---

## 🎯 For Supervisors (Dan & Team)

### What You Asked For: "Lists of sources with cosine similarity"

**📍 Location**: `ORGANIZED_RESULTS/3_similarity_analysis/`

Top 100 most similar sources for each of the 3 known interesting light curves, including:
- Similarity scores (cosine similarity [0-1])
- Cluster assignments (from HDBSCAN Run 237)
- All feature values

**⚠️ Status**: Files need to be downloaded from cluster (see below)

**🚀 Start here**: [`ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`](ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md)

---

## Quick Navigation

| What You Need | Where to Find It |
|---------------|------------------|
| **Cosine similarity lists** ⭐⭐⭐ | `ORGANIZED_RESULTS/3_similarity_analysis/` |
| **Main clustering result** (Run 237) | `ORGANIZED_RESULTS/2_clustering_runs/run_237_main/` |
| **Feature definitions** | `ORGANIZED_RESULTS/6_documentation/DATA_DICTIONARY.md` |
| **How it was done** | `ORGANIZED_RESULTS/6_documentation/METHODS.md` |
| **Complete timeline** | `ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md` |
| **Supervisor guide** | `ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md` |
| **Pipeline code** | `ORGANIZED_RESULTS/5_code/` |

---

## Repository Structure

```
Astro_UROP_Spring_25/
│
├── ORGANIZED_RESULTS/                    ⭐ MAIN DELIVERABLES
│   ├── 1_features/                       # Extracted features from light curves
│   │   ├── README.md
│   │   ├── features.pkl                  # ⚠️ TO DOWNLOAD
│   │   └── feature_histograms_640/       # ✅ 10 histogram PNGs
│   │
│   ├── 2_clustering_runs/                # HDBSCAN clustering results
│   │   ├── README.md
│   │   ├── run_217_important/            # ⚠️ TO DOWNLOAD
│   │   ├── run_237_main/                 # ⚠️ TO DOWNLOAD - PRIMARY RESULT
│   │   └── run_266_large_clusters/       # ⚠️ TO DOWNLOAD
│   │
│   ├── 3_similarity_analysis/            # ⭐⭐⭐ COSINE SIMILARITY LISTS
│   │   ├── README.md
│   │   └── analysis_results/             # ⚠️ TO DOWNLOAD
│   │       └── cluster_assignments_237_real_clippedxvar/
│   │           ├── em01_211120_020_similar.csv
│   │           ├── em01_039135_020_similar.csv
│   │           └── em01_038099_020_similar.csv
│   │
│   ├── 4_visualizations/                 # Plots and figures
│   │   ├── README.md
│   │   └── run_266_cluster_samples/      # ⚠️ TO DOWNLOAD
│   │
│   ├── 5_code/                           # Complete pipeline code
│   │   ├── README.md
│   │   ├── config.py, helper.py, bexvar_ero.py
│   │   ├── pipeline/                     # 13 analysis scripts
│   │   └── slurm_scripts/                # 7 SLURM job scripts
│   │
│   └── 6_documentation/                  # Detailed documentation
│       ├── HANDOFF_README.md             # 🚀 START HERE for supervisors
│       ├── RUN_HISTORY.md                # Complete experimental timeline
│       ├── DATA_DICTIONARY.md            # Feature & column definitions
│       └── METHODS.md                    # Methodology details
│
├── z New Feature Extraction Pipeline/    # Original working directory
│   └── (pipeline code - copied to ORGANIZED_RESULTS)
│
├── ARCHIVED_CODE/                        # Exploratory/unused code
│   ├── README.md
│   ├── deep_learning_experiments/        # Phase 1: RNN/Transformer VAEs
│   └── notebooks/                        # Jupyter notebook checkpoints
│
├── Markdown Files/                       # Original project documentation
│   ├── README.md
│   ├── Data.md
│   └── multi_bands.md
│
├── PROJECT_STATUS.md                     # Complete project history
├── FINAL_REORGANIZATION_PLAN.md          # How this was organized
├── REORGANIZATION_PROGRESS.md            # Reorganization progress tracker
└── CLAUDE.md                             # Guide for Claude Code
```

**Legend**:
- ✅ **Present** - Files are in repository now
- ⚠️ **TO DOWNLOAD** - Need to download from MIT Engaging cluster
- ⭐ **Key deliverable** - What supervisors specifically requested

---

## Project Summary

### What Was Done

Analyzed ~200,000 X-ray light curves from the **eROSITA All-Sky Survey (eRASS1)** using:

1. **Statistical Feature Extraction**
   - 10 features per light curve (flux level, variability, temporal patterns)
   - Features: weighted_mean, weighted_variance, lag1_autocorr, hurst_exp, stetson_k, bexvar, etc.

2. **HDBSCAN Clustering**
   - Unsupervised clustering in 9-dimensional feature space
   - **Run 237** (main result): min_cluster=7, epsilon=0.2, leaf method
   - Identified groups of sources with similar X-ray behavior

3. **Cosine Similarity Analysis** ⭐
   - For 3 known interesting sources, ranked all ~200k sources by similarity
   - **Top 100 most similar sources per known source**
   - This is the key deliverable requested by supervisors

4. **UMAP Visualization**
   - 2D projection of high-dimensional feature space
   - Visualize cluster structure

### Key Results

- **Run 237** is the main clustering result used for all deliverables
- Top 100 similarity lists generated for 3 known sources
- Feature histograms show distributions across sample dataset
- Complete documentation of methodology and results

---

## Quick Start - Using the Results

### Load cluster assignments:
```python
import pandas as pd

# Main clustering result (Run 237)
clusters = pd.read_csv('ORGANIZED_RESULTS/2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')

print(f"Number of clusters: {(clusters['cluster_label'] >= 0).nunique()}")
print(f"Noise points: {(clusters['cluster_label'] == -1).sum()}")
```

### Load cosine similarity results:
```python
# Top 100 similar sources for first known source
similar = pd.read_csv('ORGANIZED_RESULTS/3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/em01_211120_020_similar.csv')

print(similar[['rank', 'file_name', 'cosine_similarity', 'cluster_label']].head(10))
```

### Load features:
```python
features = pd.read_pickle('ORGANIZED_RESULTS/1_features/features.pkl')

print(f"Analyzed {len(features)} sources")
print(f"Features: {[c for c in features.columns if c not in ['file_path', 'file_name', 'light_curve']]}")
```

More examples in: [`ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`](ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md)

---

## What's Still on the Cluster

These files need to be downloaded from the MIT Engaging cluster:

### Priority 1: Cosine Similarity Lists ⭐⭐⭐
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/analysis_results/
```

### Priority 2: Main Clustering (Run 237)
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/237/
```

### Priority 3: Features File
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl
```

### Priority 4: Additional Runs & Visualizations
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/217/
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/266/
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/plots/all266/CLUSTERS/
```

**Download instructions**: See [`FINAL_REORGANIZATION_PLAN.md`](FINAL_REORGANIZATION_PLAN.md) STEP 1

---

## Documentation

### For Supervisors:
- 🚀 **[HANDOFF_README.md](ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md)** - Start here! Quick guide to all results
- **[DATA_DICTIONARY.md](ORGANIZED_RESULTS/6_documentation/DATA_DICTIONARY.md)** - What each feature and column means
- **[METHODS.md](ORGANIZED_RESULTS/6_documentation/METHODS.md)** - Detailed methodology
- **[RUN_HISTORY.md](ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md)** - Complete experimental timeline

### For Technical Details:
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Complete project status and history
- **[CLAUDE.md](CLAUDE.md)** - Technical guide for working with this codebase
- **[5_code/README.md](ORGANIZED_RESULTS/5_code/README.md)** - How to use all pipeline scripts

### Planning Documents:
- **[FINAL_REORGANIZATION_PLAN.md](FINAL_REORGANIZATION_PLAN.md)** - Detailed reorganization plan
- **[REORGANIZATION_PROGRESS.md](REORGANIZATION_PROGRESS.md)** - Progress tracker

---

## Reproducing the Analysis

See [`ORGANIZED_RESULTS/5_code/README.md`](ORGANIZED_RESULTS/5_code/README.md) for complete workflow.

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

# 4. Compute cosine similarity (KEY DELIVERABLE)
python analyze_similar_curves.py

# 5. Generate visualizations
python sample_clusters.py --run 237 --samples 25
```

All code in: `ORGANIZED_RESULTS/5_code/`

---

## Experimental Timeline

### Phase 1: Deep Learning Exploration (Jan-Feb 2026)
- Location: `ARCHIVED_CODE/deep_learning_experiments/`
- Tried: RNN VAEs, Transformer VAEs, LSTM Autoencoders
- **Outcome**: Not used in final results (data too sparse, interpretability issues)

### Phase 2: Statistical Feature Extraction (Mar-May 2026)
- Location: `ORGANIZED_RESULTS/` (main work)
- **Run 100-217**: Early parameter exploration
- **Run 237**: ⭐ Main final result (parameters: min_cluster=7, epsilon=0.2)
- **Run 266**: Large clusters for visualizations
- **640 features**: Sample dataset for histogram generation

See [`ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md`](ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md) for complete timeline.

---

## Data Source

**eRASS1** (eROSITA All-Sky Survey 1)
- ~200,000 X-ray light curves
- 3 energy bands: Low (0.2-0.6 keV), Medium (0.6-2.3 keV), High (2.3-5.0 keV)
- ~10-20 time points per source
- Cluster location: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`

---

## Known Interesting Sources

From `config.py`, the 3 sources used for cosine similarity analysis:
1. `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
2. `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
3. `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

---

## Software & Dependencies

**Python 3.9** with:
- `pandas`, `numpy`, `scipy` - Data handling
- `scikit-learn` - Feature scaling
- `hdbscan` - HDBSCAN clustering
- `umap-learn` - UMAP dimensionality reduction
- `astropy` - FITS file I/O
- `matplotlib`, `seaborn` - Visualization

See [`ORGANIZED_RESULTS/6_documentation/METHODS.md`](ORGANIZED_RESULTS/6_documentation/METHODS.md) for complete details.

---

## Contact

**Student**: Peter Dong
**Supervisors**: Dan & Riccardo (MIT Kavli Institute)

---

## Repository Organization

This repository was reorganized in March 2026 to prepare for handoff to supervisors:

- **Old work** archived to `ARCHIVED_CODE/`
- **Final results** organized in `ORGANIZED_RESULTS/`
- **Complete documentation** provided for all results
- **Code** copied to organized structure for easy access

See [`FINAL_REORGANIZATION_PLAN.md`](FINAL_REORGANIZATION_PLAN.md) for reorganization details.

---

## Next Steps for Supervisors

1. ✅ Review this README and [`ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`](ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md)
2. ⚠️ Download results from cluster (see "What's Still on the Cluster" above)
3. ✅ Load cosine similarity CSVs (`ORGANIZED_RESULTS/3_similarity_analysis/`)
4. ✅ Explore cluster assignments (`ORGANIZED_RESULTS/2_clustering_runs/run_237_main/`)
5. ✅ Cross-match with external catalogs (SIMBAD, NED) to identify source types
6. ✅ Physical interpretation of clusters and similar sources

---

**For questions or clarifications, see detailed documentation in `ORGANIZED_RESULTS/6_documentation/`**
