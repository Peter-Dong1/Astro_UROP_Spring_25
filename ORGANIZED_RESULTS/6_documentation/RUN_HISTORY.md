# Complete Run History

**Project**: eROSITA Light Curve Analysis - Event Detection & Classification
**Timeline**: January - May 2026
**Student**: Peter Dong

---

## Experimental Timeline

### Phase 1: Deep Learning Exploration (January - February)

**Location**: Root directory of repository
**Status**: Exploratory - not used in final results

**Approaches Tried**:
1. **RNN-based VAE** (Variational Autoencoder)
   - GRU encoder/decoder architecture
   - 9-feature input (3 bands × 3 values: RATE, ERRM, ERRP)
   - Loss: ELBO (Evidence Lower Bound) or Poisson NLL
   - Weights & Biases (wandb) integration for tracking
   - Scripts: `RNN_9_model.py`, `train_rnn.py`, `test_rnn.py`

2. **Transformer-based VAE**
   - Multi-head attention encoder
   - Positional encoding for temporal information
   - ResNet decoder blocks
   - Latent space outlier detection with Isolation Forest
   - Scripts: `trans_model.py`, `test_trans.py`

3. **LSTM Autoencoder**
   - Notebook experiments
   - Various latent space dimensions tested

**Findings**:
- Deep learning struggled with small number of time points (~10-20 per light curve)
- Latent space representations difficult to interpret astronomically
- Posterior collapse issues with VAEs
- Limited GPU availability on cluster

**Decision**: Moved to statistical feature extraction for better interpretability and astronomical relevance

---

### Phase 2: Statistical Feature Extraction (March - May)

**Location**: `pipeline/`
**Status**: Main work - results in final deliverables

---

## Run Number Timeline

### Early Experiments (100s Series)

**Period**: Early March
**Runs**: 100-108

**Parameters Explored**:
- `min_cluster_size`: 3, 5
- `cluster_selection_epsilon`: 0, 1, 3, 5
- `min_samples`: Very small (1-3)

**Purpose**: Initial HDBSCAN parameter exploration
**Outcome**: Parameters too permissive, many small/noisy clusters

---

### Systematic Tuning (200s Series)

**Period**: Mid-March
**Runs**: 200-223

**Fixed Parameters**:
- `min_cluster_size`: 3
- `min_samples`: 3

**Varied Parameters**:
- `cluster_selection_epsilon`: 0 to 0.5 (increments of 0.05-0.1)
- `cluster_selection_method`: 'eom' vs 'leaf'

**Important Runs**:

#### Run 217 ⭐
- **Parameters**: min_cluster=3, epsilon=0.1, leaf, min_samples=3
- **Notes**: Rerun of Run 215 (mentioned in meeting notes)
- **Status**: Important reference run
- **Outcome**: Good baseline for comparison

#### Run 223
- **Parameters**: Similar to 217 but with clipping
- **Notes**: **"Clipped version"** - removed excess_var outliers before clustering
- **Motivation**: excess_var feature had extreme outliers skewing results
- **Status**: Led to "clippedxvar" naming in later runs

---

### Refinement (230s Series)

**Period**: Late March - Early April
**Runs**: 231-243

**Motivation**: Slightly larger min_cluster_size to reduce number of tiny clusters

**Parameter Progression**:

#### Run 231
- min_cluster=5, epsilon=0.13, eom='eom', min_samples=3

#### Run 232
- min_cluster=7, epsilon=0.11, eom='eom', min_samples=5

#### Run 233
- min_cluster=7, epsilon=0.11, eom='leaf', min_samples=5
- Switch from 'eom' to 'leaf' method

#### Run 237 ⭐⭐⭐ MAIN FINAL RUN
- **Parameters**:
  - `min_cluster_size`: 7
  - `cluster_selection_epsilon`: 0.2 (increased from 0.11)
  - `cluster_selection_method`: 'leaf'
  - `min_samples`: 5
- **Notes**:
  - Based on Run 233 with epsilon adjustment
  - This is the **"real_clippedxvar"** run
  - Excess variance outliers clipped before clustering
  - Used for similarity analysis (cosine similarity lists)
- **Status**: **PRIMARY RESULT TO SHARE WITH SUPERVISORS**

#### Run 243
- Same parameters as Run 237 (verification run)

---

### Large Cluster Analysis (260s Series)

**Period**: Mid-April
**Runs**: 260-267

**Motivation**: Focus on larger, more significant clusters only

**Parameter Progression**:

#### Run 260
- min_cluster=20

#### Run 265
- min_cluster=25, min_samples=12

#### Run 266 ⭐
- **Parameters**: min_cluster=50, min_samples=25
- **Notes**: Used for generating sample cluster visualizations
- **Command**: `python sample_clusters.py --run 266 --samples 25`
- **Output**: Grid plots in `plots/all266/CLUSTERS/`

#### Run 267
- min_cluster=100, min_samples=50
- Too restrictive, very few clusters

---

### Final Experiments (300s Series)

**Period**: Late April - Early May
**Runs**: 301-311

**Notes from meeting notes are ambiguous**:
- Numbers listed: 99.7, 95, 68, 40, 30, 20
- These may refer to:
  - Cluster counts (not run parameters)
  - Percentile thresholds
  - Feature filtering experiments

#### Run 310
- Notes mention "30 without exvar"
- Possibly: 30 clusters when excluding excess_var feature

#### Run 311
- Notes mention "20 w/o exvar, min_cluster=50, min_samples=25"
- Similar to Run 266 but with feature filtering

**Status**: These runs were exploratory; Run 237 remains the primary result

---

## Special Dataset: 640 Light Curves

**Not a run number** - this is a sample dataset

**Purpose**: Generate feature distribution histograms for quick visualization

**Command**:
```bash
python bexvar_histograms.py \
  "/home/pdong/Astro UROP/pipeline/data/640/processedbatch/feature.pkl" \
  --outdir "/home/pdong/Astro UROP/pipeline/640features"
```

**Output**: 10 histogram PNG files in `640features/`
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

**Status**: ✅ Histograms present in organized repository

---

## Key Decisions Made

### Why Statistical Features Over Deep Learning?
1. **Interpretability**: Features have physical/statistical meaning
2. **Data limitations**: 10-20 time points per curve insufficient for deep learning
3. **Astronomical relevance**: Features align with known variability measures
4. **Computational efficiency**: Faster to compute than training neural networks

### Features Selected for Clustering
From `config.py`:
```python
SELECTED_FEATURES_FOR_CLUSTERING = [
    "weighted_mean",
    "weighted_variance",
    "lag1_autocorr",
    "hurst_exp",
    "mean_rise_fall_ratio",
    "stetson_k",
    "bexvar",
    "mean_var",
    "ampl_sig"
]
```

**Note**: `excess_var` and `beyond1std` were computed but not used in clustering due to extreme outliers

### Why HDBSCAN Over Other Clustering Methods?
1. **No need to specify k**: Unlike K-means, discovers number of clusters
2. **Density-based**: Finds clusters of varying shapes and densities
3. **Noise handling**: Identifies outliers (label=-1) rather than forcing assignment
4. **Hierarchical**: Can explore cluster structure at multiple scales

### Cosine Similarity Implementation
- Normalized features to unit vectors (RobustScaler)
- Computed pairwise cosine similarity: `cos(θ) = (A·B)/(||A|| ||B||)`
- Ranked all sources by similarity to 3 known interesting sources
- Extracted top 100 most similar for each
- **Output**: CSV files with similarity scores and cluster assignments

---

## Run Parameters Summary Table

| Run | min_cluster | epsilon | method | min_samples | Notes |
|-----|-------------|---------|--------|-------------|-------|
| 217 | 3 | 0.1 | leaf | 3 | Reference run (rerun of 215) |
| 223 | 3 | ~0.1 | leaf | 3 | Clipped excess_var outliers |
| 231 | 5 | 0.13 | eom | 3 | First larger min_cluster test |
| 232 | 7 | 0.11 | eom | 5 | Testing eom method |
| 233 | 7 | 0.11 | leaf | 5 | Switch to leaf method |
| **237** | **7** | **0.2** | **leaf** | **5** | **⭐ MAIN RUN - clippedxvar** |
| 243 | 7 | 0.2 | leaf | 5 | Verification of 237 |
| 260 | 20 | ? | ? | ? | Larger clusters only |
| 265 | 25 | ? | ? | 12 | Even larger |
| 266 | 50 | ? | ? | 25 | ⭐ Used for sample plots |
| 267 | 100 | ? | ? | 50 | Too restrictive |

---

## Known Issues Encountered

### Data Issues
1. **Inaccessible FITS files**: Some light curves couldn't be read (permissions/corruption)
   - Tracked in `inaccessible_lightcurves.txt`
   - Filtered out during loading

2. **Variable light curve lengths**: ~10-20 time points per source
   - Handled by truncating to max 20 points
   - Some features require minimum number of points

3. **Measurement errors**: Asymmetric errors (ERRM, ERRP)
   - Used symmetric approximation: `SYM_ERR = (ERRM + ERRP)/2` for some calculations

### Feature Computation Issues
1. **Skewness/Kurtosis**: Caused NaN errors with few time points
   - Removed from pipeline

2. **Excess variance outliers**: Extreme values in some sources
   - Solution: Clipped outliers (Run 223, 237)

3. **Hurst exponent**: Requires certain minimum time points
   - Some sources returned NaN, handled during clustering

### Computational Challenges
1. **GPU allocation**: Limited availability for deep learning experiments (Phase 1)

2. **Large dataset**: ~200,000 light curves
   - Solution: Parallel processing with SLURM array jobs
   - Scripts: `batch_feature_extraction.py`, `consolidate_features.py`

3. **Memory constraints**: Loading all features at once
   - Solution: Chunked processing where needed

---

## What Would Be Done Next (Future Work)

From meeting notes and discussions:

1. **Random Forest Analysis**
   - Train classifier to predict cluster assignment from features
   - Extract feature importance scores
   - Understand which features drive clustering

2. **Web Visualization Interface**
   - HDF5 files prepared (`features.h5`, `light_curves.h5`)
   - Interactive exploration of UMAP space
   - Click on point → view light curve

3. **Cross-matching with External Catalogs**
   - SIMBAD: Source identification
   - NED: Extragalactic sources
   - Gaia: Stellar sources
   - Identify physical source types per cluster

4. **Physical Interpretation of Clusters**
   - What do sources in each cluster have in common physically?
   - AGN vs stars vs X-ray binaries vs TDEs?

5. **Temporal Analysis**
   - Split light curves into time segments
   - Test if cluster assignment changes over time
   - Detect state changes in sources

6. **Multi-wavelength Follow-up**
   - Identify most interesting anomalous sources
   - Propose optical/radio follow-up observations

---

## Files Generated (To Be Downloaded from Cluster)

### From Run 237 (Main Run):
```
data/all/237/
├── hdbscan_data/
│   ├── cluster_assignments.csv
│   ├── cluster_probabilities.csv
│   └── outlier_scores.csv
├── umap_data/
│   ├── umap_embedding.csv
│   └── cluster_assignments.csv
└── web_data/
    ├── features.h5
    └── light_curves.h5
```

### From Similarity Analysis:
```
data/all/analysis_results/cluster_assignments_237_real_clippedxvar/
├── em01_211120_020_similar.csv
├── em01_039135_020_similar.csv
└── em01_038099_020_similar.csv
```

### From Run 266 (Visualizations):
```
plots/all266/CLUSTERS/
└── cluster_*_samples_*.png
```

---

## Reproducibility

All analysis can be reproduced using:
- Code in `pipeline/`
- Parameters documented in this file
- Data at `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned` (cluster)

**Main workflow**:
1. Feature extraction: `run_feature_extraction.py`
2. Clustering (Run 237): `run_pipeline_on_features.py --min-cluster 7 --epsilon 0.2 --min-samples 5 --run-number 237`
3. Similarity analysis: `analyze_similar_curves.py`
4. Visualizations: `sample_clusters.py --run 237 --samples 25`
