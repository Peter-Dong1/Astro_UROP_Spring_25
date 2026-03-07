# Data Dictionary

Complete description of all data files, features, and column definitions.

---

## Raw Data (FITS Files)

**Location (Cluster)**: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`

**Format**: FITS (Flexible Image Transport System) binary tables

**Naming Convention**:
```
em01_{OBSID}_{SRCNUM}_LightCurve_{INDEX}_c010_rebinned.fits
```

**Energy Bands**:
| Band | Energy Range | Index | Common Name |
|------|--------------|-------|-------------|
| Low | 0.2-0.6 keV | 0 | Soft X-ray |
| Medium | 0.6-2.3 keV | 1 | Medium X-ray |
| High | 2.3-5.0 keV | 2 | Hard X-ray |

**FITS Columns**:
- `TIME` - Observation time (MJD or mission time)
- `TIMEDEL` - Time bin width
- `RATE` - Count rate (counts/second) [3 values for 3 bands]
- `RATE_ERRM` - Negative error on rate [3 values]
- `RATE_ERRP` - Positive error on rate [3 values]

**Light Curve Processing**:
- Maximum length: 20 time points (truncated if longer)
- Typical length: 10-20 points
- Inaccessible files tracked in `inaccessible_lightcurves.txt`

---

## Features DataFrame (features.pkl)

**File**: `ORGANIZED_RESULTS/1_features/features.pkl`
**Format**: Pickled pandas DataFrame
**Size**: ~200,000 rows (one per source) × ~15 columns

### Metadata Columns

#### file_path
- **Type**: String
- **Description**: Full path to FITS file
- **Example**: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned/em01_211120_020_LightCurve_00007_c010_rebinned.fits`

#### file_name
- **Type**: String
- **Description**: FITS filename without path
- **Example**: `em01_211120_020_LightCurve_00007_c010_rebinned.fits`

#### light_curve
- **Type**: Nested pandas DataFrame
- **Description**: Raw light curve data
- **Columns**: `TIME`, `TIMEDEL`, `RATE`, `ERRM`, `ERRP`, `SYM_ERR`
- **Rows**: Variable (typically 10-20)

---

## Statistical Features

All features are computed **per energy band** (some features average across bands).

### Weighted Statistics

#### weighted_mean
- **Type**: Float
- **Unit**: Counts/second (or normalized)
- **Description**: Mean flux weighted by inverse measurement variance
- **Formula**:
  ```
  weighted_mean = Σ(rate_i / err_i²) / Σ(1 / err_i²)
  ```
- **Interpretation**: More precise measurements get higher weight
- **Range**: Varies by source brightness
- **Used in clustering**: ✅ Yes

#### weighted_variance
- **Type**: Float
- **Unit**: (Counts/second)²
- **Description**: Variance of flux weighted by inverse measurement variance
- **Formula**:
  ```
  weighted_var = Σ(w_i × (rate_i - weighted_mean)²) / Σ(w_i)
  where w_i = 1 / err_i²
  ```
- **Interpretation**: Intrinsic variability accounting for measurement precision
- **Range**: 0 to ∞ (higher = more variable)
- **Used in clustering**: ✅ Yes

#### weighted_median
- **Type**: Float
- **Unit**: Counts/second
- **Description**: Median flux weighted by inverse variance
- **Interpretation**: Robust central tendency
- **Used in clustering**: ❌ No

#### weighted_iqr
- **Type**: Float
- **Unit**: Counts/second
- **Description**: Interquartile range weighted by inverse variance
- **Interpretation**: Robust measure of spread
- **Used in clustering**: ❌ No

---

### Temporal Features

#### lag1_autocorr
- **Type**: Float
- **Range**: -1 to +1
- **Description**: Lag-1 autocorrelation - correlation between consecutive time points
- **Formula**:
  ```
  lag1_autocorr = corr(rate[:-1], rate[1:])
  ```
- **Interpretation**:
  - +1: Perfect positive correlation (smooth, trending)
  - 0: No correlation (random walk, white noise)
  - -1: Perfect negative correlation (oscillating)
- **Astronomical meaning**:
  - High positive: Persistent source (AGN, slow variability)
  - Near zero: Flickering/random (e.g., magnetar flares)
  - Negative: Periodic or quasi-periodic oscillations
- **Used in clustering**: ✅ Yes

#### hurst_exp
- **Type**: Float
- **Range**: 0 to 1
- **Description**: Hurst exponent - measures long-term memory in time series
- **Formula**: Computed via rescaled range (R/S) analysis
- **Interpretation**:
  - H < 0.5: Mean-reverting (tends to return to mean)
  - H = 0.5: Random walk (no memory)
  - H > 0.5: Persistent/trending (past behavior predicts future)
- **Astronomical meaning**:
  - H > 0.5: AGN with red noise power spectrum
  - H < 0.5: Rare (anti-persistent variability)
- **Used in clustering**: ✅ Yes

---

### Variability Measures

#### stetson_k
- **Type**: Float
- **Range**: ~0 to ∞ (typically 0-2)
- **Description**: Stetson K statistic - kurtosis-based variability measure
- **Formula**:
  ```
  δ_i = sqrt(N/(N-1)) × (rate_i - mean) / err_i
  stetson_k = (1/N) × Σ|δ_i| / sqrt(κ)
  where κ = (1/N) × Σ(δ_i²)
  ```
- **Interpretation**:
  - K ≈ 0.798: Gaussian variability
  - K < 0.798: Outlier-poor (concentrated distribution)
  - K > 0.798: Outlier-rich (heavy tails, extreme events)
- **Astronomical meaning**: Detects sources with flares or dips beyond normal noise
- **Used in clustering**: ✅ Yes

#### bexvar
- **Type**: Float
- **Range**: 0 to ∞
- **Description**: Bayesian excess variance - intrinsic variability beyond measurement noise
- **Formula**: Bayesian model comparing variable vs constant flux hypotheses
- **Interpretation**:
  - bexvar ≈ 0: Consistent with constant source
  - bexvar > 0: Significant intrinsic variability
  - Higher values = more variable
- **Astronomical meaning**: Gold standard for assessing true variability in X-ray sources
- **Implementation**: `bexvar_ero.py` (eROSITA-specific)
- **Used in clustering**: ✅ Yes

#### excess_var
- **Type**: Float
- **Range**: -∞ to ∞ (can be negative)
- **Description**: Classical excess variance
- **Formula**:
  ```
  excess_var = (S² - <σ²>) / <rate>²
  where S² = sample variance, <σ²> = mean squared error
  ```
- **Interpretation**:
  - excess_var ≈ 0: No variability beyond noise
  - excess_var > 0: Significant variability
  - excess_var < 0: Can occur due to measurement issues
- **Problem**: Can have extreme outliers with small error bars
- **Used in clustering**: ❌ No (removed in "clipped" runs due to outliers)

#### beyond1std
- **Type**: Float
- **Range**: 0 to 1
- **Description**: Fraction of data points beyond 1 standard deviation from mean
- **Formula**:
  ```
  beyond1std = (# points with |rate - mean| > std) / total_points
  ```
- **Interpretation**:
  - For Gaussian: ~0.32 expected
  - > 0.32: Heavy-tailed distribution
  - < 0.32: Concentrated near mean
- **Used in clustering**: ❌ No

---

### Shape Features

#### mean_rise_fall_ratio
- **Type**: Float
- **Range**: 0 to ∞
- **Description**: Ratio of mean rise rate to mean fall rate
- **Formula**:
  ```
  rises = rate[i+1] - rate[i] where rate[i+1] > rate[i]
  falls = rate[i] - rate[i+1] where rate[i+1] < rate[i]
  ratio = mean(rises) / mean(falls)
  ```
- **Interpretation**:
  - ratio = 1: Symmetric rises and falls
  - ratio > 1: Rises faster than it falls (FRED: Fast Rise, Exponential Decay)
  - ratio < 1: Falls faster than it rises
- **Astronomical meaning**:
  - FRED-like: Some flare events, TDEs
  - Opposite: Some eclipsing systems
- **Used in clustering**: ✅ Yes

#### mean_var
- **Type**: Float
- **Range**: 0 to ∞
- **Description**: Ratio of mean flux to variance
- **Formula**:
  ```
  mean_var = mean(rate) / var(rate)
  ```
- **Interpretation**:
  - High: Relatively constant source
  - Low: Highly variable relative to brightness
- **Astronomical meaning**: Characterizes fractional variability
- **Used in clustering**: ✅ Yes

#### ampl_sig
- **Type**: Float
- **Range**: 0 to ∞
- **Description**: Amplitude significance - peak-to-peak amplitude relative to errors
- **Formula**:
  ```
  amplitude = max(rate) - min(rate)
  mean_error = mean(errors)
  ampl_sig = amplitude / mean_error
  ```
- **Interpretation**:
  - Low: Amplitude within noise
  - High: Significant, real variability
- **Astronomical meaning**: Simple signal-to-noise for variability detection
- **Used in clustering**: ✅ Yes

---

## Cluster Assignment Files

### cluster_assignments.csv

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_{N}/hdbscan_data/cluster_assignments.csv`

**Columns**:

#### file_path
- **Type**: String
- **Description**: Full path to FITS file (matches features.pkl)

#### cluster_label
- **Type**: Integer
- **Range**: -1 to N_clusters-1
- **Description**: HDBSCAN cluster assignment
- **Values**:
  - -1: Noise/outlier (doesn't belong to any cluster)
  - 0, 1, 2, ...: Cluster ID
- **Interpretation**: Sources with same label have similar feature vectors

---

### cluster_probabilities.csv

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_{N}/hdbscan_data/cluster_probabilities.csv`

**Columns**:

#### file_path
- **Type**: String

#### cluster_probability
- **Type**: Float
- **Range**: 0 to 1
- **Description**: HDBSCAN membership probability
- **Interpretation**:
  - 1.0: Core cluster member
  - 0.5-1.0: Cluster member with some uncertainty
  - < 0.5: Borderline (may be noise)
  - 0.0: Definite noise/outlier

---

### outlier_scores.csv

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_{N}/hdbscan_data/outlier_scores.csv`

**Columns**:

#### file_path
- **Type**: String

#### outlier_score
- **Type**: Float
- **Range**: Typically 0 to 1 (can exceed 1)
- **Description**: HDBSCAN outlier score (GLOSH: Global-Local Outlier Score from Hierarchies)
- **Interpretation**:
  - Low score: Typical member of cluster
  - High score: Outlier within cluster or global outlier
  - Use for ranking most anomalous sources

---

## UMAP Embedding Files

### umap_embedding.csv

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_{N}/umap_data/umap_embedding.csv`

**Columns**:

#### file_path
- **Type**: String

#### umap_x
- **Type**: Float
- **Description**: First UMAP coordinate
- **Range**: Arbitrary (centered near 0)

#### umap_y
- **Type**: Float
- **Description**: Second UMAP coordinate
- **Range**: Arbitrary (centered near 0)

**Purpose**: 2D coordinates for visualizing high-dimensional feature space

**Parameters used**:
- `n_neighbors`: 15
- `min_dist`: 0.1
- `n_components`: 2

---

## Cosine Similarity Files ⭐⭐⭐

### em01_*_similar.csv

**Location**: `ORGANIZED_RESULTS/3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/`

**Files**:
1. `em01_211120_020_similar.csv` - Similar to source 1
2. `em01_039135_020_similar.csv` - Similar to source 2
3. `em01_038099_020_similar.csv` - Similar to source 3

**Columns**:

#### rank
- **Type**: Integer
- **Range**: 1 to 100
- **Description**: Similarity rank (1 = most similar)

#### file_name
- **Type**: String
- **Description**: FITS filename

#### file_path
- **Type**: String
- **Description**: Full path to FITS file

#### cosine_similarity
- **Type**: Float
- **Range**: 0 to 1
- **Description**: Cosine similarity score
- **Formula**:
  ```
  similarity = (A · B) / (||A|| × ||B||)
  ```
- **Interpretation**:
  - 1.0: Identical feature vectors
  - 0.9-1.0: Very similar
  - 0.7-0.9: Moderately similar
  - < 0.7: Increasingly different

#### cluster_label
- **Type**: Integer
- **Description**: Cluster assignment from Run 237

#### Feature columns
- All features from features.pkl for the similar source
- Examples: `weighted_mean`, `weighted_variance`, `bexvar`, `lag1_autocorr`, etc.

---

## HDF5 Web Files (Optional)

### features.h5

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_237_main/web_data/features.h5`

**Format**: HDF5 hierarchical data format
**Purpose**: Fast access for web visualization
**Contents**: Features table optimized for web interface

### light_curves.h5

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_237_main/web_data/light_curves.h5`

**Format**: HDF5
**Purpose**: Fast light curve retrieval for interactive plots
**Contents**: All light curve data indexed by file_path

---

## Summary Files (Generated)

### feature_summary_statistics.csv

**Location**: `ORGANIZED_RESULTS/1_features/feature_summary_statistics.csv`

**Format**: pandas describe() output
**Rows**: count, mean, std, min, 25%, 50%, 75%, max
**Columns**: All feature columns

### cluster_summary.csv

**Location**: `ORGANIZED_RESULTS/2_clustering_runs/run_237_main/cluster_summary.csv`

**Columns**:
- `cluster_label`: Cluster ID
- `count`: Number of sources in cluster

---

## Data Loading Examples

### Load features:
```python
import pandas as pd
features = pd.read_pickle('ORGANIZED_RESULTS/1_features/features.pkl')
```

### Load clusters:
```python
clusters = pd.read_csv('ORGANIZED_RESULTS/2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')
```

### Load similarity:
```python
similar = pd.read_csv('ORGANIZED_RESULTS/3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/em01_211120_020_similar.csv')
```

### Merge features with clusters:
```python
data = features.merge(clusters, on='file_path')
```
