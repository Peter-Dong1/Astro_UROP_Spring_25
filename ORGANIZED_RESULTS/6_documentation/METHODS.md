# Methodology

**Project**: eROSITA Light Curve Analysis
**Approach**: Statistical feature extraction + HDBSCAN clustering + cosine similarity ranking

---

## Overview

This analysis uses unsupervised machine learning to identify groups of similar X-ray sources and rank sources by similarity to known interesting objects. The pipeline consists of:

1. Statistical feature extraction from light curves
2. Feature normalization (RobustScaler)
3. HDBSCAN clustering in feature space
4. UMAP dimensionality reduction for visualization
5. Cosine similarity computation for ranking sources

---

## Data Source

### eRASS1 (eROSITA All-Sky Survey 1)

**Survey**: First all-sky X-ray survey by the eROSITA telescope aboard Spektr-RG

**Data characteristics**:
- ~200,000 X-ray light curves
- 3 energy bands: Low (0.2-0.6 keV), Medium (0.6-2.3 keV), High (2.3-5.0 keV)
- ~10-20 time points per source (sparse sampling)
- Format: FITS binary tables
- Location: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`

**Light curve properties**:
- Each point: (TIME, RATE, RATE_ERRM, RATE_ERRP)
- Asymmetric Poisson errors (negative and positive)
- Variable-length sequences (truncated to max 20 points)
- Some sources inaccessible (permissions/corruption) - tracked and filtered

---

## Pipeline

### 1. Feature Extraction

**Script**: `run_feature_extraction.py`

**Purpose**: Convert sparse, variable-length light curves into fixed-length feature vectors suitable for clustering.

**Process**:
1. Load FITS light curve (TIME, RATE, ERRORS)
2. Compute 10 statistical features per light curve
3. Save as pandas DataFrame (features.pkl)

**Features extracted**:

| Feature | Type | Description |
|---------|------|-------------|
| weighted_mean | Flux level | Mean flux weighted by inverse variance |
| weighted_variance | Flux level | Variance weighted by inverse variance |
| lag1_autocorr | Temporal | Correlation between consecutive points |
| hurst_exp | Temporal | Persistence measure (H>0.5 = trending) |
| mean_rise_fall_ratio | Shape | Asymmetry of rises vs falls |
| stetson_k | Variability | Kurtosis-based variability index |
| bexvar | Variability | Bayesian excess variance |
| mean_var | Variability | Ratio of mean to variance |
| ampl_sig | Variability | Amplitude relative to errors |
| excess_var* | Variability | Classical excess variance (not used in clustering) |

*Note: `excess_var` and `beyond1std` computed but excluded from clustering due to extreme outliers in some sources.

**Implementation**:
- Uses `light_curve` Python package for advanced features (hurst_exp, stetson_k)
- Custom implementations for bexvar (`bexvar_ero.py`)
- Error propagation for weighted statistics
- Handles variable-length sequences
- Parallel processing via SLURM array jobs for scalability

**Output**: `features.pkl` - DataFrame with ~200k rows × 15 columns

---

### 2. Feature Normalization

**Method**: RobustScaler (from scikit-learn)

**Why RobustScaler?**
- Robust to outliers (uses median and IQR instead of mean and std)
- Scales features to comparable ranges
- Preserves relative distances better than StandardScaler when outliers present

**Formula**:
```
X_scaled = (X - median(X)) / IQR(X)
```

**Applied in**: `run_pipeline_on_features.py` before clustering

---

### 3. HDBSCAN Clustering

**Script**: `run_pipeline_on_features.py`

**Algorithm**: HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

**Why HDBSCAN?**
1. **Density-based**: Finds clusters of varying shapes and densities (unlike K-means)
2. **Hierarchical**: Builds hierarchy of clusters, selects optimal level
3. **Noise handling**: Identifies outliers (label=-1) rather than forcing assignment
4. **No k specification**: Automatically determines number of clusters
5. **Robust**: More stable than DBSCAN to parameter choices

**Key Parameters** (Run 237 - Main Result):
- `min_cluster_size`: 7
  - Minimum number of points to form a cluster
  - Smaller = more granular clusters
  - Too small = overfitting to noise

- `cluster_selection_epsilon`: 0.2
  - Distance threshold for merging clusters
  - Larger = more merging = fewer, larger clusters
  - Units: Euclidean distance in normalized feature space

- `cluster_selection_method`: 'leaf'
  - 'leaf': Bottom-up selection (prefers fine-grained clusters)
  - 'eom': Excess of Mass (prefers stable clusters across scales)

- `min_samples`: 5
  - Neighborhood size for determining core points
  - Higher = more conservative clustering

**Distance metric**: Euclidean distance
- Applied to normalized feature vectors
- Relationship to cosine similarity (for unit vectors):
  ```
  ||x-y||² = 2(1 - cos(x,y))
  ```

**Algorithm steps** (simplified):
1. Construct mutual reachability graph
2. Build minimum spanning tree
3. Extract cluster hierarchy
4. Condense tree by merging small clusters
5. Select optimal clusters using stability measure
6. Assign noise label (-1) to non-core points

**Outputs**:
- `cluster_assignments.csv` - Cluster label per source
- `cluster_probabilities.csv` - Membership confidence [0,1]
- `outlier_scores.csv` - GLOSH outlier scores

**Parameter evolution**:
- Early runs (217): min_cluster=3, epsilon=0.1 (many small clusters)
- **Run 237**: min_cluster=7, epsilon=0.2 (balanced, used for final results)
- Run 266: min_cluster=50, epsilon=? (only large clusters)

---

### 4. UMAP Dimensionality Reduction

**Purpose**: Reduce 9-dimensional feature space to 2D for visualization while preserving structure

**Algorithm**: UMAP (Uniform Manifold Approximation and Projection)

**Why UMAP over PCA?**
- Preserves both local and global structure
- Non-linear dimensionality reduction
- Better captures cluster boundaries
- More interpretable visual separation

**Parameters**:
- `n_neighbors`: 15
  - Size of local neighborhood
  - Larger = more global structure preserved
  - Smaller = more local detail

- `min_dist`: 0.1
  - Minimum distance between points in embedding
  - Smaller = tighter clusters
  - Larger = more spread out

- `n_components`: 2
  - Output dimensionality (for 2D plots)

**Algorithm** (simplified):
1. Construct weighted k-nearest neighbor graph in high-dimensional space
2. Optimize low-dimensional layout to preserve graph structure
3. Use stochastic gradient descent with repulsive forces

**Output**:
- `umap_embedding.csv` - (file_path, umap_x, umap_y) for all sources

**Usage**: Visualization only - does not affect clustering (HDBSCAN operates on original 9D features)

---

### 5. Cosine Similarity Analysis ⭐⭐⭐

**Script**: `analyze_similar_curves.py`

**Purpose**: For each known interesting source, find top 100 most similar sources

**Method**:

1. **Load features** from `features.pkl`

2. **Normalize feature vectors** to unit length:
   ```
   v_norm = v / ||v||
   ```

3. **Compute pairwise cosine similarity**:
   ```
   similarity(A, B) = cos(θ) = (A · B) / (||A|| × ||B||)
   ```
   - For unit vectors: `similarity = A · B` (dot product)
   - Range: -1 to +1 (typically 0 to 1 for non-negative features)
   - 1 = identical, 0 = orthogonal, -1 = opposite

4. **For each known source**:
   - Compute similarity to all ~200k sources
   - Rank by descending similarity
   - Extract top 100

5. **Merge with cluster assignments** from Run 237

6. **Save to CSV**:
   - `em01_211120_020_similar.csv` (source 1)
   - `em01_039135_020_similar.csv` (source 2)
   - `em01_038099_020_similar.csv` (source 3)

**Known sources** (from `config.py`):
- `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
- `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
- `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

**Output format**:
```csv
rank, file_name, file_path, cosine_similarity, cluster_label, [features...]
1, em01_123456..., /path/..., 0.987, 5, 0.034, 0.123, ...
2, em01_234567..., /path/..., 0.976, 5, 0.029, 0.118, ...
...
```

**Computational note**:
- Pairwise similarity: O(N²) for N sources
- Optimized with vectorized operations (numpy/scipy)
- Can be computed efficiently on normalized features

---

### 6. Visualization

**Scripts**: `sample_clusters.py`, `plot_cluster_samples.py`

**Types of visualizations**:

1. **Cluster sample grids**:
   - 5×5 grid (25 random samples per cluster)
   - Light curves: TIME vs RATE with error bars
   - Purpose: Visual validation of clustering quality
   - Generated for Run 266 (large clusters)

2. **UMAP scatter plots**:
   - All sources plotted on (umap_x, umap_y)
   - Colored by cluster label
   - Purpose: Visualize cluster separation in 2D

3. **Feature histograms**:
   - Distribution of each feature across dataset
   - Generated from 640-source sample
   - Purpose: Understand feature ranges and distributions

**Typical workflow**:
```bash
# Generate cluster samples
python sample_clusters.py --run 237 --samples 25

# Plot specific cluster
python plot_cluster_samples.py --cluster 5 --num-samples 25
```

---

## Clustering Performance

### From Run 237 (Main Result):

**Parameters**: min_cluster=7, epsilon=0.2, leaf, min_samples=5

**Results** (to be filled after downloading from cluster):
- Number of clusters: [TO BE FILLED]
- Number of noise points (label=-1): [TO BE FILLED]
- Largest cluster size: [TO BE FILLED]
- Smallest cluster size: 7 (by definition)

**Cluster size distribution**: See `cluster_summary.csv`

---

## Validation

### Clustering Quality Assessment:

1. **Visual inspection**:
   - Sample light curves from each cluster reviewed
   - Check for coherent patterns within clusters

2. **Known sources**:
   - Track cluster assignments of 3 known interesting sources
   - Check if similar sources end up in same clusters

3. **Silhouette analysis** (optional):
   - Could compute silhouette scores for cluster quality
   - Not performed in current analysis

4. **Feature importance**:
   - Could use Random Forest to identify which features drive clustering
   - Mentioned in meeting notes as future work

### Similarity Analysis Validation:

1. **Transitivity check**:
   - If A is similar to B, and B is similar to C, is A similar to C?

2. **Cluster coherence**:
   - Do top 100 similar sources belong to same clusters?
   - High percentage in same cluster = clusters capture similarity well

---

## Software & Dependencies

### Python Environment

**Python version**: 3.9

**Core packages**:
- `pandas` (1.x) - DataFrame operations
- `numpy` (1.x) - Numerical computations
- `scipy` (1.x) - Statistical functions

**Machine learning**:
- `scikit-learn` (1.x) - RobustScaler, Isolation Forest
- `hdbscan` (0.8+) - HDBSCAN clustering
- `umap-learn` (0.5+) - UMAP dimensionality reduction

**Astronomy**:
- `astropy` (5.x) - FITS file I/O
- `light_curve` - Time series feature extraction

**Visualization**:
- `matplotlib` (3.x) - Plotting
- `seaborn` (0.11+) - Statistical visualizations

**Optional**:
- `h5py` - HDF5 output for web visualization
- `wandb` - Experiment tracking (deep learning phase)

**Installation**:
```bash
conda create -n myenv python=3.9
conda activate myenv
pip install pandas numpy scipy scikit-learn hdbscan umap-learn
pip install astropy matplotlib seaborn
pip install light_curve  # Time series features
pip install h5py  # Optional
```

---

## Computational Resources

**Cluster**: MIT Engaging (Supercloud)

**SLURM configuration**:
- Partition: `mit_normal_gpu` (misleading name - used CPUs only)
- Typical resources: 4-8 CPUs, 32-64 GB RAM
- Conda environment: `myenv`

**Processing times** (estimated):
- Feature extraction (full dataset): ~4-8 hours (parallelized)
- HDBSCAN clustering: ~10-30 minutes (depends on parameters)
- UMAP computation: ~5-10 minutes
- Cosine similarity: ~5-20 minutes (depends on implementation)

**Parallelization**:
- Feature extraction: SLURM array jobs (100 jobs processing chunks)
- Clustering: Single job (HDBSCAN not easily parallelizable)
- Similarity: Can be parallelized per known source

---

## Reproducibility

### Full pipeline from scratch:

```bash
# 1. Feature extraction (on cluster, parallel)
sbatch --array=0-99 get_features.slurm

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
python bexvar_histograms.py "features.pkl" --outdir "histograms"
```

### Required inputs:
- eRASS1 light curve FITS files (on cluster)
- `config.py` with paths and parameters
- `helper.py` for data loading

### Random seeds:
- UMAP: Set random_state for reproducibility
- Cluster sampling: Numpy random seed set in scripts
- HDBSCAN: Deterministic (no randomness)

---

## Limitations & Caveats

### Data limitations:
1. **Sparse sampling**: 10-20 points per light curve
   - Limits temporal resolution
   - Some features (lag1_autocorr, hurst_exp) require minimum points

2. **Variable lengths**: Light curves have different numbers of points
   - Handled via feature extraction (not direct sequence comparison)

3. **Measurement errors**: Asymmetric Poisson errors
   - Accounted for in weighted statistics
   - Symmetric approximation used where needed

### Methodological limitations:
1. **Feature selection**: 9 features used - other features possible
   - Could explore additional time series features
   - Could use wavelet or Fourier features

2. **Clustering parameters**: Single parameter set used (Run 237)
   - Explored others, but could explore more systematically
   - Grid search or Bayesian optimization possible

3. **Distance metric**: Euclidean in normalized space
   - Other metrics possible (Manhattan, Mahalanobis)
   - Cosine similarity used post-hoc, not for clustering

4. **No ground truth**: Unsupervised learning
   - No labeled data for validation
   - Physical interpretation required to assess quality

---

## Future Directions

From meeting notes and discussions:

1. **Feature importance**: Random Forest classifier on cluster labels
2. **Web interface**: Interactive exploration using HDF5 files
3. **Cross-matching**: External catalogs (SIMBAD, NED, Gaia)
4. **Physical interpretation**: Map clusters to source types
5. **Temporal evolution**: Split light curves, test cluster stability
6. **Multi-wavelength**: Incorporate optical/radio data

---

## References

**Algorithms**:
- HDBSCAN: Campello, Moulavi, Sander (2013) - "Density-Based Clustering Based on Hierarchical Density Estimates"
- UMAP: McInnes, Healy, Melville (2018) - "UMAP: Uniform Manifold Approximation and Projection"
- Bayesian excess variance: Multiple astronomy papers

**Software**:
- HDBSCAN Python package: https://github.com/scikit-learn-contrib/hdbscan
- UMAP Python package: https://github.com/lmcinnes/umap

**Data**:
- eROSITA: Predehl et al. (2021) - "The eROSITA X-ray telescope on SRG"
- eRASS1: Merloni et al. (2024) - "The SRG/eROSITA All-Sky Survey"
