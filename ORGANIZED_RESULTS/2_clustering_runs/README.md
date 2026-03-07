# Clustering Runs

All runs use **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) with **UMAP** dimensionality reduction for visualization.

---

## Run 217 (Important Reference Run)

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

**Date**: Mid-semester
**Parameters**:
- `min_cluster_size`: 3
- `cluster_selection_epsilon`: 0.1
- `cluster_selection_method`: 'leaf'
- `min_samples`: 3

**Notes**:
- This was a rerun of Run 215 mentioned specifically in meeting notes
- Important reference for comparison with later runs

**Expected files** (in `run_217_important/`):
```
hdbscan_data/
├── cluster_assignments.csv      # Cluster label per source
├── cluster_probabilities.csv    # Membership probabilities
└── outlier_scores.csv           # HDBSCAN outlier scores
umap_data/
└── umap_embedding.csv           # 2D UMAP coordinates
```

---

## Run 237 (MAIN FINAL RUN) ⭐⭐⭐

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

**Date**: Late semester
**Parameters**:
- `min_cluster_size`: 7 (from run 233)
- `cluster_selection_epsilon`: 0.2
- `cluster_selection_method`: 'leaf'
- `min_samples`: 5

**Notes**:
- **This is the "real_clippedxvar" run** mentioned in analysis_results folders
- Excess variance outliers were clipped before clustering
- **This is the primary clustering result to share with supervisors**
- Based on Run 233 with epsilon adjustment from 0.11 to 0.2

**Expected files** (in `run_237_main/`):
```
hdbscan_data/
├── cluster_assignments.csv      # ⭐ Main cluster labels
├── cluster_probabilities.csv    # Membership probabilities
└── outlier_scores.csv           # Outlier scores
umap_data/
├── umap_embedding.csv           # ⭐ 2D visualization coordinates
└── cluster_assignments.csv      # Clusters with UMAP coords
web_data/
├── features.h5                  # HDF5 features for web visualization
└── light_curves.h5              # HDF5 light curves for web visualization
cluster_summary.csv              # Generated summary statistics
```

---

## Run 266 (Large Cluster Analysis)

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

**Date**: Late semester
**Parameters**:
- `min_cluster_size`: 50
- `min_samples`: 25

**Notes**:
- Designed to find larger, more significant clusters only
- Used for generating sample cluster visualizations
- Sample plots command: `python sample_clusters.py --run 266 --samples 25`

**Expected files** (in `run_266_large_clusters/`):
```
hdbscan_data/
└── cluster_assignments.csv
```

Sample plots should be in: `../4_visualizations/run_266_cluster_samples/`

---

## HDBSCAN Algorithm

**Why HDBSCAN?**
- Density-based clustering finds clusters of varying shapes and densities
- Automatically identifies noise/outlier points (cluster label = -1)
- No need to pre-specify number of clusters (unlike K-means)
- More robust than standard DBSCAN

**Key Parameters**:
- `min_cluster_size`: Minimum number of points to form a cluster
- `cluster_selection_epsilon`: Distance threshold for merging clusters
- `cluster_selection_method`: 'leaf' = bottom-up, 'eom' = Excess of Mass
- `min_samples`: Neighborhood size for determining core points

**Distance Metric**: Euclidean distance on normalized features
- Features normalized with RobustScaler (robust to outliers)
- Euclidean distance relates to cosine similarity: ||x-y||² = 2(1-cos(x,y))

---

## UMAP Dimensionality Reduction

**Purpose**: Reduce high-dimensional feature space (9-10 features) to 2D for visualization

**Parameters**:
- `n_neighbors`: 15
- `min_dist`: 0.1
- `n_components`: 2

**Advantages over PCA**:
- Preserves both local and global structure
- Better for non-linear manifolds
- More interpretable clusters

---

## Cluster File Formats

### cluster_assignments.csv
```csv
file_path,cluster_label
/path/to/em01_xxx_LightCurve_xxx.fits,5
/path/to/em01_yyy_LightCurve_yyy.fits,-1
...
```
- `cluster_label = -1` indicates noise/outlier
- `cluster_label >= 0` indicates cluster membership

### umap_embedding.csv
```csv
file_path,umap_x,umap_y
/path/to/em01_xxx_LightCurve_xxx.fits,2.35,-1.42
...
```
- 2D coordinates for scatter plot visualization

---

## Loading Cluster Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load main clustering (Run 237)
clusters = pd.read_csv('run_237_main/hdbscan_data/cluster_assignments.csv')
umap_coords = pd.read_csv('run_237_main/umap_data/umap_embedding.csv')

# Cluster statistics
n_clusters = (clusters['cluster_label'] >= 0).nunique()
n_noise = (clusters['cluster_label'] == -1).sum()

print(f"Found {n_clusters} clusters")
print(f"Noise points: {n_noise}")
print(f"Cluster sizes:\n{clusters['cluster_label'].value_counts()}")

# Visualize UMAP with clusters
merged = clusters.merge(umap_coords, on='file_path')
plt.scatter(merged['umap_x'], merged['umap_y'],
           c=merged['cluster_label'], cmap='tab20', s=1, alpha=0.5)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('HDBSCAN Clusters (Run 237)')
plt.colorbar(label='Cluster')
plt.show()
```

---

## Evolution Across Runs

| Run | min_cluster | epsilon | min_samples | Notes |
|-----|-------------|---------|-------------|-------|
| 217 | 3 | 0.1 | 3 | Reference run (rerun of 215) |
| 237 | 7 | 0.2 | 5 | **MAIN RUN** - clipped excess_var |
| 266 | 50 | ? | 25 | Large clusters only |

**Key insight**: Run 237 uses larger min_cluster_size (7) and larger epsilon (0.2) compared to Run 217, resulting in fewer but more significant clusters.
