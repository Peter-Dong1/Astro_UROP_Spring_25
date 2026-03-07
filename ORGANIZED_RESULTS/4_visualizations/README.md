# Visualizations

This directory contains plots and visualizations generated during the analysis.

---

## run_266_cluster_samples/

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

Sample light curve plots from Run 266 (large cluster analysis).

**Content**:
- Grid plots showing 25 random sample light curves per cluster
- Format: `cluster_{ID}_samples_{timestamp}.png`

**How generated**:
```bash
python sample_clusters.py --run 266 --samples 25
```

**Expected structure**:
```
run_266_cluster_samples/
├── cluster_0_samples_0325_14.png
├── cluster_1_samples_0325_14.png
├── cluster_2_samples_0325_14.png
...
```

**Cluster Location**:
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/plots/all266/CLUSTERS/
```

---

## Visualization Types

### Cluster Sample Grids
- **Purpose**: Visual inspection of light curve patterns within each cluster
- **Layout**: 5×5 grid (25 samples per cluster)
- **Shows**: TIME vs RATE with error bars for each light curve

### UMAP Scatter Plots
**Status**: ⚠️ Can be generated after downloading cluster data

- **Purpose**: 2D visualization of high-dimensional feature space
- **Shows**: All sources plotted on UMAP coordinates, colored by cluster
- **File**: To be generated from `run_237_main/umap_data/umap_embedding.csv`

### Feature Histograms
✅ **Already present** in `../1_features/feature_histograms_640/`
- Distribution of each feature across 640-source sample
- 10 PNG files (one per feature)

---

## Generating Additional Visualizations

### Plot cluster samples for Run 237:
```bash
cd ../5_code
python pipeline/sample_clusters.py --run 237 --samples 25 --outdir ../4_visualizations/run_237_cluster_samples
```

### Create UMAP visualization:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
clusters = pd.read_csv('../2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')
umap = pd.read_csv('../2_clustering_runs/run_237_main/umap_data/umap_embedding.csv')

# Merge
data = clusters.merge(umap, on='file_path')

# Plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(data['umap_x'], data['umap_y'],
                     c=data['cluster_label'], cmap='tab20',
                     s=1, alpha=0.5)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('HDBSCAN Clusters - Run 237 (UMAP Projection)')
plt.grid(alpha=0.3)
plt.savefig('run_237_umap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Plot cluster size distribution:
```python
import pandas as pd
import matplotlib.pyplot as plt

clusters = pd.read_csv('../2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')

# Count cluster sizes
cluster_sizes = clusters['cluster_label'].value_counts().sort_index()
cluster_sizes = cluster_sizes[cluster_sizes.index >= 0]  # Exclude noise (-1)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(cluster_sizes)), cluster_sizes.values)
plt.xlabel('Cluster ID')
plt.ylabel('Number of Sources')
plt.title(f'Cluster Size Distribution - Run 237 (Total: {len(cluster_sizes)} clusters)')
plt.grid(axis='y', alpha=0.3)
plt.savefig('run_237_cluster_sizes.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Plot feature comparison across clusters:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load features and clusters
features = pd.read_pickle('../1_features/features.pkl')
clusters = pd.read_csv('../2_clustering_runs/run_237_main/hdbscan_data/cluster_assignments.csv')

# Merge
data = features.merge(clusters, on='file_path')

# Select top 5 largest clusters
top_clusters = data['cluster_label'].value_counts().head(5).index

# Plot feature distributions
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
feature_cols = ['weighted_mean', 'weighted_variance', 'lag1_autocorr',
                'hurst_exp', 'bexvar', 'stetson_k', 'mean_var',
                'mean_rise_fall_ratio', 'ampl_sig', 'excess_var']

for i, feature in enumerate(feature_cols):
    ax = axes[i // 5, i % 5]
    for cluster_id in top_clusters:
        cluster_data = data[data['cluster_label'] == cluster_id][feature]
        ax.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster_id}', bins=30)
    ax.set_title(feature)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions_by_cluster.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Expected Outputs After Full Analysis

```
4_visualizations/
├── run_266_cluster_samples/           # ⚠️ From cluster
│   └── cluster_*_samples_*.png
├── run_237_cluster_samples/           # To be generated
│   └── cluster_*_samples_*.png
├── run_237_umap.png                   # To be generated
├── run_237_cluster_sizes.png          # To be generated
└── feature_distributions_by_cluster.png  # To be generated
```

---

## Notes

- All visualization scripts are in `../5_code/pipeline/`
- Primary script: `plot_cluster_samples.py` or `sample_clusters.py`
- UMAP coordinates from Run 237 provide best 2D representation
- Cluster sample plots help validate clustering quality visually
