# Cosine Similarity Analysis

**⭐⭐⭐ THIS IS WHAT DAN AND YOUR SUPERVISORS SPECIFICALLY REQUESTED ⭐⭐⭐**

> *"Dan and I were wondering if you could share with us... these lists of sources with cosine similarity"*

---

## Overview

For each of the 3 known interesting light curves, this analysis finds the **top 100 most similar sources** using cosine similarity on feature vectors.

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

---

## Known Source Light Curves

From `config.py`:
1. `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
2. `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
3. `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

---

## Expected Directory Structure

```
analysis_results/
├── cluster_assignments_237_real_clippedxvar/    # ⭐ MAIN DELIVERABLE
│   ├── em01_211120_020_similar.csv             # Top 100 similar to source 1
│   ├── em01_039135_020_similar.csv             # Top 100 similar to source 2
│   └── em01_038099_020_similar.csv             # Top 100 similar to source 3
│
├── feature_histograms_237_real_clippedxvar/
│   └── [Feature distribution plots for Run 237]
│
└── sig_nev_analysis/
    ├── SIG_NEV_mappings.pkl                    # Source ID cross-reference
    └── [Similar source lists with SIG/NEV IDs]
```

**Cluster Location**:
```
/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/analysis_results/
```

---

## CSV File Format

Each similarity CSV should contain:

**Expected columns**:
- `rank` - Similarity rank (1 = most similar)
- `file_name` - FITS filename
- `file_path` - Full path to light curve
- `cosine_similarity` - Similarity score [0, 1] where 1 = identical
- `cluster_label` - Which HDBSCAN cluster this source belongs to (from Run 237)
- Feature values: `weighted_mean`, `weighted_variance`, `bexvar`, `lag1_autocorr`, etc.

**Example**:
```csv
rank,file_name,file_path,cosine_similarity,cluster_label,weighted_mean,bexvar,...
1,em01_123456_020_LightCurve_00123_c010_rebinned.fits,/pool001/.../em01_123456...,0.987,5,0.034,0.123,...
2,em01_234567_020_LightCurve_00234_c010_rebinned.fits,/pool001/.../em01_234567...,0.976,5,0.029,0.118,...
...
```

---

## How Similarity Was Computed

**Script**: `analyze_similar_curves.py`

**Method**:
1. Load features from `features.pkl`
2. Normalize feature vectors to unit length (RobustScaler)
3. Compute pairwise cosine similarity:
   ```
   similarity = cos(θ) = (A · B) / (||A|| × ||B||)
   ```
   where A and B are feature vectors
4. For each known source, rank all other sources by similarity
5. Extract top 100 most similar
6. Merge with cluster assignments from Run 237
7. Save to CSV

**Features used for similarity**:
- weighted_mean, weighted_variance
- lag1_autocorr, hurst_exp
- mean_rise_fall_ratio, stetson_k
- bexvar, mean_var, ampl_sig

**Note on Distance vs Similarity**:
- HDBSCAN used Euclidean distance: `d(x,y) = ||x-y||`
- Relationship to cosine similarity: `||x-y||² = 2(1 - cos(x,y))`
- For unit-normalized vectors, Euclidean distance inversely relates to cosine similarity

---

## Using the Results

### Load similarity results:
```python
import pandas as pd

# Load top 100 similar sources for first known source
similar_1 = pd.read_csv(
    'analysis_results/cluster_assignments_237_real_clippedxvar/em01_211120_020_similar.csv'
)

print(f"Top similar source: {similar_1.iloc[0]['file_name']}")
print(f"Similarity score: {similar_1.iloc[0]['cosine_similarity']:.4f}")
print(f"Belongs to cluster: {similar_1.iloc[0]['cluster_label']}")

# Show top 10
print("\nTop 10 most similar sources:")
print(similar_1[['rank', 'file_name', 'cosine_similarity', 'cluster_label']].head(10))
```

### Analyze cluster distribution:
```python
# What clusters do similar sources belong to?
cluster_dist = similar_1['cluster_label'].value_counts()
print("Similar sources by cluster:")
print(cluster_dist)

# How many similar sources are in the same cluster as the known source?
known_cluster = similar_1.iloc[0]['cluster_label']
same_cluster = (similar_1['cluster_label'] == known_cluster).sum()
print(f"\nSources in same cluster as known source: {same_cluster}/100")
```

### Compare feature distributions:
```python
import matplotlib.pyplot as plt

# Compare bexvar distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, source_file in enumerate(['em01_211120_020_similar.csv',
                                  'em01_039135_020_similar.csv',
                                  'em01_038099_020_similar.csv']):
    df = pd.read_csv(f'analysis_results/cluster_assignments_237_real_clippedxvar/{source_file}')
    axes[i].hist(df['bexvar'], bins=20, alpha=0.7)
    axes[i].set_title(f'Source {i+1} - bexvar distribution')
    axes[i].set_xlabel('bexvar')
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()
```

---

## Validation Checks

After downloading, verify:
- [ ] 3 CSV files exist (one per known source)
- [ ] Each has exactly 100 rows (top 100 similar)
- [ ] Similarity scores are between 0 and 1
- [ ] Similarity scores are in descending order (rank 1 = highest)
- [ ] Cluster labels are present (from Run 237)
- [ ] Feature values are present

---

## If Files Don't Exist

### Regenerate on cluster:
```bash
ssh pdong@engaging.mit.edu
cd "/home/pdong/Astro UROP/z New Feature Extraction Pipeline"

# Run similarity analysis
python analyze_similar_curves.py

# Check output
ls -lh data/all/analysis_results/cluster_assignments_237_real_clippedxvar/
```

### Check script configuration:
The script should use:
- Features file: `data/all/amp_max_features/features.pkl`
- Cluster assignments: Run 237 results
- Known sources from `config.py`

---

## Scientific Interpretation

**Why these sources?**
- High cosine similarity means similar patterns across all features
- Sources in same cluster likely share physical characteristics
- Could indicate:
  - Similar source types (AGN, stars, X-ray binaries, etc.)
  - Similar variability patterns
  - Similar spectral properties (if energy bands differ)

**Next steps for supervisors**:
1. Cross-match with external catalogs (SIMBAD, NED)
2. Identify source types
3. Look for novel/unexpected similar sources
4. Physical interpretation of clusters containing similar sources
