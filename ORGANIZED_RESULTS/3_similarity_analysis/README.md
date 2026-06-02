# 3_similarity_analysis/

Cosine similarity analysis results — requested by supervisors. Output files are on the cluster.

---

## What was done

For each of the 3 known interesting light curves, the top 1000 most similar sources were
identified using cosine similarity on the 9 clustering features, then merged with Run 237
cluster assignments.

**Known sources** (from `pipeline/config.py`):
1. `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
2. `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
3. `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

**Method**: Features normalized to unit vectors (RobustScaler), then pairwise cosine
similarity computed. For unit vectors, cosine similarity and Euclidean distance are
related by `||x-y||² = 2(1 - cos(x,y))`.

**Script**: `pipeline/scripts/analyze_similar_curves.py`

---

## Output location (cluster)

```
/home/pdong/Astro UROP/pipeline/data/all/analysis_results/cluster_assignments_237_real_clippedxvar/
```

Each CSV contains: `rank`, `file_name`, `file_path`, `cosine_similarity`, `cluster_label`,
and all feature values.
