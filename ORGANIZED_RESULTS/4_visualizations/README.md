# 4_visualizations/

No visualization files are stored locally. See below for what exists on the cluster
and what's available in this repo.

---

## Available locally

**Feature histograms** (640-source sample) — in `../1_features/feature_histograms_640/`
- 10 PNG files, one histogram per feature

---

## On the cluster

**Run 266 cluster sample grids**
```
/home/pdong/Astro UROP/pipeline/plots/all266/CLUSTERS/
```
5×5 grids of 25 random light curves per cluster. Generated with:
```bash
python pipeline/scripts/sample_clusters.py --run 266 --samples 25
```

**Run 237 HDBSCAN/UMAP plots** — outlier grids, UMAP scatter colored by cluster and
by each feature, correlation matrix, known-curve comparisons, similarity histograms.
```
/home/pdong/Astro UROP/pipeline/plots/all<number>/
```
