# 2_clustering_runs/

HDBSCAN clustering results. All output files are on the cluster — nothing is stored locally here.
See `../6_documentation/RUN_HISTORY.md` for the full experimental timeline.

---

## Run 217 — Reference Run

**Parameters**: `min_cluster_size=3`, `epsilon=0.1`, `method=leaf`, `min_samples=3`

Rerun of Run 215 (mentioned in meeting notes). Used as a baseline for comparison with later runs.

**Cluster location**: `/home/pdong/Astro UROP/pipeline/data/all/217/`

---

## Run 237 — Main Final Run ⭐

**Parameters**: `min_cluster_size=7`, `epsilon=0.2`, `method=leaf`, `min_samples=5`

The primary clustering result. Excess variance outliers were clipped before clustering
(the "real_clippedxvar" run). This is the run used for the cosine similarity deliverable
shared with supervisors.

**Cluster location**: `/home/pdong/Astro UROP/pipeline/data/all/237/`

---

## Run 266 — Large Cluster Analysis

**Parameters**: `min_cluster_size=50`, `min_samples=25`

Designed to surface only large, significant clusters. Used for generating the cluster
sample grid plots.

**Cluster location**: `/home/pdong/Astro UROP/pipeline/data/all/266/`

---

## Run Parameters Summary

| Run | min_cluster | epsilon | method | min_samples | Notes |
|-----|-------------|---------|--------|-------------|-------|
| 217 | 3 | 0.1 | leaf | 3 | Reference run |
| 237 | 7 | 0.2 | leaf | 5 | **Main run** — clipped excess_var |
| 266 | 50 | — | — | 25 | Large clusters only |

---

## Algorithm Notes

**HDBSCAN** is density-based: no need to pre-specify cluster count, handles noise
(label=-1), and finds clusters of varying shape and density.

**UMAP** reduces the 9-feature space to 2D for visualization. Parameters:
`n_neighbors=15`, `min_dist=0.1`.

Features normalized with `RobustScaler` before clustering. See `pipeline/config.py`
for `SELECTED_FEATURES_FOR_CLUSTERING`.
