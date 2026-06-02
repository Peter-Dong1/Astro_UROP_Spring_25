# ORGANIZED_RESULTS

Archived outputs and documentation from the eROSITA light curve analysis project.
Live pipeline code lives in `../pipeline/`.

---

## What's here locally

| Directory | Contents |
|---|---|
| `1_features/` | Feature distribution histograms (640-source sample, 10 PNGs) |
| `2_clustering_runs/` | Documentation of HDBSCAN runs 217, 237, 266 — results on cluster |
| `3_similarity_analysis/` | Documentation of cosine similarity analysis — results on cluster |
| `4_visualizations/` | Documentation of cluster sample plots — results on cluster |
| `6_documentation/` | Methods, data dictionary, run history, handoff notes |

## What's on the cluster

All large outputs (features.pkl, cluster assignments, UMAP embeddings, similarity CSVs,
cluster sample plots) remain on the MIT Engaging cluster at:
```
/home/pdong/Astro UROP/pipeline/
```

See `6_documentation/RUN_HISTORY.md` for the full run parameter history and what was produced.
