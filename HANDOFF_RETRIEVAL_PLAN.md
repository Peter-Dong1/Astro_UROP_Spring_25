# Handoff Retrieval Plan (On-Cluster)

**You are on the cluster.** All data lives under `z New Feature Extraction Pipeline/`. This plan tells you what exists, what to copy into `ORGANIZED_RESULTS/`, and what the handoff checklist expects vs what actually exists (no file generation).

---

## What exists in the pipeline folder

| What | Location | Status |
|------|----------|--------|
| **Cosine similarity outputs** | `z New Feature Extraction Pipeline/data/all/analysis_results/cluster_assignments_237_real_clippedxvar/run_237/` | ✅ Present: 3× `*_assignments.txt`, 3× `*_cluster_distribution.png` (one per known source). **No** `*_similar.csv` files. |
| **Run 237 (main clustering)** | `z New Feature Extraction Pipeline/data/all/237/` | ✅ `hdbscan_data/cluster_assignments.csv`, `umap_data/cluster_assignments.csv`. **No** cluster_probabilities.csv, outlier_scores.csv, or umap_embedding.csv. |
| **Features** | `z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl` | ✅ Present (~142 MB). |
| **Run 217** | `z New Feature Extraction Pipeline/data/all/217/` | ✅ Same structure as 237 (hdbscan_data, umap_data with cluster_assignments.csv). |
| **Run 266** | `z New Feature Extraction Pipeline/data/all/266/` | ✅ Same structure. |
| **Run 266 cluster sample plots** | `z New Feature Extraction Pipeline/plots/all266/CLUSTERS/` | ✅ 27 PNG files (cluster_*-1* and cluster_0..25). |

---

## Checklist vs reality

- **Priority 1 (cosine similarity):** Checklist expects `em01_211120_020_similar.csv`, `em01_039135_020_similar.csv`, `em01_038099_020_similar.csv` with columns rank, file_name, cosine_similarity, cluster_label. **Reality:** You have `*_assignments.txt` (top similar file paths + cluster IDs) and `*_cluster_distribution.png` for each of the 3 sources. Use those as the deliverable unless you later add a step that exports CSV from the same pipeline.
- **Priority 2 (Run 237):** Checklist expects cluster_probabilities.csv, outlier_scores.csv, umap_embedding.csv. **Reality:** Only cluster_assignments.csv exists in hdbscan_data and umap_data. Copy what exists.
- **Priorities 3–4:** features.pkl, Run 217, Run 266, and Run 266 CLUSTERS plots all exist; copy them.

---

## Step-by-step plan (copy only; run from project root)

Use these commands from your **project root** (`/home/pdong/Astro UROP` or `Astro UROP` on the cluster). Create directories only if they don’t exist.

### 1. Priority 1 – Cosine similarity (what exists)

```bash
cd "/home/pdong/Astro UROP"

mkdir -p ORGANIZED_RESULTS/3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar

cp "z New Feature Extraction Pipeline/data/all/analysis_results/cluster_assignments_237_real_clippedxvar/run_237/"* \
   ORGANIZED_RESULTS/3_similarity_analysis/analysis_results/cluster_assignments_237_real_clippedxvar/
```

**Result:** You’ll have the 3 `*_assignments.txt` and 3 `*_cluster_distribution.png` files in `ORGANIZED_RESULTS/3_similarity_analysis/...`. The checklist asks for `*_similar.csv`; those don’t exist. Either treat the `.txt` + `.png` as the deliverable or plan a separate step later to generate CSVs (that would be “generation,” not in this plan).

### 2. Priority 2 – Run 237 (main clustering)

```bash
mkdir -p ORGANIZED_RESULTS/2_clustering_runs/run_237_main/hdbscan_data
mkdir -p ORGANIZED_RESULTS/2_clustering_runs/run_237_main/umap_data

cp "z New Feature Extraction Pipeline/data/all/237/hdbscan_data/cluster_assignments.csv" \
   ORGANIZED_RESULTS/2_clustering_runs/run_237_main/hdbscan_data/

cp "z New Feature Extraction Pipeline/data/all/237/umap_data/cluster_assignments.csv" \
   ORGANIZED_RESULTS/2_clustering_runs/run_237_main/umap_data/
```

**Result:** Run 237 cluster assignments in place. No cluster_probabilities, outlier_scores, or umap_embedding to copy.

### 3. Priority 3 – Features file

```bash
mkdir -p ORGANIZED_RESULTS/1_features

cp "z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl" \
   ORGANIZED_RESULTS/1_features/
```

**Result:** `ORGANIZED_RESULTS/1_features/features.pkl` (~142 MB).

### 4. Priority 4 – Run 217, Run 266, Run 266 cluster plots

```bash
mkdir -p ORGANIZED_RESULTS/2_clustering_runs/run_217_important
mkdir -p ORGANIZED_RESULTS/2_clustering_runs/run_266_large_clusters
mkdir -p ORGANIZED_RESULTS/4_visualizations/run_266_cluster_samples

cp -r "z New Feature Extraction Pipeline/data/all/217/"* \
      ORGANIZED_RESULTS/2_clustering_runs/run_217_important/

cp -r "z New Feature Extraction Pipeline/data/all/266/"* \
      ORGANIZED_RESULTS/2_clustering_runs/run_266_large_clusters/

cp "z New Feature Extraction Pipeline/plots/all266/CLUSTERS/"* \
   ORGANIZED_RESULTS/4_visualizations/run_266_cluster_samples/
```

**Result:** Run 217 and 266 clustering outputs and 27 Run 266 cluster sample plots in ORGANIZED_RESULTS.

### 5. Optional (Priority 5)

- **HDF5 / SIG_NEV:** Only if you see them; no copy commands here. Check:
  - `z New Feature Extraction Pipeline/data/all/237/web_data/` (or similar) for features.h5, light_curves.h5
  - `z New Feature Extraction Pipeline/data/all/amp_max_features/SIG_NEV_mappings.pkl`
- **Feature histograms (640):** Already in `ORGANIZED_RESULTS/1_features/feature_histograms_640/` from earlier reorganization; nothing to copy from pipeline unless you have a different set.

---

## After copying

1. **Verify:** Use the “Data Verification” section in `HANDOFF_CHECKLIST.md` (load features.pkl, load cluster_assignments.csv, check row counts, etc.).
2. **Document gaps:** In HANDOFF_README or METHODS, note that:
   - Cosine similarity deliverable is `*_assignments.txt` + `*_cluster_distribution.png` (not `*_similar.csv`), and
   - Run 237 has only cluster_assignments.csv (no probabilities, outlier_scores, or umap_embedding).
3. **Handoff:** Share `ORGANIZED_RESULTS/` (and repo) with supervisors; point them to `ORGANIZED_RESULTS/6_documentation/HANDOFF_README.md`.

---

## Summary

| Action | Command / location |
|--------|---------------------|
| Copy similarity outputs | `run_237/*` → `ORGANIZED_RESULTS/3_similarity_analysis/.../cluster_assignments_237_real_clippedxvar/` |
| Copy Run 237 | `data/all/237/hdbscan_data` and `umap_data` → `ORGANIZED_RESULTS/2_clustering_runs/run_237_main/` |
| Copy features.pkl | `data/all/amp_max_features/features.pkl` → `ORGANIZED_RESULTS/1_features/` |
| Copy Run 217 | `data/all/217/*` → `ORGANIZED_RESULTS/2_clustering_runs/run_217_important/` |
| Copy Run 266 | `data/all/266/*` → `ORGANIZED_RESULTS/2_clustering_runs/run_266_large_clusters/` |
| Copy Run 266 plots | `plots/all266/CLUSTERS/*` → `ORGANIZED_RESULTS/4_visualizations/run_266_cluster_samples/` |

No scripts to run for retrieval; only the copy commands above. Any future step to produce `*_similar.csv` or umap_embedding would be a separate (generation) task.
