# eROSITA Light Curve Analysis

**Project**: Detection and clustering of X-ray transients in eRASS1 data
**Institution**: MIT Kavli Institute
**Period**: January – May 2026
**Student**: Peter Dong

---

## What this project does

Analyzes ~200,000 X-ray light curves from the eROSITA All-Sky Survey (eRASS1). The final pipeline:
1. Extracts 11 statistical features per light curve (weighted variance, Hurst exponent, Bayesian excess variance, etc.)
2. Clusters sources with HDBSCAN + UMAP dimensionality reduction
3. Ranks sources by cosine similarity to 3 known interesting light curves

Early-semester deep learning experiments (RNN VAE, Transformer VAE) were abandoned — light curves have only 10–20 time points, too sparse for deep learning. Statistical features proved more effective and interpretable.

---

## Repository structure

```
Astro_UROP_Spring_25/
├── pipeline/               # Live feature extraction + analysis pipeline
├── ORGANIZED_RESULTS/      # Archived outputs and documentation
├── ARCHIVED_CODE/          # Deep learning experiments (not used in final results)
├── CLAUDE.md               # Claude Code instructions
└── .gitignore
```

---

## pipeline/

The active codebase. See `pipeline/README.md` for the full reference.

**Two ways to run feature extraction**, both produce the same 11 features:

| Path | When to use | Steps |
|---|---|---|
| Path A (single node) | Simpler; needs one large node (64 CPUs, 128 GB) | `sbatch pipeline/slurm/path_a.slurm` → `sbatch pipeline/slurm/analyze.slurm` |
| Path B (distributed) | Fits within per-user SLURM limits | Split → Extract (array 0–28) → Consolidate → Analyze |

**Always `sbatch` from inside `pipeline/`** — `PIPELINE_DIR` resolves from `$SLURM_SUBMIT_DIR`.

**Prerequisites before any job**: `inaccessible_lightcurves.txt` must exist in `pipeline/`
(generate once with `python pipeline/helper.py` on the cluster).

### Key files

| File | Purpose |
|---|---|
| `pipeline/config.py` | All paths and parameters. Change `number` to version a new run. |
| `pipeline/helper.py` | FITS loader — `load_light_curve`, `load_all_fits_files`, `load_n_light_curves` |
| `pipeline/bexvar_ero.py` | Bayesian excess variance via ultranest nested sampling |
| `pipeline/lib/feature_functions.py` | All 11 feature extraction functions (shared by both paths) |
| `pipeline/run_pipeline_on_features.py` | Final analysis: outlier detection, HDBSCAN, UMAP, all plots |
| `pipeline/pipeline_modules/` | IO, clustering, and plotting helpers used by the analysis step |
| `pipeline/scripts/` | Utility scripts: NaN checks, feature inspection, histograms, similarity analysis |
| `pipeline/slurm/` | All SLURM job scripts (see `pipeline/slurm/README.md`) |

### Features extracted (11 total)

`weighted_mean`, `weighted_variance`, `lag1_autocorr`, `hurst_exp`, `mean_rise_fall_ratio`,
`beyond1std`, `stetson_k`, `excess_var`, `bexvar`, `mean_var`, `ampl_sig`

**9 used for clustering** (excludes `beyond1std` and `excess_var` — see `config.py`):
`weighted_mean`, `weighted_variance`, `lag1_autocorr`, `hurst_exp`, `mean_rise_fall_ratio`,
`stetson_k`, `bexvar`, `mean_var`, `ampl_sig`

### Data

eROSITA FITS files on cluster: `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`

Energy bands: Low 0.2–0.6 keV · Medium 0.6–2.3 keV · High 2.3–5.0 keV

Final features file (on cluster): `pipeline/data/all/amp_max_features/features.pkl`

---

## ORGANIZED_RESULTS/

Archived outputs from completed analysis runs. See `ORGANIZED_RESULTS/README.md`.

| Subdirectory | What's there |
|---|---|
| `1_features/` | 10 feature histogram PNGs from a 640-source sample (locally present) |
| `2_clustering_runs/` | Documentation of runs 217, 237, 266 — results on cluster |
| `3_similarity_analysis/` | Documentation of cosine similarity analysis — results on cluster |
| `4_visualizations/` | Documentation of cluster sample plots — results on cluster |
| `6_documentation/` | `METHODS.md`, `DATA_DICTIONARY.md`, `RUN_HISTORY.md`, `HANDOFF_README.md` |

**Primary result**: Run 237 (`min_cluster_size=7`, `epsilon=0.2`, `method=leaf`) with
excess variance clipped. Cluster outputs and cosine similarity CSVs on cluster at
`/home/pdong/Astro UROP/pipeline/data/all/`.

**Known interesting sources** (used for similarity ranking):
- `em01_211120_020_LightCurve_00007_c010_rebinned.fits`
- `em01_039135_020_LightCurve_00058_c010_rebinned.fits`
- `em01_038099_020_LightCurve_00005_c010_rebinned.fits`

---

## ARCHIVED_CODE/

Deep learning experiments from Phase 1 (January–February 2026). Not part of final results.
See `ARCHIVED_CODE/README.md` for full context on what was tried and why it was abandoned.

- `deep_learning_experiments/` — RNN VAE, Transformer VAE, plotting utilities, SLURM scripts
- `notebooks/` — Jupyter notebook checkpoints from exploratory analysis

---

## Conda environment

All pipeline code runs in the `myenv` environment on the MIT Engaging cluster.

```bash
conda activate myenv
```
