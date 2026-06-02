# pipeline — Complete Reference

This README documents every file in this directory: what it does, what it depends on, what it outputs, and what must run before it.

---

## Table of Contents

1. [Overview & Pipeline Paths](#overview--pipeline-paths)
2. [Directory Structure](#directory-structure)
3. [Core Configuration & Utilities](#core-configuration--utilities)
4. [Feature Extraction Scripts](#feature-extraction-scripts)
5. [Analysis & Visualization Scripts](#analysis--visualization-scripts)
6. [Diagnostic & Utility Scripts](#diagnostic--utility-scripts)
7. [SLURM Scripts](#slurm-scripts)
8. [Data Directory Layout](#data-directory-layout)
9. [Feature Reference](#feature-reference)

---

## Overview & Pipeline Paths

There are two ways to run the full pipeline. Both end at the same analysis step.

### Path A — Single Node (11 features, one job)

```
sbatch slurm/path_a.slurm
    └─ run_feature_extraction.py
           └─ produces: data/all/processed/features_chunk_*.pkl  (intermediate)
                        data/all/amp_max_features/features.pkl   ← FEATURES_FILE
                        data/all/amp_max_features/SIG_NEV_mappings.pkl

sbatch slurm/analyze.slurm
    └─ run_pipeline_on_features.py
           └─ produces: all plots + CSVs + HDF5s (see §Analysis)
```

**When to use**: Simpler setup; best if you have access to a large single node (64 CPUs, 128 GB).

### Path B — Distributed Batch (11 features, 4 steps — works within per-user job limits)

```
Step 1:  sbatch slurm/path_b_1_split.slurm        (~1h, 4 GB)
             └─ split_curves.py
                    └─ produces: data/split_light_curves/light_curves_partition_00..27.pkl
                                 (28 files, each a list of FITS path strings)

Step 2:  sbatch slurm/path_b_2_extract.slurm      (array 0-28, up to 48h each, 128 GB)
             └─ batch_feature_extraction.py --chunk-file ... --chunk-id ... --output-dir ...
                    └─ produces: extracted_features/features_<fits_basename>.pkl  (one per curve)

Step 3:  sbatch slurm/path_b_3_consolidate.slurm  (12h, 128 GB)
             └─ consolidate_features.py
                    └─ produces: data/all/amp_max_features/features.pkl  ← FEATURES_FILE

Step 4:  sbatch slurm/analyze.slurm               (12h, 180 GB)
             └─ run_pipeline_on_features.py
                    └─ produces: all plots + CSVs + HDF5s (see §Analysis)
```

**When to use**: Works within MIT Engaging's per-user job limits. Steps must be run in order.
Steps 1 and 3 are cheap single jobs; Step 2 is the heavy array job (29 independent tasks).

> **`FEATURES_FILE` path**: now computed from `BASE_DIR` in `config.py` — no hardcoded paths.

---

## Directory Structure

```
pipeline/
│
├── config.py                      # Central configuration (paths, parameters)
├── helper.py                      # FITS loading utilities
├── bexvar_ero.py                  # Bayesian excess variance algorithm
│
├── lib/
│   ├── __init__.py
│   └── feature_functions.py       # All shared feature logic (11 features, both paths)
│
├── run_feature_extraction.py      # Path A: single-node extraction (~80 lines)
├── batch_feature_extraction.py    # Path B step 2: per-partition extraction (~75 lines)
├── split_curves.py                # Path B step 1: split FITS paths into partitions
├── consolidate_features.py        # Path B step 3: merge per-curve pickles
│
├── run_pipeline_on_features.py    # Main analysis: outliers, HDBSCAN, UMAP, plots
│
├── slurm/                         # SLURM job scripts (see slurm/README.md)
│   ├── path_a.slurm               # Path A (1 node, 64 CPUs, 128 GB, 12h)
│   ├── path_b_1_split.slurm       # Path B step 1 (4 CPUs, 4 GB, 1h)
│   ├── path_b_2_extract.slurm     # Path B step 2 (array 0-28, 85 CPUs, 128 GB, 48h)
│   ├── path_b_3_consolidate.slurm # Path B step 3 (4 CPUs, 128 GB, 12h)
│   └── analyze.slurm              # Final analysis (4 CPUs, 180 GB, 12h)
│
├── scripts/                       # Utility/one-off scripts (see scripts/README.md)
│   ├── analyze_similar_curves.py  # Similarity search + cluster histograms for known curves
│   ├── sample_clusters.py         # Grid plots of 25 samples per cluster from a saved run
│   ├── check_nans.py              # Report NaN counts in features.pkl
│   ├── inspect_features.py        # Print column types/contents of features.pkl
│   └── plot_feature_histograms.py # Per-feature histograms from any features .pkl
│
├── data/                          # Generated data (not tracked in git)
│   ├── all/
│   │   ├── processed/             # Intermediate chunk pickles (Path A)
│   │   └── amp_max_features/      # Final features (FEATURES_FILE lives here)
│   └── split_light_curves/        # Partition files for Path B batch processing
│
├── plots/                         # Generated plots (not tracked in git)
│   └── all<number>/
│       ├── features/
│       ├── hdbscan/
│       └── umap/
│
└── extracted_features/            # Per-curve .pkl files from Path B batch job
```

---

## Core Configuration & Utilities

---

### `config.py`

**What it does**: Central configuration file. Defines all paths and parameters used by every other
script. Automatically creates output directories on import.

**Depends on**: Nothing (only `os`).

**Key settings**:
| Variable | Value | Description |
|---|---|---|
| `LOAD_SIZE` | `'all'` | How many light curves to load (`'all'` or an integer) |
| `BASE_DIR` | directory of `config.py` | Root of the pipeline |
| `DATA_DIR` | `<BASE_DIR>/data/all` | Where data outputs go |
| `FEATURES_FILE` | hardcoded cluster path | Final merged feature pickle |
| `PLOT_DIR` | `<BASE_DIR>/plots` | Root for all plots |
| `FILE_PLOT_DIR` | `plots/all501` | Versioned plot directory (controlled by `number = 501`) |
| `KNOWN_LIGHT_CURVES` | 3 filenames | Special sources of scientific interest |
| `DEFAULT_BAND` | `"med"` | Energy band used for extraction (0.6–2.3 keV) |
| `SELECTED_FEATURES_FOR_CLUSTERING` | 9 features | Features used in HDBSCAN/UMAP |

**Outputs (directories created)**:
- `data/all/`
- `data/all/processed/`
- `plots/`
- `plots/all501/`
- `plots/all501/features/`
- `plots/all501/hdbscan/`
- `plots/all501/umap/`

**Must run before**: Nothing. But must be imported first by all other scripts.

> **Important**: `number = 501` controls which versioned plot directory is used. Change this when
> starting a new analysis run to avoid overwriting previous results. `FEATURES_FILE` is hardcoded
> to the cluster path — update it if running in a different environment.

---

### `helper.py`

**What it does**: Provides all FITS file loading functionality. The pipeline-specific version of
the root-level `helper.py` — it adds `FRACEXP`, `COUNTS`, `BACKCOUNTS`, and `BACKRATIO` columns
that the root version does not load.

**Depends on**: `astropy`, `pandas`, `numpy`, `sklearn` (model_selection only), `inaccessible_lightcurves.txt`
(in the working directory).

**Key functions**:

| Function | Purpose |
|---|---|
| `load_all_fits_files(data_dir)` | Returns glob of all `*_rebinned.fits` paths |
| `load_light_curve(file_path, band, trunc)` | Opens one FITS file, returns a DataFrame with `TIME`, `TIMEDEL`, `RATE`, `ERRM`, `ERRP`, `SYM_ERR`, `FRACEXP`, `COUNTS`, `BACKCOUNTS`, `BACKRATIO`. Truncates to `trunc` rows (default 20). Sets `.attrs['FILE_NAME']`. |
| `load_n_light_curves(n, fits_files, band, trunc)` | Loads `n` curves (or `'all'`). Filters out files listed in `inaccessible_lightcurves.txt`. Returns list(s) of DataFrames. |
| `check_lightcurve_permissions(data_dir)` | Scans all FITS files for read permission and empty data. Writes `inaccessible_lightcurves.txt` to the current working directory. |
| `partition_data(light_curves, test_size, val_size)` | Splits list into train/val/test using `train_test_split`. |

**Outputs** (when run directly as `__main__`):
- `inaccessible_lightcurves.txt` — list of unreadable/empty FITS paths, one per line.

**Must run before**: Running `check_lightcurve_permissions()` once before any large extraction job
is recommended to populate `inaccessible_lightcurves.txt`. This file is checked by every loading
call to skip bad files silently.

---

### `bexvar_ero.py`

**What it does**: Implements the Bayesian excess variance (`bexvar`) algorithm
(Buchner & Bogensberger). Given photon count data with background, it estimates the
log-normal scatter of the intrinsic source count rate using nested sampling via `ultranest`.

This file contains *functions only* — the original standalone `__main__` script body is commented
out. All three core functions are imported by `run_feature_extraction.py` and
`batch_feature_extraction.py`.

**Depends on**: `ultranest`, `numpy`, `scipy`, `astropy`, `matplotlib`.

**Key functions**:

| Function | Purpose |
|---|---|
| `lscg_gen(src_counts, bkg_counts, bkg_area, rate_conversion, density_gp)` | Generates an appropriate log(source count rate) grid for numerical integration |
| `estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, rate_conversion)` | Computes the per-time-bin posterior PDF of the source count rate over the grid |
| `bexvar(log_src_crs_grid, pdfs)` | Runs `ultranest` nested sampling; returns posterior samples of `(log_mean, log_sigma)` of the log-normal count rate distribution |

**Outputs**: None when imported (all outputs are computed in-memory and returned).

**Must run before**: Nothing; this is a library module.

---

## Feature Extraction Scripts

---

### `run_feature_extraction.py`

**What it does**: **Path A single-node feature extraction.** Loads every FITS light curve (medium
band), extracts 10 statistical features per curve using multi-process parallelism, and saves
results to disk.

**Depends on**:
- `config.py` — for `FEATURES_FILE`, `DEFAULT_BAND`, `LOAD_SIZE`, `PROCESSED_DATA_DIR`
- `helper.py` — for `load_all_fits_files`, `load_n_light_curves`, `DEFAULT_DATA_DIR`
- `bexvar_ero.py` — for `lscg_gen`, `estimate_source_cr_marginalised`, `bexvar`
- `light_curve` package — for `BeyondNStd`, `StetsonK`, `MeanVariance`
- `inaccessible_lightcurves.txt` — expected in the working directory

**How it works**:
1. Loads all FITS files → loads all light curves in the medium band
2. Splits curves into chunks of 5
3. Submits each chunk to a `ProcessPoolExecutor` (up to 85 workers)
4. Each worker calls `df_extract_statistical_features_error()` per curve (see §Feature Reference)
5. Chunk results saved as `features_chunk_<N>.pkl` in `PROCESSED_DATA_DIR`
6. All chunks concatenated into a single DataFrame and saved as `FEATURES_FILE`
7. A secondary `SIG_NEV_mappings.pkl` is saved alongside `FEATURES_FILE`

**Arguments** (command-line, currently commented out in `main()`):
- `--job-id` — which job this is (0-based), for array jobs
- `--num-jobs` — total number of parallel jobs

**Outputs**:
| File | Description |
|---|---|
| `data/all/processed/features_chunk_<N>.pkl` | Intermediate per-chunk DataFrame (one per 5 curves) |
| `data/all/amp_max_features/features.pkl` | **Final merged features DataFrame** (= `FEATURES_FILE`) |
| `data/all/amp_max_features/SIG_NEV_mappings.pkl` | Dict mapping `file_path → {sig_nev, excess_var}` |

**Features DataFrame schema** (one row per light curve):
| Column | Type | Description |
|---|---|---|
| `file_path` | str | FITS basename |
| `feature_names` | list[str] | Ordered list of feature names |
| `feature_values` | np.ndarray | Corresponding feature values |
| `light_curve` | pd.DataFrame | Raw light curve data (TIME, RATE, ERRM, ERRP) |

**Must run before**: `run_pipeline_on_features.py` (Path A).

**Run via**: `get_features.slurm`

---

### `batch_feature_extraction.py`

**What it does**: **Path B per-partition feature extraction.** Takes a single pre-split partition
file (produced by `split_curves.py`), extracts features for each light curve in it, and saves one
`.pkl` file per curve. Designed to be run as a SLURM array job — one task per partition.

**Depends on**: Same as `run_feature_extraction.py` (shares identical feature extraction logic).

**Arguments** (required):
- `--chunk-file` — path to the partition `.pkl` file (e.g., `light_curves_partition_07.pkl`)
- `--chunk-id` — integer index used for labeling (the SLURM array task ID)
- `--output-dir` — directory to write individual feature files into

**How it works**:
1. Loads the partition pickle (a list of light curve DataFrames)
2. Submits individual curves to a `ProcessPoolExecutor` (up to 85 workers)
3. Each worker calls `process_chunk()` which calls `df_extract_statistical_features_error()`
4. Per-curve result saved as `features_<fits_basename>.pkl` in `--output-dir`

**Outputs**:
| File | Description |
|---|---|
| `<output-dir>/features_<fits_basename>.pkl` | One-row features DataFrame per light curve |

**Must run before**: `consolidate_features.py` (Path B step 3).

**Run via**: `run_chunks.slurm`

---

### `split_curves.py`

**What it does**: **Path B step 1.** Loads all light curves from the raw FITS files and splits
them into N equal partitions, saving each as a pickle file. This allows `batch_feature_extraction.py`
to be distributed across many SLURM array tasks.

**Depends on**:
- `helper.py` — for `load_all_fits_files`, `load_n_light_curves`
- `config.py` — for `DEFAULT_BAND`, `BASE_DIR`
- `inaccessible_lightcurves.txt`

**Key parameter**: `num_partitions = 28` (hardcoded in function signature, matches `run_chunks.slurm`
array size of `0-28`).

**Outputs**:
| File | Description |
|---|---|
| `data/split_light_curves/light_curves_partition_00.pkl` | Partition 0 (list of DataFrames) |
| `data/split_light_curves/light_curves_partition_01.pkl` | Partition 1 |
| ... | ... |
| `data/split_light_curves/light_curves_partition_27.pkl` | Partition 27 |

**Must run before**: `batch_feature_extraction.py` (Path B step 2).

**Run via**: `split_curves.slurm`

---

### `consolidate_features.py`

**What it does**: **Path B step 3.** Scans the `processedbatch/` directory for all
`features_*.pkl` files produced by `batch_feature_extraction.py`, loads them all, concatenates
into a single DataFrame, and saves as `bexvar_features/features.pkl`.

**Depends on**: Per-curve feature pickles from `batch_feature_extraction.py`.

**Key paths** (hardcoded at top of file, update before running):
- `PROCESSED_DATA_DIR` — directory containing `features_*.pkl` files
- `CONSOLIDATED_OUTPUT_PATH` — where to save the merged result

**Outputs**:
| File | Description |
|---|---|
| `data/all/bexvar_features/features.pkl` | All per-curve pickles merged into one DataFrame |

**Must run before**: `append_new_features.py` (Path B step 4).

**Run via**: `consol_feat.slurm`

---

### `append_new_features.py`

**What it does**: **Path B step 4.** Takes the consolidated `bexvar_features/features.pkl` and
appends one new feature (`ampl_sig`) to every row's `feature_names`/`feature_values`.

**Depends on**:
- `data/all/bexvar_features/features.pkl` — must exist (from `consolidate_features.py`)
- `helper.py` — for `DEFAULT_DATA_DIR`, `load_light_curve`

**Feature added**:
- `ampl_sig` — **Amplitude significance**: measures whether the peak-to-trough amplitude of the
  light curve is statistically significant given measurement errors.
  Formula: `ampl_sig = (r_max - σ_max) - (r_min + σ_min)) / sqrt(σ_max² + σ_min²)`
  where min/max are the minimum and maximum rate data points and `σ` is the symmetric error at
  that point. Returns 0 if computation fails or fewer than 3 points.

**Key paths** (hardcoded at top of file):
- `INPUT_PICKLE` — source: `bexvar_features/features.pkl`
- `OUTPUT_PICKLE` — destination: `amp_max_features/features.pkl` (= final `FEATURES_FILE`)

**Outputs**:
| File | Description |
|---|---|
| `data/all/amp_max_features/features.pkl` | Final features DataFrame with `ampl_sig` appended (= `FEATURES_FILE`) |

**Must run before**: `run_pipeline_on_features.py`.

**Run via**: `append_feat.slurm`

---

## Analysis & Visualization Scripts

---

### `run_pipeline_on_features.py`

**What it does**: The main analysis script. Loads the final `FEATURES_FILE`, runs outlier
detection, HDBSCAN clustering, UMAP dimensionality reduction, and generates a comprehensive set
of plots and data files. This is the "end" of both Path A and Path B.

**Depends on**:
- `config.py` — all paths and parameters
- `FEATURES_FILE` — the merged features pickle (must exist)
- `hdbscan`, `umap`, `sklearn`, `seaborn`, `matplotlib`, `h5py`

**Pipeline steps inside `main()`**:

1. Load `FEATURES_FILE` into `features_df`
2. Extract list of light curve DataFrames from the embedded `light_curve` column
3. Save known-curve feature values to a text file
4. Drop rows with NaN in numeric columns
5. **Filter by bexvar**: keep only curves in the top 7% by `bexvar` value (93rd percentile threshold)
6. **Feature selection**: reduce to `SELECTED_FEATURES_FOR_CLUSTERING` (9 features)
7. **Outlier detection** (`detect_outliers()`):
   - Scale with `RobustScaler`
   - Isolation Forest (250 trees, contamination=0.05)
   - Local Outlier Factor (20 neighbors, euclidean)
   - Per-feature LOF importance (leave-one-feature-out)
   - IQR method (5× IQR threshold)
   - Z-score method (|z| > 4)
   - Combined outlier flag: (IsolF AND LOF) OR (IQR AND Z-score)
   - Cosine-similarity re-ranking against known curves (α=0.3)
8. **PCA visualization** of outliers
9. **2-stage HDBSCAN clustering**:
   - First pass: `min_cluster_size=24`, `min_samples=12`, `epsilon=0.11`, `eom='leaf'`
   - For any cluster > 10,000 points: second pass with tighter parameters
10. **UMAP embedding** (2D, euclidean, n_neighbors=15, min_dist=0.1)
11. Save cluster assignments CSV
12. Various per-cluster and similarity plots

**Key configurable parameters** (set at top of file, outside `config.py`):
| Variable | Value | Description |
|---|---|---|
| `DEFAULT_MIN_CLUSTER_SIZE` | 24 | HDBSCAN min cluster size |
| `DEFAULT_EPSILON` | 0.11 | HDBSCAN cluster selection epsilon |
| `DEFAULT_EOM` | `'leaf'` | Cluster selection method |
| `DEFAULT_MIN_SAMPLES` | 12 | HDBSCAN min samples |

**Outputs** (all relative to `FILE_PLOT_DIR = plots/all501/`):

| Output | Description |
|---|---|
| `plots/all501/features/clusters_pca.png` | 2D PCA scatter of all curves; outliers in red, known curves as stars |
| `plots/all501/hdbscan/hdbscan_clusters.png` | HDBSCAN cluster labels plotted on PCA projection |
| `plots/all501/umap/umap_hdbscan_clusters.png` | HDBSCAN cluster labels on UMAP embedding (discrete colormap) |
| `plots/all501/features/corner_plot_reg_Noise Included.png` | Seaborn pairplot (scatter), all features vs features, colored by cluster |
| `plots/all501/features/corner_plot_reg_Noise Excluded.png` | Same, with noise cluster removed |
| `plots/all501/features/corner_plot_KD_Noise Included.png` | Seaborn pairplot (KDE contours) |
| `plots/all501/features/corner_plot_KD_Noise Excluded.png` | Same, noise excluded |
| `plots/all501/hdbscan/known_light_curves_features.txt` | Tab-separated table of feature values for the 3 known curves |
| `plots/all501/hdbscan/cluster_statistics.txt` | Cluster sizes, which cluster each known curve fell into, noise fraction |
| `plots/all501/hdbscan/cluster_feature_importance_<timestamp>.txt` | Top 5 distinguishing features per cluster (mean difference method) |
| `plots/all501/hdbscan/<tag>_cluster_sizes.txt` | Tab-separated: cluster label, count, whether it contains a known curve |
| `plots/all501/hdbscan/<tag>_cluster_sizes.png` | Bar chart of cluster sizes (log scale) |
| `plots/all501/hdbscan/similar_hits/<known>_similar_cluster_hist.png` | Bar chart: which clusters do the top 200 cosine-similar curves to each known curve land in |
| `plots/all501/hdbscan/similar_hits/<known>_similar_cluster_counts.csv` | CSV of above |
| `data/all/<number>/hdbscan_data/cluster_assignments.csv` | Per-curve cluster label: `file_path`, `cluster_label` |
| `data/all/<number>/grid_plots_<timestamp>/outlier_grid_<N>.png` | 5×5 grids of outlier light curves (up to 8 files × 25 curves) |
| `data/all/<number>/grid_plots_<timestamp>/regular_grid_<N>.png` | 5×5 grids of non-outlier light curves (25 curves) |
| `data/all/<number>/cluster_plots_<timestamp>/Cluster_<N>_samples.png` | 5-column grid of 25 sample light curves per cluster |
| `data/all/<number>/significant_curves/cluster_<N>_samples.png` | Sample cluster-mates for each known/significant curve |
| `data/all/<number>/similar_curves/` | Grid plots of top N similar curves for each known curve |
| `data/all/<number>/web_data/features.h5` | HDF5: feature matrix, feature names, HDBSCAN labels, outlier scores, UMAP coords |
| `data/all/<number>/web_data/light_curves.h5` | HDF5: per-curve TIME and RATE arrays (gzip compressed) |

**Must run before**: Nothing — this is the terminal analysis step.

**Run via**: `analyze_features.slurm`

---

### `analyze_similar_curves.py`

**What it does**: Secondary analysis focused on the three known interesting light curves. For each
known curve, finds the top 100 most similar curves by cosine similarity, plots which clusters they
fall into, and produces normalized feature histograms for all features. Also has functions to load
and analyze `SIG_NEV_mappings.pkl`.

**Depends on**:
- `FEATURES_FILE` — final features pickle
- `config.py`
- `run_pipeline_on_features.py` — imports `plot_light_curve`
- Cluster assignment CSV files (at `data/all/<run_number>/hdbscan_data/cluster_assignments.csv`)
- `SIG_NEV_mappings.pkl` — from `run_feature_extraction.py`

**Outputs** (all under `data/all/analysis_results/`):
| Output | Description |
|---|---|
| `analysis_results/cluster_assignments_237_real_clippedxvar/run_<N>/` | Per-known-curve cluster assignment analysis for run N |
| `analysis_results/feature_histograms_237_real_clippedxvar/` | Normalized histograms for all features |
| `analysis_results/sig_nev_analysis/` | Plots related to NEV significance |

**Must run before**: Nothing (terminal analysis). Requires `run_pipeline_on_features.py` to have
been run at least once to produce cluster assignments.

---

### `plot_cluster_samples.py`

**What it does**: Standalone CLI tool. Given a cluster ID and a cluster CSV, plots N random sample
light curves from that cluster by reading the original FITS files directly. Use this for
ad-hoc visual inspection of any cluster.

**Depends on**:
- `helper.py` — for `DEFAULT_DATA_DIR`
- `config.py` — for `DATA_DIR`
- A `cluster_assignments.csv` file (produced by `run_pipeline_on_features.py`)
- Access to the raw FITS files at `DEFAULT_DATA_DIR`

**Usage**:
```bash
python plot_cluster_samples.py <cluster_id> \
    --num-samples 10 \
    --cluster-csv <path/to/cluster_assignments.csv> \
    --output-dir cluster_plots
```

**Outputs**:
| File | Description |
|---|---|
| `cluster_plots/cluster_<ID>_samples.png` | Stacked subplots of N sample light curves from the chosen cluster |

**Must run before**: Nothing. Requires a cluster CSV from `run_pipeline_on_features.py`.

---

### `sample_clusters.py`

**What it does**: Generates grid plots (5 columns × variable rows, up to 25 samples) for every
cluster in a given run's cluster assignments. More automated than `plot_cluster_samples.py` — runs
through all clusters in one shot.

**Depends on**:
- `config.py` — for `DATA_DIR`
- `run_pipeline_on_features.py` — imports `load_features`, `plot_light_curve`
- `data/all/<run_number>/hdbscan_data/cluster_assignments.csv` (produced by `run_pipeline_on_features.py`)

**Usage**:
```bash
python sample_clusters.py --run 501 --samples 25 --outdir /path/to/output
```

**Outputs**:
| File | Description |
|---|---|
| `data/all/<run>/sample_cluster_plots/cluster_<N>_samples_<timestamp>.png` | One 5×M grid per cluster |

**Must run before**: Nothing. Requires a cluster assignment CSV.

---

### `bexvar_histograms.py`

**What it does**: CLI tool that produces one histogram PNG per feature from a features pickle.
Useful for quickly checking the distribution of each extracted feature.

**Depends on**: A features `.pkl` file (any format with `feature_names` and `feature_values` columns).

**Usage**:
```bash
python bexvar_histograms.py /path/to/features.pkl --outdir feature_histograms
```

**Outputs**:
| File | Description |
|---|---|
| `<outdir>/weighted_mean_hist.png` | Histogram (50 bins) of the `weighted_mean` feature |
| `<outdir>/bexvar_hist.png` | Histogram of `bexvar` |
| `<outdir>/<feature_name>_hist.png` | One per feature in the pickle |

**Must run before**: Nothing. Run any time after a features pickle exists.

---

## Diagnostic & Utility Scripts

---

### `inspect_features.py`

**What it does**: Loads `FEATURES_FILE` and prints a formatted column tree showing the type and
sample values for every column. Useful for debugging the schema of a features pickle after running
any extraction step.

**Depends on**: `config.py` (for `FEATURES_FILE`).

**Outputs**: Printed to stdout only. No files written.

**Usage**: `python inspect_features.py`

---

### `check_nans.py`

**What it does**: Loads `FEATURES_FILE`, selects all numeric columns, and reports which have NaN
values and how many. Use this after extraction to verify data quality before running the pipeline.

**Depends on**: `config.py` (for `FEATURES_FILE`).

**Outputs**: Printed to stdout only. No files written.

**Usage**: `python check_nans.py`

---

### `test_split_curves.py`

**What it does**: Quick sanity check — reads the first 12 partition files from `split_light_curves/`
and prints the total number of light curves across them. Use this after `split_curves.py` to verify
the split was successful.

**Depends on**: Partition pickle files in `split_light_curves/` (from `split_curves.py`).

**Outputs**: Printed to stdout only.

**Usage**: `python test_split_curves.py` (must be run from within the pipeline directory).

---

## SLURM Scripts

All scripts run on the `mit_normal` partition and activate the `myenv` conda environment.
All paths in SLURM scripts are hardcoded to `/home/pdong/Astro UROP/pipeline/`.

---

### `get_features.slurm`

| Setting | Value |
|---|---|
| Job name | `get_feature` |
| Output logs | `logs/get_feature.out` / `.err` |
| Partition | `mit_normal` |
| Nodes | 1 |
| CPUs | 64 |
| Memory | 128 GB |
| Time limit | 12 hours |

**Runs**: `run_feature_extraction.py`

**Purpose**: Single-node Path A feature extraction. 64 cores available but the script uses up to
85 workers via `ProcessPoolExecutor` (oversubscription is intentional — bexvar is I/O-bound).

**Must run after**: `inaccessible_lightcurves.txt` has been generated (by `helper.py`).
**Must run before**: `analyze_features.slurm`.

---

### `batch_get_features.slurm`

| Setting | Value |
|---|---|
| Job name | `get_feature` |
| Output logs | `logs/get_feature_%A_%a.out` / `.err` |
| Partition | `mit_normal` |
| CPUs per task | 48 |
| Memory | 128 GB |
| Time limit | 12 hours |
| Array | `0,100` (two tasks only) |

**Runs**: `run_feature_extraction.py --job-id $SLURM_ARRAY_TASK_ID --num-jobs 200`

**Purpose**: Experimental array variant of `get_features.slurm`. The `--job-id` / `--num-jobs`
logic in `run_feature_extraction.py` is currently **commented out**, so this currently runs the
same as `get_features.slurm`. Intended for future parallel sub-job splitting.

---

### `split_curves.slurm`

| Setting | Value |
|---|---|
| Job name | `split_curves` |
| Partition | `mit_normal` |
| CPUs | 64 |
| Memory | 350 GB |
| Time limit | 12 hours |

**Runs**: `split_curves.py`

**Purpose**: Path B step 1. Loads the entire dataset into RAM (hence the high 350 GB memory
request) and splits into 28 partitions.

**Must run before**: `run_chunks.slurm`.

---

### `run_chunks.slurm`

| Setting | Value |
|---|---|
| Job name | `extract_features` |
| Output logs | `logs/extract_features_%A_%a.out` / `.err` |
| Partition | `mit_preemptable` |
| CPUs per task | 85 |
| Memory | 128 GB |
| Time limit | 48 hours |
| Array | `0-28` (29 tasks) |

**Runs**: `batch_feature_extraction.py --chunk-file .../light_curves_partition_<NN>.pkl --chunk-id <IDX> --output-dir .../extracted_features`

**Purpose**: Path B step 2. Each array task processes one partition. Task `$IDX` processes
`light_curves_partition_<NN>.pkl` and writes per-curve pickles to `extracted_features/`.

**Must run after**: `split_curves.slurm`.
**Must run before**: `consol_feat.slurm`.

---

### `consol_feat.slurm`

| Setting | Value |
|---|---|
| Job name | `consolidate_features` |
| Partition | `mit_normal` |
| CPUs | 4 |
| Memory | 128 GB |
| Time limit | 12 hours |

**Runs**: `consolidate_features.py`

**Purpose**: Path B step 3. Merges all per-curve pickles into one DataFrame.

**Must run after**: All `run_chunks.slurm` array tasks have completed.
**Must run before**: `append_feat.slurm`.

---

### `append_feat.slurm`

| Setting | Value |
|---|---|
| Job name | `append_feat` |
| Partition | `mit_normal` |
| CPUs | 4 |
| Memory | 128 GB |
| Time limit | 12 hours |

**Runs**: `append_new_features.py`

**Purpose**: Path B step 4. Appends `ampl_sig` to the consolidated pickle.

**Must run after**: `consol_feat.slurm`.
**Must run before**: `analyze_features.slurm`.

---

### `analyze_features.slurm`

| Setting | Value |
|---|---|
| Job name | `ana_feature` |
| Partition | `mit_normal` |
| CPUs | 4 |
| Memory | 128 GB |
| Time limit | 12 hours |

**Runs**: `run_pipeline_on_features.py`

**Purpose**: Final analysis step. Produces all clustering, outlier detection, and visualization
outputs.

**Must run after**: Either `get_features.slurm` (Path A) or `append_feat.slurm` (Path B).

---

## Data Directory Layout

After a full Path B run, the data directory looks like:

```
data/
├── split_light_curves/
│   ├── light_curves_partition_00.pkl
│   ├── light_curves_partition_01.pkl
│   └── ...  (28 total)
│
└── all/
    ├── processed/                        ← Path A chunk files
    │   └── features_chunk_<N>.pkl
    │
    ├── bexvar_features/                  ← Path B post-consolidation
    │   └── features.pkl
    │
    ├── amp_max_features/                 ← FEATURES_FILE lives here
    │   ├── features.pkl                  ← final features DataFrame
    │   └── SIG_NEV_mappings.pkl
    │
    └── <number>/                         ← per-run analysis outputs (number=501)
        ├── hdbscan_data/
        │   └── cluster_assignments.csv
        ├── grid_plots_<timestamp>/
        │   ├── outlier_grid_1.png
        │   └── regular_grid_1.png
        ├── cluster_plots_<timestamp>/
        │   └── Cluster_<N>_samples.png
        ├── significant_curves/
        │   └── cluster_<N>_samples.png
        ├── similar_curves/
        └── web_data/
            ├── features.h5
            └── light_curves.h5

extracted_features/                       ← Path B per-curve pickles
    └── features_em01_XXXXXX_XXX_LightCurve_YYYYY_c010_rebinned.pkl
```

---

## Feature Reference

All 11 features are extracted by both paths. Shared logic lives in `lib/feature_functions.py`.

The following features are extracted per light curve (medium band, 0.6–2.3 keV):

| Feature | Description | Default if failed |
|---|---|---|
| `weighted_mean` | Inverse-variance-weighted mean of RATE. `Σ(w_i · R_i) / Σ(w_i)` where `w_i = 1/σ_i²` | 0 |
| `weighted_variance` | Inverse-variance-weighted variance around the weighted mean | 0 |
| `lag1_autocorr` | Lag-1 autocorrelation: covariance of `R[t]` with `R[t+1]`, normalized by variance of `R[t]` | 0 |
| `hurst_exp` | Hurst exponent via rescaled range: H > 0.5 = persistence, H < 0.5 = mean-reverting | 0.5 |
| `mean_rise_fall_ratio` | Count of positive consecutive differences / count of negative ones (capped at 10) | 1.0 |
| `beyond1std` | Fraction of data points more than 1σ from the weighted mean (via `light_curve` package) | 0 |
| `stetson_k` | Stetson K statistic: robust kurtosis of normalized residuals (via `light_curve` package) | 0 |
| `excess_var` | Normalized Excess Variance (NEV): `(σ²_obs - σ̄²_err) / R̄²`, clipped to minimum 0.001 | 0 |
| `bexvar` | Bayesian excess variance: median of the posterior `log(σ)` from nested sampling | 0 |
| `mean_var` | Mean variance from `light_curve.MeanVariance()` | 0 |

| `ampl_sig` | Amplitude significance: `((R_max - σ_max) - (R_min + σ_min)) / sqrt(σ_max² + σ_min²)` | 0 |

The **9 features used for clustering** (from `SELECTED_FEATURES_FOR_CLUSTERING` in `config.py`):
`weighted_mean`, `weighted_variance`, `lag1_autocorr`, `hurst_exp`, `mean_rise_fall_ratio`,
`stetson_k`, `bexvar`, `mean_var`, `ampl_sig`

The `excess_var` and `beyond1std` features are extracted but not used in the clustering step
(though `bexvar` filtering in `run_pipeline_on_features.py` uses `bexvar` before the clustering).

---

## Known Issues & Notes

- **Hardcoded paths resolved**: `FEATURES_FILE` and `EXTRACTED_FEATURES_DIR` in `config.py`
  are now derived from `BASE_DIR` (the pipeline directory itself). No `/home/pdong/` paths remain
  in the extraction scripts. `run_pipeline_on_features.py` and `analyze_similar_curves.py` may
  still contain hardcoded paths in their output subdirectory logic — check before running in a
  new environment.

- **`--job-id` / `--num-jobs` flags**: Defined in `run_feature_extraction.py`'s `parse_args()`
  for future SLURM array splitting of PATH A. Currently unused — the script uses
  `ProcessPoolExecutor` internally instead.

- **`bexvar` is slow**: It runs `ultranest` nested sampling for every light curve. This is the
  dominant cost of feature extraction. Each curve takes ~seconds; the full dataset of ~300k curves
  requires significant wall time. This is why high parallelism (85 workers) is used.

- **`number` in `config.py`**: The `number = 501` variable versions the output plot and data
  directories. Increment this before each new analysis run to preserve previous results.

- **`inaccessible_lightcurves.txt`**: Must exist in the working directory before loading data.
  Run `helper.py` as `__main__` once to generate it: `python helper.py`.
