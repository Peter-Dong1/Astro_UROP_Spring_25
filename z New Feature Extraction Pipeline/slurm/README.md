# slurm/

SLURM job scripts for the MIT Engaging cluster. All scripts auto-detect `PIPELINE_DIR`
relative to their own location, activate the `myenv` conda environment, and write logs
to `$PIPELINE_DIR/logs/`.

Submit with: `sbatch slurm/<script>.slurm` from the pipeline root.

---

## Prerequisites (before any script can run)

1. **`myenv` conda environment** exists on the cluster with all required packages
   (`astropy`, `light_curve`, `ultranest`, `hdbscan`, `umap-learn`, etc.)
2. **eROSITA data** is accessible at `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`
3. **`inaccessible_lightcurves.txt`** exists in the directory you run `sbatch` from.
   Generate it once with `python helper.py` run interactively on the cluster.
   Without it, bad/empty FITS files won't be filtered (extraction won't crash, but will be noisy).

> **Important**: Always `sbatch` from inside `z New Feature Extraction Pipeline/`.
> `PIPELINE_DIR` is set to `$SLURM_SUBMIT_DIR` (the directory you ran `sbatch` from),
> and `inaccessible_lightcurves.txt` is also looked up from that same directory.

---

## Path A — single node (simpler, requires one large node)

Run in order:

1. **`path_a.slurm`** → `run_feature_extraction.py`
   - Partition: `mit_normal` | 64 CPUs | 128 GB | 12h
   - *Requires*: `inaccessible_lightcurves.txt`
   - Loads all light curves, extracts 11 features in parallel, writes `FEATURES_FILE`.

2. **`analyze.slurm`** → `run_pipeline_on_features.py`
   - Partition: `mit_normal` | 4 CPUs | 180 GB | 12h
   - *Requires*: `data/all/amp_max_features/features.pkl` (FEATURES_FILE) from step 1
   - Runs outlier detection, HDBSCAN, UMAP, and all plots.

---

## Path B — distributed array (fits within per-user job limits)

Run in order:

1. **`path_b_1_split.slurm`** → `split_curves.py`
   - Partition: `mit_normal` | 4 CPUs | 4 GB | 1h
   - *Requires*: `inaccessible_lightcurves.txt`
   - Splits all accessible FITS paths into 28 partition pickles under `data/split_light_curves/`.

2. **`path_b_2_extract.slurm`** → `batch_feature_extraction.py` (array job)
   - Partition: `mit_preemptable` | 85 CPUs per task | 128 GB | 48h | **array 0–28**
   - *Requires*: `data/split_light_curves/light_curves_partition_00.pkl` … `_27.pkl` (all 28 files from step 1)
   - Each of the 29 tasks processes one partition; writes per-curve pickles to `extracted_features/`.
   - Wait for **all** array tasks to finish before proceeding.

3. **`path_b_3_consolidate.slurm`** → `consolidate_features.py`
   - Partition: `mit_normal` | 4 CPUs | 128 GB | 12h
   - *Requires*: all 29 array tasks from step 2 complete → `extracted_features/features_*.pkl` populated
   - Merges all per-curve pickles into `FEATURES_FILE`.

4. **`analyze.slurm`** → `run_pipeline_on_features.py`
   - *Requires*: `data/all/amp_max_features/features.pkl` (FEATURES_FILE) from step 3
   - Same as Path A step 2.
