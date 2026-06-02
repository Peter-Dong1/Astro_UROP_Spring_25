# slurm/

SLURM job scripts for the MIT Engaging cluster. All scripts auto-detect `PIPELINE_DIR`
relative to their own location, activate the `myenv` conda environment, and write logs
to `$PIPELINE_DIR/logs/`.

Submit with: `sbatch slurm/<script>.slurm` from the pipeline root.

---

## Path A — single node (simpler, requires one large node)

Run in order:

1. **`path_a.slurm`** → `run_feature_extraction.py`
   - Partition: `mit_normal` | 64 CPUs | 128 GB | 12h
   - Loads all light curves, extracts 11 features in parallel, writes `FEATURES_FILE`.

2. **`analyze.slurm`** → `run_pipeline_on_features.py`
   - Partition: `mit_normal` | 4 CPUs | 180 GB | 12h
   - Runs outlier detection, HDBSCAN, UMAP, and all plots.

---

## Path B — distributed array (fits within per-user job limits)

Run in order:

1. **`path_b_1_split.slurm`** → `split_curves.py`
   - Partition: `mit_normal` | 4 CPUs | 4 GB | 1h
   - Splits all accessible FITS paths into 28 partition pickles under `data/split_light_curves/`.

2. **`path_b_2_extract.slurm`** → `batch_feature_extraction.py` (array job)
   - Partition: `mit_preemptable` | 85 CPUs per task | 128 GB | 48h | **array 0–28**
   - Each of the 29 tasks processes one partition; writes per-curve pickles to `extracted_features/`.
   - Wait for **all** array tasks to finish before proceeding.

3. **`path_b_3_consolidate.slurm`** → `consolidate_features.py`
   - Partition: `mit_normal` | 4 CPUs | 128 GB | 12h
   - Merges all `extracted_features/features_*.pkl` into `FEATURES_FILE`.

4. **`analyze.slurm`** → `run_pipeline_on_features.py`
   - Same as Path A step 2.
