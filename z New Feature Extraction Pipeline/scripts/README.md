# scripts/

Utility and one-off analysis scripts. These are **not** part of the main pipeline run
and are not called by any SLURM job. Run them interactively after a pipeline run completes.

Each script patches `sys.path` automatically, so you can run from any working directory:
```bash
python scripts/check_nans.py
python scripts/inspect_features.py
python scripts/plot_feature_histograms.py data/all/amp_max_features/features.pkl
```

---

## Files

### `check_nans.py`

Loads `FEATURES_FILE` and prints a per-column NaN count. Run after consolidation to
verify that feature extraction produced clean output before starting the analysis step.

### `inspect_features.py`

Prints a detailed column-by-column breakdown of `FEATURES_FILE`: dtypes, shapes, and
sample values. Useful for debugging unexpected feature schemas.

### `plot_feature_histograms.py`

Plots one histogram PNG per feature from any pickled features DataFrame.

```bash
python scripts/plot_feature_histograms.py <path/to/features.pkl> [--outdir DIR]
```

Default output directory: `feature_histograms/` in the current working directory.

### `sample_clusters.py`

Plots a 5×5 sample grid of light curves for each cluster, loading from a specific
saved run's `cluster_assignments.csv`.

```bash
python scripts/sample_clusters.py --run <run_number> [--samples N] [--outdir DIR]
```

`run_number` matches the `number` variable in `config.py` used during that run.

### `analyze_similar_curves.py`

For each known light curve (from `KNOWN_LIGHT_CURVES` in `config.py`), finds the top N
most-similar curves by cosine similarity and generates cluster-assignment histograms.
Also plots normalized feature histograms.

Requires a completed pipeline run with saved cluster assignments at
`data/all/<run_number>/hdbscan_data/cluster_assignments.csv`.
