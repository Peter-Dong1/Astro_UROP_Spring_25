# pipeline_modules/

Modules used exclusively by `run_pipeline_on_features.py` (the final analysis step).
Nothing else in the pipeline imports from here.

## Files

### `pipeline_io.py`

I/O helpers for loading and preparing data at analysis time.

| Function | Purpose |
|---|---|
| `load_features()` | Reads `FEATURES_FILE` pickle; raises if missing. |
| `build_index_maps(features_df)` | Builds `name_to_idxs` and `fullpath_to_idx` dicts for fast lookup. |
| `warn_on_duplicate_basenames(name_to_idxs)` | Prints a warning if any two rows share a basename. |
| `filter_features(features_df, selected_features)` | Subsets `feature_values` to only the named features. |
| `prepare_light_curves(features_df)` | Extracts `TIME/RATE/ERRM/ERRP` from each row's embedded `light_curve` field. |
| `save_cluster_labels(features_df, labels, output_dir)` | Writes `cluster_assignments.csv`. |
| `save_known_light_curve_features(features_df, output_dir)` | Writes feature values for `KNOWN_LIGHT_CURVES` to a TSV. |
| `apply_bexvar_filter(features_df, percentile)` | Keeps only the top `(100-percentile)%` of sources by `bexvar`. Disabled by default. |
| `save_analysis_results(...)` | Saves features + light curves to HDF5 for web visualization. Disabled by default. |

### `pipeline_clustering.py`

Outlier detection and clustering.

| Function | Purpose |
|---|---|
| `detect_outliers(features_df)` | Runs Isolation Forest + LOF; returns results dict with `combined_outlier`, `iso_score`, `lof_score`, `scaled_features`. |
| `run_hdbscan_clustering(features_df, two_stage, ...)` | UMAP → HDBSCAN. `two_stage=True` re-clusters large clusters above `size_threshold`. Returns `(labels, feature_matrix, umap_embedding)`. |
| `save_cluster_size_histogram(paths, labels, output_dir, tag)` | Bar chart of cluster sizes. |
| `plot_subcluster_histograms(orig_labels, new_labels, threshold, outdir)` | Histograms showing how large original clusters split in two-stage mode. |
| `analyze_cluster_feature_importance(matrix, names, labels)` | Prints per-cluster feature mean/std table. |

### `pipeline_plotting.py`

All visualization functions. All functions save to disk and return the output path.

| Function | Purpose |
|---|---|
| `visualize_features(scaled_arr, outlier_mask, paths)` | PCA scatter colored by outlier status. |
| `create_grid_plots(light_curves, results, output_dir, timestamp)` | Grid of outlier and normal sample curves. |
| `plot_cluster_samples(light_curves, features_df, labels, output_dir, timestamp)` | One grid per cluster with 25 random samples. |
| `plot_noise_cluster_samples(light_curves, labels, output_dir)` | Same but for noise label (-1). |
| `plot_significant_curves_with_cluster(known_lcs, light_curves, features_df, labels, output_dir)` | Plots each known curve with its cluster context. |
| `plot_top_similar_curves(light_curves, features_df, known_lcs, output_dir)` | Cosine-similarity top-N nearest neighbors for each known curve. |
| `plot_correlation_matrix(feature_matrix, feature_names)` | Seaborn heatmap of inter-feature correlations. |
| `histogram_similar_curve_cluster_hits(features_df, labels, known_lcs, output_dir, n_similar)` | Cluster-membership histogram for the top-N most similar curves. |
| `plot_umap_colored_by_feature(features_df, umap_embedding, feature_list, output_dir, ...)` | One UMAP scatter per feature, colored by feature value. |
