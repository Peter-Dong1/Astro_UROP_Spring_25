from pipeline_modules.pipeline_io import (
    build_index_maps, warn_on_duplicate_basenames, load_features,
    filter_features, prepare_light_curves, save_cluster_labels,
    save_known_light_curve_features, apply_bexvar_filter
)
from pipeline_modules.pipeline_clustering import (
    detect_outliers, run_hdbscan_clustering,
    save_cluster_size_histogram, plot_subcluster_histograms,
    analyze_cluster_feature_importance
)
from pipeline_modules.pipeline_plotting import (
    visualize_features, create_grid_plots, plot_cluster_samples,
    plot_noise_cluster_samples, plot_significant_curves_with_cluster,
    plot_top_similar_curves, plot_correlation_matrix,
    histogram_similar_curve_cluster_hits, plot_umap_colored_by_feature
)
from config import (
    HDBSCAN_OUTPUT_DIR, UMAP_OUTPUT_DIR, DATA_DIR,
    FEATURE_OUTPUT_DIR, KNOWN_LIGHT_CURVES, SELECTED_FEATURES_FOR_CLUSTERING, number
)
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    """Main function to run the analysis pipeline."""
    print("Starting light curve analysis pipeline...")
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = timestamp[5:11] + "_" + timestamp[10:12]

    try:
        # Load features
        features_df = load_features()
        print(f"Loaded {len(features_df)} light curves")

        file_paths, basenames, name_to_idxs, fullpath_to_idx = build_index_maps(features_df)
        warn_on_duplicate_basenames(name_to_idxs)

        light_curves = prepare_light_curves(features_df)

        # --- load SIG_NEV map ---------------------------------------
        # sig_nev_file = os.path.join(os.path.dirname(FEATURES_FILE), 'SIG_NEV_mappings.pkl')
        # with open(sig_nev_file, 'rb') as f:
        #     sig_map = pickle.load(f)

        # # annotate features_df
        # features_df['sig_nev'] = features_df['file_path'].map(
        #     lambda p: sig_map.get(p, {}).get('sig_nev', np.nan)
        # )

        # # mask of curves we keep (sig_nev >= 0.01)
        # keep_mask = features_df['sig_nev'] >= 0.001
        # n_before = len(features_df)
        # features_df = features_df.loc[keep_mask].reset_index(drop=True)
        # light_curves = [lc for lc, keep in zip(light_curves, keep_mask) if keep]
        # print(f"Filtered out {n_before - len(features_df)} curves with SIG_NEV < 0.01, {len(features_df)} remain")

        features_df['basename'] = features_df['file_path'].apply(lambda p: os.path.basename(p))
        save_known_light_curve_features(features_df, HDBSCAN_OUTPUT_DIR)

        numeric_cols = features_df.select_dtypes(include='number').columns.tolist()
        features_df = features_df.dropna(subset=numeric_cols)

        # To filter by bexvar, uncomment:
        # features_df = apply_bexvar_filter(features_df, percentile=93)
        # light_curves = prepare_light_curves(features_df)  # re-sync after filter

        print(f"Running pipeline on {len(features_df)} light curves.")

        features_df = filter_features(features_df, SELECTED_FEATURES_FOR_CLUSTERING)


        results = detect_outliers(features_df)
        print("Finished Detecting Outliers")

        # Visualize outliers (make sure scaled_features is 2D)
        scaled_arr = np.vstack(results['scaled_features'].values)
        outlier_mask = results['combined_outlier'].values
        paths       = features_df['file_path'].values
        visualize_features(scaled_arr, outlier_mask, paths)


        # Run HDBSCAN clustering
        hdbscan_labels, feature_matrix, umap_embedding = run_hdbscan_clustering(
                        features_df,
                        two_stage=True,
                        size_threshold=10000
                    )

        # Save cluster assignments to CSV
        save_cluster_labels(
            features_df=features_df,
            hdbscan_labels=hdbscan_labels,
            output_dir=DATA_DIR + f'/{number}/hdbscan_data'
        )

        # single‐pass
        labels_single, _, umap_embedding_single= run_hdbscan_clustering(
            features_df,
            two_stage=False,
            output_file=os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_single_pass.png")
        )
        save_cluster_size_histogram(paths, labels_single, HDBSCAN_OUTPUT_DIR, "single_pass")

        # two‐stage
        labels_two, _, umap_embedding_two = run_hdbscan_clustering(
            features_df,
            two_stage=True,
            size_threshold=10000,
            output_file=os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_two_stage.png")
        )
        save_cluster_size_histogram(paths, labels_two, HDBSCAN_OUTPUT_DIR, "two_stage")

        histogram_similar_curve_cluster_hits(
            features_df=features_df,
            cluster_labels=labels_two,
            known_light_curves=KNOWN_LIGHT_CURVES,
            output_dir=HDBSCAN_OUTPUT_DIR,
            n_similar=1000
        )

        cluster_plots_dir = plot_cluster_samples(
            light_curves=light_curves,
            features_df=features_df,
            cluster_labels=labels_two,
            output_dir=HDBSCAN_OUTPUT_DIR,
            timestamp=timestamp
        )
        print(f"Cluster plots: {cluster_plots_dir}")

        plot_noise_cluster_samples(light_curves, labels_two, HDBSCAN_OUTPUT_DIR)

        # plot only those original clusters with >10 000 members
        plot_subcluster_histograms(
            orig_labels=labels_single,
            new_labels=labels_two,
            threshold=10000,
            outdir=HDBSCAN_OUTPUT_DIR + "/subcluster_hists"
        )


        df_split = pd.DataFrame({
            "single": labels_single,
            "two":    labels_two
        })
        split_counts = df_split.groupby("single")["two"].nunique()
        clusters_that_split = split_counts[split_counts > 1].index.tolist()
        print("Original clusters subdivided by two-stage pass:", clusters_that_split)

        # --- NEW: Visualize each split on the UMAP embedding ---
        # (make sure you still have `umap_embedding` from your UMAP step)
        for orig in clusters_that_split:
            mask    = (labels_single == orig)
            coords  = umap_embedding_two[mask]   # shape (n_points, 2)
            new_lbls = labels_two[mask]

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                coords[:,0], coords[:,1],
                c=new_lbls, cmap="tab10",
                s=30, alpha=0.8, edgecolor="k", linewidth=0.2
            )
            plt.title(f"Cluster {orig} → split into {new_lbls.max()-new_lbls.min()+1} parts")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.colorbar(sc, label="Two-stage subcluster ID")
            plt.tight_layout()
            out_fn = os.path.join(HDBSCAN_OUTPUT_DIR, f"split_cluster_{orig}.png")
            plt.savefig(out_fn, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Wrote split-cluster plot: {out_fn}")

        # Create correlation matrix and pairplot
        feature_names = features_df['feature_names'].iloc[0]
        corr_matrix_file = plot_correlation_matrix(feature_matrix, feature_names)
        # pairplot_file1 = plot_feature_pairplot(feature_matrix, feature_names, hdbscan_labels, remove_noise=False)
        # pairplot_file2 = plot_feature_pairplot(feature_matrix, feature_names, hdbscan_labels, remove_noise=True)
        # Create grid plots of outliers and regular curves
        grid_plots_dir = create_grid_plots(
            light_curves, results, FEATURE_OUTPUT_DIR, timestamp
        )

        # Plot significant curves (known light curves) with their clusters
        sig_plots_dir = plot_significant_curves_with_cluster(
            KNOWN_LIGHT_CURVES, light_curves, features_df, hdbscan_labels, HDBSCAN_OUTPUT_DIR
        )

        # Plot similar curves to known light curves
        similar_plots_dir = plot_top_similar_curves(
            light_curves, features_df, KNOWN_LIGHT_CURVES, FEATURE_OUTPUT_DIR
        )

        # Analyze feature importance for HDBSCAN clusters
        analyze_cluster_feature_importance(
            feature_matrix, feature_names, hdbscan_labels
        )

        print(f"\nAnalysis pipeline completed in {time.time() - start_time:.2f} seconds")
        print("\nOutput directories:")
        print(f"Grid plots: {grid_plots_dir}")
        # print(f"Cluster plots: {cluster_plots_dir}")
        print(f"Significant curve plots: {sig_plots_dir}")
        print(f"Similar curves plots: {similar_plots_dir}")
        print(f"Correlation matrix: {corr_matrix_file}")
        # print(f"Feature pairplot (Noise Included): {pairplot_file1}")
        # print(f"Feature pairplot (Noise Excluded): {pairplot_file2}")

        # Extract UMAP embedding coordinates for web visualization
        umap_x = umap_embedding_two[:, 0] if umap_embedding_two is not None else None
        umap_y = umap_embedding_two[:, 1] if umap_embedding_two is not None else None
        umap_coords = (umap_x, umap_y) if umap_x is not None and umap_y is not None else None

        # Make UMAP color-by-feature plots for all features currently in features_df
        _ = plot_umap_colored_by_feature(
            features_df=features_df,
            umap_embedding=umap_embedding_two,   # reuse the embedding you already computed
            feature_list="all",              # or e.g. ['bexvar', 'skewness', 'kurtosis']
            output_dir=UMAP_OUTPUT_DIR,
            highlight_known=True,
            robust_color_limits=True,
            log_color=False
        )


        # Save results for web visualization using HDF5
        # features_file, light_curves_file = save_analysis_results(
        #     features_df,
        #     umap_labels,
        #     hdbscan_labels,
        #     results,
        #     feature_matrix,
        #     feature_names,
        #     output_dir=DATA_DIR,
        #     umap_embedding=umap_embedding
        # )
        # print(f"Saved features to: {features_file}")
        # print(f"Saved light curves to: {light_curves_file}")

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
