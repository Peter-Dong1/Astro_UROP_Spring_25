import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from config import DATA_DIR
from run_pipeline_on_features import load_features, plot_light_curve


def plot_cluster_sample_grid(run_number, cluster_col='cluster_label', sample_size=25, output_dir=None):
    """
    Create grid plots of sample light curves from each cluster.

    Args:
        run_number (int): The run number for which cluster assignments were saved.
        cluster_col (str): Column name for cluster labels.
        sample_size (int): Number of samples per cluster.
        output_dir (str): Output directory for saving plots.
    """
    print(f"\nPlotting cluster samples for run {run_number}...")
    cluster_file = Path(DATA_DIR) / f"{run_number}" / "hdbscan_data" / "cluster_assignments.csv"
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    features_df = load_features()
    cluster_df = pd.read_csv(cluster_file)

    # Merge cluster labels into the features DataFrame
    features_df = features_df.merge(cluster_df, on='file_path', how='left')

    light_curves = [pd.DataFrame({
        'TIME': lc['TIME'],
        'RATE': lc['RATE'],
        'ERRM': lc['ERRM'],
        'ERRP': lc['ERRP']
    }) for lc in features_df['light_curve']]

    timestamp = datetime.now().strftime("%m%d_%H")
    output_dir = output_dir or Path(DATA_DIR) + "/" + f"{run_number}" + "/" + "sample_cluster_plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting to plot cluster samples for run {run_number}...")
    for cluster in sorted(features_df[cluster_col].dropna().unique()):
        cluster = int(cluster)
        cluster_indices = features_df.index[features_df[cluster_col] == cluster].tolist()
        if not cluster_indices:
            continue
        print(f"Plotting cluster {cluster} with {len(cluster_indices)} light curves...")
        sampled_indices = np.random.choice(cluster_indices, min(sample_size, len(cluster_indices)), replace=False)

        n_rows = (len(sampled_indices) + 4) // 5
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
        axes = axes.ravel()

        for i, idx in enumerate(sampled_indices):
            plot_light_curve(axes[i], light_curves[idx], title=f"Sample {i+1}")
        for ax in axes[len(sampled_indices):]:
            ax.axis("off")

        plt.suptitle(f"Cluster {cluster} (n={len(cluster_indices)} total)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_file = output_dir + f"/cluster_{cluster}_samples_{timestamp}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved cluster plot to {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sample light curves from each cluster")
    parser.add_argument("--run", type=int, required=True, help="Run number to identify the cluster assignment")
    parser.add_argument("--samples", type=int, default=25, help="Number of samples per cluster")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory")

    args = parser.parse_args()
    plot_cluster_sample_grid(args.run, sample_size=args.samples, output_dir=args.outdir)
