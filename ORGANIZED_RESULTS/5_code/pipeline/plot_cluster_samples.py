#!/usr/bin/env python3
"""
Script to plot sample light curves from specific clusters.

Usage:
    python plot_cluster_samples.py <cluster_id> [--num-samples N] [--cluster-csv PATH] [--output-dir DIR]

Example:
    python plot_cluster_samples.py 3 --num-samples 5 --cluster-csv web_data/cluster_assignments.csv --output-dir cluster_plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
from pathlib import Path
from config import DATA_DIR
from helper import DEFAULT_DATA_DIR

def load_light_curve(file_path):
    """Load a light curve from a FITS file."""
    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time = data['TIME']
            rate = data['RATE']
            if 'ERRM' in data.columns.names and 'ERRP' in data.columns.names:
                errm = data['ERRM']
                errp = data['ERRP']
            else:
                # If no error columns, create zero arrays
                errm = np.zeros_like(time)
                errp = np.zeros_like(time)
            return time, rate, errm, errp
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None, None, None

def plot_light_curve(ax, time, rate, errm, errp, title=None):
    """Plot a single light curve with error bars."""
    if time is None or rate is None:
        ax.text(0.5, 0.5, 'Error loading light curve',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='red', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Plot with error bars if available
    if errm is not None and errp is not None and np.any(errm > 0) and np.any(errp > 0):
        ax.errorbar(time, rate, yerr=[errm, errp], fmt='o', markersize=4,
                   alpha=0.6, capsize=2, color='#1f77b4',
                   markeredgecolor='black', markeredgewidth=0.5)
    else:
        ax.plot(time, rate, 'o', markersize=4, alpha=0.6, color='#1f77b4',
               markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Time (MJD)', fontsize=10)
    ax.set_ylabel('Rate (count/s)', fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, pad=8, fontweight='semibold')

    # Style the plot
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Set facecolor to white
    ax.set_facecolor('white')

def plot_cluster_samples(cluster_csv, cluster_id, num_samples=5, output_dir=f'{os.path.dirname(os.path.abspath(__file__))}/cluster_plots'):
    """
    Plot sample light curves from a specific cluster.

    Args:
        cluster_csv (str): Path to the cluster assignments CSV file
        cluster_id (int): ID of the cluster to plot
        num_samples (int): Number of samples to plot
        output_dir (str): Directory to save the plot
    """
    # Load cluster assignments
    try:
        df = pd.read_csv(cluster_csv)
    except Exception as e:
        print(f"Error loading cluster assignments from {cluster_csv}: {str(e)}")
        return

    # Filter files for the requested cluster
    cluster_files = df[df['cluster_label'] == cluster_id]['file_path'].values

    if len(cluster_files) == 0:
        print(f"No files found in cluster {cluster_id}")
        return

    # Limit number of samples
    num_samples = min(num_samples, len(cluster_files))
    sample_files = np.random.choice(cluster_files, size=num_samples, replace=False)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


    # Set the style
    plt.style.use('seaborn')

    # Create figure with subplots
    fig, axes = plt.subplots(
        num_samples, 1,
        figsize=(12, 2.5 * num_samples),  # Wider and taller for better visibility
        squeeze=False,
        sharex=True,
        dpi=100
    )

    # Set the figure background to white
    fig.patch.set_facecolor('white')

    # Plot each light curve
    for i, file_path in enumerate(sample_files):
        try:
            time, rate, errm, errp = load_light_curve(DEFAULT_DATA_DIR + "/" + file_path)
            file_name = os.path.basename(file_path)
            plot_light_curve(axes[i, 0], time, rate, errm, errp,
                           f'Cluster {cluster_id} - {file_name}')
        except Exception as e:
            print(f"Error plotting {file_path}: {str(e)}")
            continue

    # Adjust layout with more padding
    plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=2.0)
    plt.subplots_adjust(hspace=0.35)  # Add more space between subplots

    # Add a main title
    fig.suptitle(f'Sample Light Curves - Cluster {cluster_id} (n={len(cluster_files)})',
                fontsize=14, y=0.995, fontweight='bold')

    # Save the figure with higher DPI
    output_file = os.path.join(output_dir, f'cluster_{cluster_id}_samples.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot sample light curves from a cluster')
    parser.add_argument('cluster_id', type=int, help='Cluster ID to plot')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to plot (default: 10)')
    parser.add_argument('--cluster-csv', type=str,
                       default=f'{DATA_DIR}/217/umap_data/cluster_assignments.csv',
                       help='Path to cluster_assignments.csv')
    parser.add_argument('--output-dir', type=str,
                       default='cluster_plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    plot_cluster_samples(
        cluster_csv=args.cluster_csv,
        cluster_id=args.cluster_id,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # Add code to calculate and save excess variance statistics
    # try:
    #     features_df = load_features()
    #     if 'excess_var' in features_df.columns:
    #         excess_var_values = features_df['excess_var'].dropna()
    #         stats = {
    #             'max': excess_var_values.max(),
    #             'min': excess_var_values.min(),
    #             'mean': excess_var_values.mean(),
    #             'median': excess_var_values.median(),
    #             'count': len(excess_var_values)
    #         }

    #         # Create output directory if it doesn't exist
    #         os.makedirs(f'{os.path.dirname(os.path.abspath(__file__))}/statistics', exist_ok=True)

    #         # Write statistics to file
    #         with open(f'{os.path.dirname(os.path.abspath(__file__))}/statistics/excess_variance_stats.txt', 'w') as f:
    #             f.write("Excess Variance Statistics\n")
    #             f.write("=========================\n")
    #             f.write(f"Maximum: {stats['max']:.6f}\n")
    #             f.write(f"Minimum: {stats['min']:.6f}\n")
    #             f.write(f"Mean: {stats['mean']:.6f}\n")
    #             f.write(f"Median: {stats['median']:.6f}\n")
    #             f.write(f"Count: {stats['count']}\n")

    #         print(f"\nExcess variance statistics saved to: statistics/excess_variance_stats.txt")
    #     else:
    #         print("\nWarning: 'excess_var' column not found in features")
    # except Exception as e:
    #     print(f"\nError calculating excess variance statistics: {str(e)}")
