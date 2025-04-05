import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import random
import math

from helper import (
    load_light_curve,
    load_n_light_curves,
    load_all_fits_files,
    create_dataframe_of_light_curves,
    DEFAULT_DATA_DIR
)

# Create output directory for plots if it doesn't exist
OUTPUT_DIR = '/home/pdong/Astro UROP/plots/light_curve_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_single_light_curve(light_curve, title, output_file=None):
    """
    Plot a single light curve with error bars.

    Parameters:
        light_curve (pd.DataFrame): Light curve data
        title (str): Plot title
        output_file (str): Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))

    # Plot the light curve with error bars
    plt.errorbar(
        light_curve['TIME'],
        light_curve['RATE'],
        yerr=[light_curve['ERRM'], light_curve['ERRP']],
        fmt='o',
        capsize=3,
        markersize=5,
        label=f"Source: {light_curve.attrs.get('FILE_NAME', 'Unknown')}"
    )

    plt.xlabel('Time (s)')
    plt.ylabel('Count Rate (counts/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()

    plt.close()

def plot_multiple_energy_bands(light_curves_low, light_curves_med, light_curves_high, index=0, output_file=None):
    """
    Plot light curves from different energy bands for the same source.

    Parameters:
        light_curves_low (list): List of light curves in low energy band
        light_curves_med (list): List of light curves in medium energy band
        light_curves_high (list): List of light curves in high energy band
        index (int): Index of the light curve to plot
        output_file (str): Path to save the plot (if None, plot is displayed)
    """
    if index >= len(light_curves_low) or index >= len(light_curves_med) or index >= len(light_curves_high):
        print(f"Index {index} out of range. Maximum index: {min(len(light_curves_low), len(light_curves_med), len(light_curves_high)) - 1}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Get the light curves for the specified index
    lc_low = light_curves_low[index]
    lc_med = light_curves_med[index]
    lc_high = light_curves_high[index]

    # Source name for the title
    source_name = lc_low.attrs.get('FILE_NAME', 'Unknown')

    # Plot low energy band (0.2-0.6 keV)
    axes[0].errorbar(
        lc_low['TIME'],
        lc_low['RATE'],
        yerr=[lc_low['ERRM'], lc_low['ERRP']],
        fmt='o',
        capsize=3,
        color='blue',
        label='0.2-0.6 keV'
    )
    axes[0].set_ylabel('Count Rate (counts/s)')
    axes[0].set_title(f'Light Curve - Low Energy Band (0.2-0.6 keV)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot medium energy band (0.6-2.3 keV)
    axes[1].errorbar(
        lc_med['TIME'],
        lc_med['RATE'],
        yerr=[lc_med['ERRM'], lc_med['ERRP']],
        fmt='o',
        capsize=3,
        color='green',
        label='0.6-2.3 keV'
    )
    axes[1].set_ylabel('Count Rate (counts/s)')
    axes[1].set_title(f'Light Curve - Medium Energy Band (0.6-2.3 keV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot high energy band (2.3-5.0 keV)
    axes[2].errorbar(
        lc_high['TIME'],
        lc_high['RATE'],
        yerr=[lc_high['ERRM'], lc_high['ERRP']],
        fmt='o',
        capsize=3,
        color='red',
        label='2.3-5.0 keV'
    )
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Count Rate (counts/s)')
    axes[2].set_title(f'Light Curve - High Energy Band (2.3-5.0 keV)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(f'Light Curves for Source: {source_name}', fontsize=16)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()

    plt.close()

def plot_light_curve_comparison(light_curves, indices, band_name, output_file=None):
    """
    Plot multiple light curves from the same energy band for comparison.

    Parameters:
        light_curves (list): List of light curves
        indices (list): List of indices to plot
        band_name (str): Name of the energy band
        output_file (str): Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))

    for i, idx in enumerate(indices):
        if idx >= len(light_curves):
            print(f"Index {idx} out of range. Maximum index: {len(light_curves) - 1}")
            continue

        lc = light_curves[idx]
        source_name = lc.attrs.get('FILE_NAME', f'Source {idx}')

        plt.errorbar(
            lc['TIME'],
            lc['RATE'],
            yerr=[lc['ERRM'], lc['ERRP']],
            fmt='o',
            capsize=3,
            color=colors[i],
            label=f"Source: {source_name}"
        )

    plt.xlabel('Time (s)')
    plt.ylabel('Count Rate (counts/s)')
    plt.title(f'Comparison of Light Curves - {band_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_light_curve_grid(light_curves, band_name, output_file=None, n_curves=100):
    """
    Plot a grid of light curves from the same energy band.

    Parameters:
        light_curves (list): List of light curves
        band_name (str): Name of the energy band
        output_file (str): Path to save the plot (if None, plot is displayed)
        n_curves (int): Number of light curves to plot (default: 100)
    """
    # Ensure we don't try to plot more curves than available
    n_curves = min(n_curves, len(light_curves))

    # Calculate grid dimensions (default 10x10)
    grid_size = int(np.ceil(np.sqrt(n_curves)))
    rows = grid_size
    cols = grid_size

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20), sharex=False, sharey=False)
    fig.suptitle(f'Grid of {n_curves} Light Curves - {band_name}', fontsize=16)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Plot each light curve in its own subplot
    for i in range(n_curves):
        if i >= len(light_curves):
            break

        lc = light_curves[i]
        source_name = lc.attrs.get('FILE_NAME', f'Source {i}')

        # Get short source name for the title
        short_name = os.path.splitext(source_name)[0][-18:-9] if source_name else f'Src {i}'

        # Plot the light curve
        axes[i].errorbar(
            lc['TIME'],
            lc['RATE'],
            yerr=[lc['ERRM'], lc['ERRP']],
            fmt='o',
            capsize=2,
            markersize=3,
            color='blue'
        )

        # Set title and grid
        axes[i].set_title(short_name, fontsize=8)
        axes[i].grid(True, alpha=0.3)

        # Remove x and y labels for cleaner appearance
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])

        # Add tick marks but make them smaller
        axes[i].tick_params(axis='both', which='both', length=2, labelsize=6)

    # Hide any unused subplots
    for i in range(n_curves, rows * cols):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved grid plot to {output_file}")
    else:
        plt.show()

    plt.close()

if __name__ == '__main__':
    # Load FITS files
    print("Loading FITS files...")
    fits_files = load_all_fits_files()
    print(f"Found {len(fits_files)} FITS files")

    # Load a sample of light curves (adjust n as needed)
    n_samples = 100
    print(f"Loading {n_samples} sample light curves...")

    # Load light curves for all energy bands
    light_curves_low, light_curves_med, light_curves_high = load_n_light_curves(n_samples, fits_files, band='all', trunc = 1000)

    print(f"Loaded {len(light_curves_low)} low band, {len(light_curves_med)} medium band, and {len(light_curves_high)} high band light curves")

    # Example 1: Plot a single light curve from each band
    print("\nGenerating example plots...")

    # Plot a single light curve from the low energy band
    # plot_single_light_curve(
    #     light_curves_low[0],
    #     f"Low Energy Band Light Curve (0.2-0.6 keV) - {light_curves_low[0].attrs.get('FILE_NAME', 'Unknown')}",
    #     os.path.join(OUTPUT_DIR, "example_low_band.png")
    # )

    # # Plot a single light curve from the medium energy band
    # plot_single_light_curve(
    #     light_curves_med[0],
    #     f"Medium Energy Band Light Curve (0.6-2.3 keV) - {light_curves_med[0].attrs.get('FILE_NAME', 'Unknown')}",
    #     os.path.join(OUTPUT_DIR, "example_medium_band.png")
    # )

    # # Plot a single light curve from the high energy band
    # plot_single_light_curve(
    #     light_curves_high[0],
    #     f"High Energy Band Light Curve (2.3-5.0 keV) - {light_curves_high[0].attrs.get('FILE_NAME', 'Unknown')}",
    #     os.path.join(OUTPUT_DIR, "example_high_band.png")
    # )

    # Example 2: Plot all energy bands for the same source
    plot_multiple_energy_bands(
        light_curves_low,
        light_curves_med,
        light_curves_high,
        index=0,
        output_file=os.path.join(OUTPUT_DIR, "all_energy_bands_comparison.png")
    )

    # Example 3: Compare multiple sources in the same energy band
    # plot_light_curve_comparison(
    #     light_curves_med,
    #     indices=[0, 1, 2],
    #     band_name="Medium Energy Band (0.6-2.3 keV)",
    #     output_file=os.path.join(OUTPUT_DIR, "multiple_sources_comparison.png")
    # )

    # Example 4: Plot a grid of 100 light curves (or as many as available)
    n_grid_samples = min(100, len(light_curves_med))  # Use up to 100 curves
    print(f"\nGenerating grid plot of {n_grid_samples} light curves...")
    plot_light_curve_grid(
        light_curves_med[:n_grid_samples],
        band_name="Medium Energy Band (0.6-2.3 keV)",
        output_file=os.path.join(OUTPUT_DIR, "light_curve_grid.png")
    )

    print("\nAll example plots have been generated and saved to the 'light_curve_plots' directory.")
