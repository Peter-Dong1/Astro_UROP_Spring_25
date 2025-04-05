"""
Helper functions for loading and processing eROSITA light curve data.

This module provides utilities for:
- Loading FITS files containing light curve data
- Processing light curves across different energy bands
- Creating dataframes from light curve data
"""

import os
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import random
import math

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Default path to the directory containing the FITS files
DEFAULT_DATA_DIR = '/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned'

def create_dataframe_of_light_curves(n, fits_files, band='all'):
    """
    Create a dataframe of dataframes where each row corresponds to a light curve
    and each column represents a specific energy band.

    Parameters:
        fits_files (list): List of paths to FITS files.

    Returns:
        pd.DataFrame: A dataframe of dataframes.
    """

    data = []

    for file_path in fits_files:
        light_curve_low = load_light_curve(file_path, band=0)  # Low band
        light_curve_med = load_light_curve(file_path, band=1)  # Medium band
        light_curve_high = load_light_curve(file_path, band=2)  # High band

        data.append({
            'file_name': os.path.basename(file_path),
            'low_band': light_curve_low,
            'medium_band': light_curve_med,
            'high_band': light_curve_high
        })

    # Convert the list of dictionaries into a dataframe
    df_of_dfs = pd.DataFrame(data)
    return df_of_dfs

def read_inaccessible_lightcurves():
    """
    Read the list of inaccessible light curves from the text file in the notebook directory.

    Returns:
        list: List of file paths that were inaccessible
    """
    file_path = os.path.join(os.getcwd(), "inaccessible_lightcurves.txt")

    try:
        with open(file_path, 'r') as f:
            # Read all lines and remove any trailing whitespace
            inaccessible_files = [line.strip() for line in f.readlines()]
        return inaccessible_files
    except FileNotFoundError:
        print(f"No inaccessible light curves file found at {file_path}")
        return []

# Function to load a single FITS file and return as a Pandas DataFrame
def load_light_curve(file_path, band=1, trunc=20):
    """
    Load light curve data from a FITS file and return a Pandas DataFrame including asymmetric errors (ERRM and ERRP).

    Parameters:
        file_path (str): Path to the FITS file.
        band (int): Energy band index to load data for (default: 1).

    Returns:
        pd.DataFrame or None: DataFrame with light curve data, or None if file is skipped.
    """
    with fits.open(file_path) as hdul:
        try:
            data = hdul[1].data  # Assuming light curve data is in the second HDU
            light_curve = pd.DataFrame({
                'TIME': data['TIME'],
                'TIMEDEL': data['TIMEDEL'],
                'RATE': data['RATE'][:, band],  # Light curve intensity
                'ERRM': data['RATE_ERRM'][:, band],  # Negative error
                'ERRP': data['RATE_ERRP'][:, band],  # Positive error
                'SYM_ERR': (data['RATE_ERRM'][:, band] + data['RATE_ERRP'][:, band]) / 2,  # Symmetric error approximation
            })

            # Truncate to a maximum of 20 data points
            if len(light_curve) > trunc:
                light_curve = light_curve.iloc[:trunc]
            # Attach metadata as attributes
            light_curve.attrs['FILE_NAME'] = os.path.basename(file_path)
            light_curve.attrs['OUTLIER'] = False
            return light_curve
        except KeyError:
            print(f"Skipping file {file_path}: some key not found")
            return None

def load_all_fits_files(data_dir=None):
    """
    Loads all fits files from the specified directory.

    Parameters:
        data_dir (str, optional): The filepath where the data is located.
                                If None, uses the default directory.

    Returns:
        list: A list of all the fits files
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    return glob.glob(os.path.join(data_dir, "*.fits"))

def load_n_light_curves(n, fits_files, band='all', trunc=20):
    """
    Loads a specified amount of light curves to analyze.

    Parameters:
        n (int): Number of light curves to load.
        fits_files (list): A list of all the fits files

    Returns:
        light_curves_1 (list): A list of n light curves in 0.2-0.6 keV,
        light_curves_2 (list): A list of n light curves in 0.6-2.3keV
        light_curves_3 (list): A list of n light curves in 2.3-5.0keV
    """

    inaccess_files = read_inaccessible_lightcurves()


    if n != 'all':
        # Randomly select n files
        fits_files = random.sample(fits_files, n)

    temp = []
    for lc in  fits_files:
        if lc not in inaccess_files:
            temp.append(lc)
    fits_files = temp

    # Load all bands of the light curves into a list of DataFrames
    if band == 'all':
        print('starting 1st band')
        light_curves_1 = [df for df in (load_light_curve(file, band = 0, trunc = trunc) for file in fits_files) if df is not None]
        print('starting 2nd band')
        light_curves_2 = [df for df in (load_light_curve(file, band = 1, trunc = trunc) for file in fits_files) if df is not None]
        print('starting 3rd band')
        light_curves_3 = [df for df in (load_light_curve(file, band = 2, trunc = trunc) for file in fits_files) if df is not None]
        print('finished loading all bands')

        return light_curves_1, light_curves_2, light_curves_3
    elif band == 'low':
        total_files = len(fits_files)
        tenths = total_files // 10

        light_curves_1 = []
        for i, file in enumerate(fits_files):
            df = load_light_curve(file, band=0, trunc = trunc)
            if df is not None:
                light_curves_3.append(df)
            if total_files > 10:
                if (i + 1) % tenths == 0:
                    print(f"Processed {(i + 1) / total_files:.0%} of files")

        return light_curves_1
    elif band == 'med':
        total_files = len(fits_files)
        tenths = total_files // 10

        light_curves_2 = []
        for i, file in enumerate(fits_files):
            df = load_light_curve(file, band=1, trunc = trunc)
            if df is not None:
                light_curves_2.append(df)
            if total_files > 10:
                if (i + 1) % tenths == 0:
                    print(f"Processed {(i + 1) / total_files:.0%} of files")

        return light_curves_2
    elif band == 'high':
        total_files = len(fits_files)
        tenths = total_files // 10

        light_curves_3 = []
        for i, file in enumerate(fits_files):
            df = load_light_curve(file, band=2, trunc = trunc)
            if df is not None:
                light_curves_3.append(df)
            if total_files > 10:
                if (i + 1) % tenths == 0:
                    print(f"Processed {(i + 1) / total_files:.0%} of files")

        return light_curves_3
    else:
        raise KeyError("Input for Band is not valid")

def check_lightcurve_permissions(data_dir = '/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned'):
    """
    Check permissions for all light curve files in the given directory and save inaccessible ones to a file
    in the same directory as this script. Also checks to make sure that the light curves are not empty

    Args:
        data_dir (str): Path to the directory containing light curve files

    Returns:
        List[str]: List of files that could not be accessed
    """
    inaccessible_files = []

    # Check if the data directory exists and is accessible
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")

    if not os.access(data_dir, os.R_OK):
        raise PermissionError(f"No read permission for directory {data_dir}")

    # Get all FITS files in the directory
    fits_files = glob.glob(os.path.join(data_dir, "*.fits"))

    # Check each file for read permission

    for file_path in fits_files:
        if not os.access(file_path, os.R_OK): # read Permission
            inaccessible_files.append(file_path)
            continue
        with fits.open(file_path) as hdul:
            try:
                data = hdul[1].data
                if data is None or len(data) == 0:
                    inaccessible_files.append(file_path)
            except IndexError:
                print(f"Skipping file {file_path}: Index Out of Range")
                inaccessible_files.append(file_path)


    # Save inaccessible files to a text file in script directory
    output_file = os.path.join(os.getcwd(), "inaccessible_lightcurves.txt")
    with open(output_file, "w") as f:
        for file_path in inaccessible_files:
            f.write(f"{file_path}\n")

    print(f"Saved inaccessible files list to: {output_file}")
    return inaccessible_files

# Partition into train set and test set
def partition_data(light_curves, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Partition a list of light curves into train, validation, and test sets.

    Parameters:
        light_curves (list): List of light curve DataFrames.
        test_size (float): Proportion of data to use for the test set.
        val_size (float): Proportion of train data to use for the validation set.
        random_seed (int): Random seed for reproducibility.

    Returns:
        train_set (list): List of light curves for training.
        val_set (list): List of light curves for validation (if val_size > 0).
        test_set (list): List of light curves for testing.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Split into train+val and test sets
    train_val_set, test_set = train_test_split(light_curves, test_size=test_size, random_state=random_seed)

    if val_size > 0:
        # Split train_val into train and validation sets
        train_size = 1 - val_size
        train_set, val_set = train_test_split(train_val_set, test_size=val_size, random_state=random_seed)
        return train_set, val_set, test_set
    else:
        # If no validation set is needed, return only train and test sets
        return train_val_set, test_set


# Only run this if the script is run directly (not imported)
if __name__ == '__main__':
    # Example usage of the functions
    fits_files = load_all_fits_files()
    print(f"Found {len(fits_files)} FITS files")

    check_lightcurve_permissions()
