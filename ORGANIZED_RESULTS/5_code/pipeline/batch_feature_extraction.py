import numpy as np
from scipy import stats
from scipy.stats import linregress
import pandas as pd
import os
import time
from datetime import datetime
import pickle
from astropy.io import fits
import light_curve as Light_Curve_Package
from sklearn.preprocessing import RobustScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

import subprocess
from astropy.io import fits
# import sqlite3

import os, sys
# make sure libs/ is on sys.path
# HERE = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(HERE, "libs", "bexvar"))
from bexvar_ero import lscg_gen, estimate_source_cr_marginalised, bexvar
from pathlib import Path

from scipy.stats import norm, mstats
from astropy.table import Table

import sys, site
print("PYTHON:", sys.executable)
print("PATHS:", sys.path)
print("SITE-PACKAGES:", site.getsitepackages())

from config import (
    FEATURES_FILE,
    DEFAULT_BAND,
    KNOWN_LIGHT_CURVES,
    LOAD_SIZE,
    PROCESSED_DATA_DIR
)

from helper import (
    load_light_curve,
    load_n_light_curves,
    load_all_fits_files,
    create_dataframe_of_light_curves,
    read_inaccessible_lightcurves,
    DEFAULT_DATA_DIR
)

import argparse
# DB_PATH = os.path.join(PROCESSED_DATA_DIR, "features.sqlite")


# Define functions for time series analysis
def lag1_autocorrelation(time_series):
    """
    Calculate the lag-1 autocorrelation of a time series.

    Parameters:
        time_series (array-like): Input time series data

    Returns:
        float: Lag-1 autocorrelation coefficient
    """
    ts = np.array(time_series)
    ts_mean = np.mean(ts)
    ts_shifted = ts[1:]  # Shifted by one, removing first element so "lagged"
    ts_original = ts[:-1]  # Original time series, excluding last element

    # Measurement of covariance between the original values and the lagged values, relative to the mean
    numerator = np.sum((ts_original - ts_mean) * (ts_shifted - ts_mean))
    denominator = np.sum((ts_original - ts_mean) ** 2)  # Variance of original time series
    return numerator / denominator if denominator != 0 else 0

def hurst_exponent(time_series):
    """
    Calculate the Hurst exponent of a time series using rescaled range analysis.
    H > 0.5 indicates persistence, H < 0.5 indicates mean-reverting behavior.

    Parameters:
        time_series (array-like): Input time series data

    Returns:
        float: Hurst exponent
    """
    ts = np.array(time_series)
    N = len(ts)

    # Need at least 4 points to calculate Hurst exponent
    if N < 4:
        return 0.5  # Return neutral value for very short series

    # Consider lags from 2 to min(N-1, 100) to avoid excessive computation for large series
    max_lag = min(N-1, 100)
    lags = range(2, max_lag)

    # Calculate tau values and filter out zeros and negative values
    tau = []
    valid_lags = []

    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std_val = np.std(diff)
        if std_val > 0:  # Only keep positive standard deviations
            tau.append(std_val)
            valid_lags.append(lag)

    # If we don't have enough valid points, return default value
    if len(valid_lags) < 4:
        return 0.5

    # Linear fit to estimate the Hurst exponent
    try:
        reg = linregress(np.log(valid_lags), np.log(tau))
        return reg.slope * 2.0  # Hurst exponent
    except (ValueError, RuntimeWarning):
        # If regression fails, return neutral value
        return 0.5

def rise_fall_ratio_over_time(time_series):
    """
    Calculate the rise/fall ratio at every step of the time series.

    Parameters:
        time_series (array-like): Input time series data

    Returns:
        float: Mean rise/fall ratio (excluding undefined values)
    """
    ts = np.array(time_series)

    # If time series is too short, return neutral value
    if len(ts) < 3:
        return 1.0

    # Calculate differences between consecutive elements
    rises = ts[1:] - ts[:-1]

    rise_count = np.sum(rises > 0)  # Number of positive changes
    fall_count = np.sum(rises < 0)  # Number of negative changes

    # Calculate the overall rise/fall ratio and handle division by zero
    if fall_count == 0:
        if rise_count == 0:
            return 1.0  # Neutral value if no changes
        else:
            return 10.0  # Cap at a high value if only rises
    else:
        return min(rise_count / fall_count, 10.0)  # Cap at 10.0 to avoid extreme values

def compute_bexvar_via_cli(fits_path, band=0):
    # 1) run the installed script on the command line
    subprocess.run(
        ["bexvar_ero.py", fits_path],   # or full path to be safe
        stdout=subprocess.DEVNULL,
        check=True
    )
    # 2) read back its output FITS
    out = f"{fits_path}-bexvar-{band}.fits"
    hdr = fits.getheader(out, ext=1)
    return hdr['SCATT'], hdr['SCATT_LO'], hdr['SCATT_HI']

def df_extract_statistical_features_error(df):
    """Extract statistical features from a single light curve with error handling"""
    try:
        # Store file path
        file_path = df.attrs.get('FILE_NAME', 'Unknown')

        # Check required columns
        required_columns = ['TIME', 'RATE', 'ERRM', 'ERRP']
        if not all(col in df.columns for col in required_columns):
            print(f"[Error] Missing required columns in file: {file_path}")
            return None

        N = len(df)
        # Check enough data points
        # if N < 3:
        #     print(f"[Error] Not enough data points in file: {file_path}")
        #     return None

        # Calculate symmetric error
        df['SYM_ERR'] = (df['ERRM'] + df['ERRP']) / 2

        # Initialize feature list
        feature_names = []
        feature_values = []

        # Store light curve data
        light_curve = pd.DataFrame({
            'TIME': df['TIME'].values,
            'RATE': df['RATE'].values,
            'ERRM': df['ERRM'].values,
            'ERRP': df['ERRP'].values
        })

        # Convert data to numpy arrays for Light_Curve_Package
        times = np.asarray(df["TIME"].values, dtype=np.float64)
        rates = np.asarray(df["RATE"].values, dtype=np.float64)
        errors = np.asarray(df["SYM_ERR"].values, dtype=np.float64)

        # Basic statistics with error handling
        try:
            weights = 1 / df['SYM_ERR']**2
            weighted_mean = np.sum(weights * df['RATE']) / np.sum(weights)
            feature_names.append('weighted_mean')
            feature_values.append(weighted_mean)
        except:
            print(f"[Warning] Failed to compute weighted_mean for file: {file_path}")
            feature_names.append('weighted_mean')
            feature_values.append(0)

        # Add other features with error handling
        try:
            weighted_var = np.sum(weights * (df['RATE'] - weighted_mean)**2) / np.sum(weights)
            feature_names.append('weighted_variance')
            feature_values.append(weighted_var)
        except:
            print(f"[Warning] Failed to compute weighted_variance for file: {file_path}")
            feature_names.append('weighted_variance')
            feature_values.append(0)

        # Add time series features
        try:
            lag1_autocorr = lag1_autocorrelation(df['RATE'])
            feature_names.append('lag1_autocorr')
            feature_values.append(lag1_autocorr)
        except:
            print(f"[Warning] Failed to compute lag1_autocorr for file: {file_path}")
            feature_names.append('lag1_autocorr')
            feature_values.append(0)

        try:
            hurst_exp = hurst_exponent(df['RATE'])
            feature_names.append('hurst_exp')
            feature_values.append(hurst_exp)
        except:
            print(f"[Warning] Failed to compute hurst_exp for file: {file_path}")
            feature_names.append('hurst_exp')
            feature_values.append(0.5)  # Neutral value for Hurst exponent

        try:
            mean_rise_fall = rise_fall_ratio_over_time(df['RATE'])
            feature_names.append('mean_rise_fall_ratio')
            feature_values.append(mean_rise_fall)
        except:
            print(f"[Warning] Failed to compute mean_rise_fall_ratio for file: {file_path}")
            feature_names.append('mean_rise_fall_ratio')
            feature_values.append(1.0)  # Neutral value for rise/fall ratio

        # Initialize Light_Curve_Package features
        beyond1std = Light_Curve_Package.BeyondNStd(nstd=1)
        stetson_k = Light_Curve_Package.StetsonK()
        # excess_var = Light_Curve_Package.ExcessVariance()
        mean_var = Light_Curve_Package.MeanVariance()

        # Add Light_Curve_Package features with error handling
        try:
            beyond1std_val = beyond1std(times, rates, errors, sorted=True, check=False)
            feature_names.append('beyond1std')
            feature_values.append(float(beyond1std_val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute beyond1std for file: {file_path}, Error: {str(e)}")
            feature_names.append('beyond1std')
            feature_values.append(0)

        try:
            stetson_k_val = stetson_k(times, rates, errors, sorted=True, check=False)
            feature_names.append('stetson_k')
            feature_values.append(float(stetson_k_val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute stetson_k for file: {file_path}, Error: {str(e)}")
            feature_names.append('stetson_k')
            feature_values.append(0)

        try:
            # ex_var_val = excess_var(times, rates, errors, sorted=True, check=False)
            # feature_names.append('excess_var')
            # feature_values.append(float(ex_var_val[0]))

            # Original Excess Variance with clipping
            # ex_var_val = excess_var(times, rates, errors, sorted=True, check=False)
            # Manual calculation of Normalized Excess Variance (NEV) and related quantities
            # Formula 10: Mean count rate (R̄_S)
            R_bar = np.mean(rates)

            # Formula 11: Observed variance (σ²_obs)
            N = len(rates)

            # Formula 11: σ²_obs = (1/(N-1)) * Σ(R_i - R̄)²
            sigma_obs_sq = np.sum((rates - R_bar) ** 2) / (N - 1)

            # Formula 12: Mean squared error (σ̄²_err)
            # Using symmetric errors as calculated earlier (df['SYM_ERR'])
            sigma_err_sq_mean = np.mean(errors ** 2)

            # Formula 13: Excess variance (σ²_XS)
            sigma_xs_sq = sigma_obs_sq - sigma_err_sq_mean

            # Formula 14: Normalized Excess Variance (NEV)
            nev = sigma_xs_sq / (R_bar ** 2)

            # Clip to minimum 0.001 as per Buchner et al.
            nev = max(nev, 0.001)

            # Formula 15: Fractional rms amplitude (F_var)
            f_var = np.sqrt(nev)

            feature_names.append('excess_var')
            feature_values.append(nev)

            # Calculate SIG_NEV (significance of NEV) Formula 16
            sig_nev = np.sqrt(2/N * (sigma_err_sq_mean /R_bar**2) ** 2 + sigma_err_sq_mean / N * (2 * f_var / R_bar) ** 2)
            sig_nev_value = nev / sig_nev if sig_nev != 0 else 0.0


        except Exception as e:
            print(f"[Warning] Failed to compute excess_var for file: {file_path}, Error: {str(e)}")
            feature_names.append('excess_var')
            feature_values.append(0)

        try:

            # Somewhere inside your feature-extraction function:
            # scatt, scatt_lo, scatt_hi = compute_bexvar_via_cli(DEFAULT_DATA_DIR + "/" + file_path, band=1)

            # 1) build the per‐curve inputs exactly as bexvar_ero.py does:
            #    a) pick out the same HDU & band, apply FRACEXP>0.1, etc.
            band = 1
            fullpath = os.path.join(DEFAULT_DATA_DIR, df.attrs['FILE_NAME'])

            tab = Table.read(fullpath, hdu='RATE', format='fits')
            lc = tab[tab['FRACEXP'][:, band] > 0.1]
            c  = lc['COUNTS'][:, band]
            bc = lc['BACK_COUNTS'][:, band]
            bgarea = 1.0 / lc['BACKRATIO']
            fe = lc['FRACEXP'][:, band]
            rate_conv = fe * lc['TIMEDEL']

            # 2) generate the log‐grid
            log_grid = lscg_gen(c, bc, bgarea, rate_conv, density_gp=100)

            # 3) compute per‐bin posterior PDFs
            pdfs = np.vstack([
                estimate_source_cr_marginalised(log_grid, ci, bci, bga, rc)
                for ci, bci, bga, rc in zip(c, bc, bgarea, rate_conv)
            ])

            # 4) run the nested sampler in‐process
            log_mean_samps, log_sigma_samps = bexvar(log_grid, pdfs)

            # 5) summarize the scatter posterior (e.g. 16/50/84 percentiles)
            qs = norm().cdf([-1, 0, 1])
            lo, med, hi = mstats.mquantiles(log_sigma_samps, prob=qs)

            # feature_names.extend(['bexvar_scatt', 'bexvar_scatt_lo', 'bexvar_scatt_hi'])
            # feature_values.extend([med, lo, hi])

            feature_names.append("bexvar")
            feature_values.append(med)
            # feature_names.append("bexvar_scatt_lo")
            # feature_values.append(scatt_lo)
            # feature_names.append("bexvar_scatt_hi")
            # feature_values.append(scatt_hi)
        except Exception as e:
            print(f"[Warning] Failed to compute bexvar for file: {file_path}, Error: {str(e)}")
            feature_names.append('bexvar')
            feature_values.append(0)

        try:
            m_var_val = mean_var(times, rates, errors, sorted=True, check=False)
            feature_names.append('mean_var')
            feature_values.append(float(m_var_val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute mean_var for file: {file_path}, Error: {str(e)}")
            feature_names.append('mean_var')
            feature_values.append(0)

        # Create result DataFrame
        result = pd.DataFrame({
            'file_path': [file_path],
            'feature_names':[feature_names],
            'feature_values': [np.array(feature_values, dtype=np.float64)],
            # 'features_values': [feature_values],
            'light_curve': [light_curve]  # Store the light curve data
        })

        return result, sig_nev_value

    except Exception as e:
        print(f"[Error] Failed to process file {file_path}: {str(e)}")
        return None

def df_process_all_light_curves_error(light_curves):
    """
    Process a list of DataFrames containing light curve data and combine results.

    Parameters:
        light_curves (list of pd.DataFrame): List of light curve DataFrames.

    Returns:
        pd.DataFrame: Combined DataFrame with 'file_path', 'feature_names', and 'feature_values'.
    """
    print(f"\nProcessing {len(light_curves)} light curves for feature extraction...")
    start_time = time.time()
    features_list = []

    sig_nev_mappings = {}

    # Add progress tracking
    for i, lc in enumerate(light_curves):
        # skip any curve with fewer than 3 data points
        if len(lc) < 3:
            print(f"  → Skipping curve {i} ({len(lc)} points)")
            continue
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(light_curves) - i) / rate if rate > 0 else 0
            print(f"Progress: {i}/{len(light_curves)} ({i/len(light_curves)*100:.1f}%) - {rate:.1f} curves/sec - Est. remaining: {remaining:.1f} sec")
        features, sig_nev = df_extract_statistical_features_error(lc)
        features_list.append(features)
        # print(sig_nev)
        # print(features['feature_values'][0][-2])
        # print(features['file_path'][0])
        try:
            sig_nev_mappings[features['file_path'][0]] = {
                            'sig_nev': sig_nev,
                            'excess_var': features['feature_values'][0][-2]
                        }
        except Exception as e:
            print(f"Failed to extract sig_nev for file: {features['file_path'][0]} \n Exception: {e}")

    # Concatenate all rows into a single DataFrame
    result = pd.concat(features_list, ignore_index=True)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

    try:
        # Save SIG_NEV mappings to file
        sig_nev_file = os.path.join(os.path.dirname(FEATURES_FILE), 'SIG_NEV_mappings.pkl')
        with open(sig_nev_file, 'wb') as f:
            pickle.dump(sig_nev_mappings, f)
        print(f"Saved SIG_NEV mappings to {sig_nev_file}")
    except Exception as e:
        print(f"Error saving SIG_NEV mappings: {str(e)}")
    return result

def chunked(iterable, size):
    """Yield successive size-length chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size], i

def process_chunk(chunk_light_curves, band, output_dir):
    """
    Process one chunk of light curves (a list of light curve DataFrames),
    extract features from each, and save each result as a separate pickle file
    named after the light curve FITS file.
    """
    pid = os.getpid()
    print(f"[PID {pid}] Starting chunk with {len(chunk_light_curves)} light curves…")

    for i, lc in enumerate(chunk_light_curves):
        try:
            file_name = lc.attrs.get("FILE_NAME", f"unknown_{i}")
            print(f"[{i}/{len(chunk_light_curves)}] Processing {file_name}...")

            result, sig_nev = df_extract_statistical_features_error(lc)
            if result is None:
                print(f"[Warning] Skipping {file_name} due to processing error.")
                continue

            # Sanitize file name and save result
            base_name = Path(file_name).name.replace(".fits", "").replace("/", "_")
            out_path = os.path.join(output_dir, f"features_{base_name}.pkl")
            result.to_pickle(out_path)
            print(f"  → Saved features to {out_path}")

        except Exception as e:
            print(f"[Error] Failed to process curve {i}: {e}")

    print(f"[PID {pid}] Finished chunk")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--chunk-file', type=str, required=True,
                   help="Path to pickled light curve chunk (.pkl)")
    p.add_argument('--chunk-id', type=int, default=0,
                   help="Index to label output files")
    p.add_argument('--output-dir', type=str, default=PROCESSED_DATA_DIR,
                   help="Directory to save processed features")
    return p.parse_args()

def main():
    """Main function to process all light curves and save features."""

    print("Starting feature extraction pipeline...")

    # Load all light curves
    # print("Loading light curves...")
    # try:
    #     fits_files  = load_all_fits_files()
    #     if not fits_files:
    #         raise ValueError("No fits files were loaded successfully")
    # except Exception as e:
    #     print(f"Error loading fits files: {str(e)}")
    #     return

    # light_curves = load_n_light_curves(LOAD_SIZE, fits_files, band = DEFAULT_BAND)

    # print(f"Successfully loaded {len(light_curves)} light curves")

    args = parse_args()
    chunk_file = args.chunk_file
    chunk_idx = args.chunk_id
    output_dir = args.output_dir

    print(f"Loading light curves from chunk: {chunk_file}")
    with open(chunk_file, "rb") as f:
        light_curves = pickle.load(f)
    print(f"Loaded {len(light_curves)} light curves from chunk.")

    # 3) Choose chunk size (e.g., 100 curves per chunk)
    chunk_size = 1
    # Directory where each chunk writes out: features_chunk_<idx>.pkl
    out_dir = PROCESSED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Build a set of chunk indices already done
    existing = glob.glob(os.path.join(out_dir, "features_chunk_*.pkl"))
    done_idxs = {int(Path(fn).stem.split("_")[-1]) for fn in existing}

    # 4) Use ProcessPoolExecutor to run up to 48 chunks in parallel
    chunk_files = []
    with ProcessPoolExecutor(max_workers=85) as exe:
        futures = {}
        # Submit each chunk to the pool
        for chunk, chunk_idx in chunked(light_curves, chunk_size):
            # only this job’s share
            # print(chunk_idx)
            # if (chunk_idx % num_jobs) != job_id:
            #     continue

            # print(f"[Job {job_id}] Submitting chunk {chunk_idx}")
            fut = exe.submit(
                process_chunk, chunk, DEFAULT_BAND, out_dir
            )
            futures[fut] = chunk_idx


        # 4) As each future finishes, record its output
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                if result:
                    chunk_files.append(result)
                    print(f"✓ Chunk {idx} completed → {result}")
                else:
                    print(f"⚠️  Chunk {idx} returned None")
            except Exception as e:
                print(f"✗ Chunk {idx} failed:", e)

    # 5) Combine all chunk files (old + new)
    # dfs = []
    # for fn in sorted(chunk_files, key=lambda f: int(Path(f).stem.split("_")[-1])):
    #     try:
    #         dfs.append(pd.read_pickle(fn))
    #     except Exception as e:
    #         print(f"Failed to read {fn}: {e}")
    # if not dfs:
    #     print("No chunk files found—nothing to combine.")
    #     return

    # final_df = pd.concat(dfs, ignore_index=True)
    # os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    # final_df.to_pickle(FEATURES_FILE)
    # print(f"\nAll chunks combined into {FEATURES_FILE}")

    # features_df = final_df

    # # Print sample of the features
    # if len(features_df) > 0:
    #     print("\nSample of extracted features:")
    #     sample = features_df.iloc[0]
    #     print(f"File: {sample['file_path']}")
    #     for name, value in zip(sample['feature_names'], sample['feature_values']):
    #         print(f"- {name}: {value:.6f}")

    #     # Print statistics about feature extraction
    #     success_rate = len(features_df) / len(light_curves) * 100
    #     print(f"\nFeature extraction success rate: {success_rate:.1f}%")

    #     if success_rate < 100:
    #         print("Note: Some light curves were skipped due to errors. Check the log for details.")

if __name__ == "__main__":
    main()
