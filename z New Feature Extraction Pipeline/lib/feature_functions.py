"""
feature_functions.py — shared feature extraction logic

All functions in this module are used by both PATH A (run_feature_extraction.py)
and PATH B (batch_feature_extraction.py). Extracting them here eliminates the
~550-line duplication that previously existed between those two scripts.

Features extracted per light curve (11 total):
    weighted_mean, weighted_variance, lag1_autocorr, hurst_exp,
    mean_rise_fall_ratio, beyond1std, stetson_k, excess_var,
    bexvar, mean_var, ampl_sig
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
import subprocess
from scipy.stats import linregress, norm, mstats
from astropy.io import fits
from astropy.table import Table
import light_curve as Light_Curve_Package

# Import bexvar algorithm (ultranest nested sampling)
# Alias to avoid collision with the 'bexvar' feature name string used below
from bexvar_ero import lscg_gen, estimate_source_cr_marginalised, bexvar as _bexvar_nested

# FEATURES_FILE is needed only by df_process_all_light_curves_error to
# determine where to save SIG_NEV_mappings.pkl alongside the final features.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import FEATURES_FILE
from helper import DEFAULT_DATA_DIR


# ---------------------------------------------------------------------------
# Time-series helper functions
# ---------------------------------------------------------------------------

def lag1_autocorrelation(time_series):
    """
    Calculate the lag-1 autocorrelation of a time series.

    Parameters:
        time_series (array-like): Input time series data

    Returns:
        float: Lag-1 autocorrelation coefficient (0 if denominator is zero)
    """
    ts = np.array(time_series)
    ts_mean = np.mean(ts)
    ts_shifted = ts[1:]   # lagged by one element
    ts_original = ts[:-1] # original, excluding last element

    numerator = np.sum((ts_original - ts_mean) * (ts_shifted - ts_mean))
    denominator = np.sum((ts_original - ts_mean) ** 2)
    return numerator / denominator if denominator != 0 else 0


def hurst_exponent(time_series):
    """
    Calculate the Hurst exponent via rescaled range analysis.
    H > 0.5 = persistence; H < 0.5 = mean-reverting; H ≈ 0.5 = random walk.

    Parameters:
        time_series (array-like): Input time series data

    Returns:
        float: Hurst exponent (default 0.5 for short/invalid series)
    """
    ts = np.array(time_series)
    N = len(ts)

    if N < 4:
        return 0.5

    max_lag = min(N - 1, 100)
    lags = range(2, max_lag)

    tau = []
    valid_lags = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std_val = np.std(diff)
        if std_val > 0:
            tau.append(std_val)
            valid_lags.append(lag)

    if len(valid_lags) < 4:
        return 0.5

    try:
        reg = linregress(np.log(valid_lags), np.log(tau))
        return reg.slope * 2.0
    except (ValueError, RuntimeWarning):
        return 0.5


def rise_fall_ratio_over_time(time_series):
    """
    Calculate the rise/fall ratio across consecutive steps of the time series.

    Parameters:
        time_series (array-like): Input time series data

    Returns:
        float: Rise/fall ratio, capped at 10.0
    """
    ts = np.array(time_series)

    if len(ts) < 3:
        return 1.0

    rises = ts[1:] - ts[:-1]
    rise_count = np.sum(rises > 0)
    fall_count = np.sum(rises < 0)

    if fall_count == 0:
        return 1.0 if rise_count == 0 else 10.0
    return min(rise_count / fall_count, 10.0)


def compute_bexvar_via_cli(fits_path, band=0):
    """
    Run bexvar_ero.py as a subprocess and read back the scatter from the output FITS.
    Note: this function is kept for reference but is NOT called by
    df_extract_statistical_features_error — that function uses the faster
    in-process nested sampling approach instead.
    """
    subprocess.run(
        ["bexvar_ero.py", fits_path],
        stdout=subprocess.DEVNULL,
        check=True
    )
    out = f"{fits_path}-bexvar-{band}.fits"
    hdr = fits.getheader(out, ext=1)
    return hdr['SCATT'], hdr['SCATT_LO'], hdr['SCATT_HI']


# ---------------------------------------------------------------------------
# ampl_sig helper
# ---------------------------------------------------------------------------

def compute_ampl_sig(df):
    """
    Compute amplitude significance from an already-loaded light curve DataFrame.

    The DataFrame must already have RATE and SYM_ERR columns (SYM_ERR is
    computed near the top of df_extract_statistical_features_error before
    this function is called).

    Formula:
        ampl_sig = ((R_max - σ_max) - (R_min + σ_min)) / sqrt(σ_max² + σ_min²)

    Returns:
        float: amplitude significance, or 0.0 on failure / insufficient data
    """
    try:
        if df is None or len(df) < 3:
            return 0.0
        rates = df['RATE'].values
        errors = df['SYM_ERR'].values
        idx_min = np.argmin(rates)
        idx_max = np.argmax(rates)
        r_min, r_max = rates[idx_min], rates[idx_max]
        sigma_min, sigma_max = errors[idx_min], errors[idx_max]
        denom = np.sqrt(sigma_max ** 2 + sigma_min ** 2)
        if denom == 0:
            return 0.0
        return ((r_max - sigma_max) - (r_min + sigma_min)) / denom
    except Exception as e:
        print(f"[Warning] compute_ampl_sig failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def df_extract_statistical_features_error(df):
    """
    Extract 11 statistical features from a single light curve DataFrame.

    Features extracted (in order):
        weighted_mean, weighted_variance, lag1_autocorr, hurst_exp,
        mean_rise_fall_ratio, beyond1std, stetson_k, excess_var,
        bexvar, mean_var, ampl_sig

    Parameters:
        df (pd.DataFrame): Light curve with columns TIME, RATE, ERRM, ERRP.
                           Must have df.attrs['FILE_NAME'] set.

    Returns:
        tuple(pd.DataFrame, float): (one-row features DataFrame, sig_nev_value)
            or None on critical failure.
    """
    try:
        file_path = df.attrs.get('FILE_NAME', 'Unknown')

        required_columns = ['TIME', 'RATE', 'ERRM', 'ERRP']
        if not all(col in df.columns for col in required_columns):
            print(f"[Error] Missing required columns in file: {file_path}")
            return None

        # Symmetric error — must be computed before ampl_sig
        df['SYM_ERR'] = (df['ERRM'] + df['ERRP']) / 2

        feature_names = []
        feature_values = []

        # Store light curve data for downstream use
        light_curve = pd.DataFrame({
            'TIME': df['TIME'].values,
            'RATE': df['RATE'].values,
            'ERRM': df['ERRM'].values,
            'ERRP': df['ERRP'].values
        })

        # Arrays for light_curve package
        times  = np.asarray(df["TIME"].values,   dtype=np.float64)
        rates  = np.asarray(df["RATE"].values,   dtype=np.float64)
        errors = np.asarray(df["SYM_ERR"].values, dtype=np.float64)

        # 1. weighted_mean
        try:
            weights = 1 / df['SYM_ERR'] ** 2
            weighted_mean = np.sum(weights * df['RATE']) / np.sum(weights)
            feature_names.append('weighted_mean')
            feature_values.append(weighted_mean)
        except Exception:
            print(f"[Warning] Failed to compute weighted_mean for file: {file_path}")
            feature_names.append('weighted_mean')
            feature_values.append(0)

        # 2. weighted_variance
        try:
            weighted_var = np.sum(weights * (df['RATE'] - weighted_mean) ** 2) / np.sum(weights)
            feature_names.append('weighted_variance')
            feature_values.append(weighted_var)
        except Exception:
            print(f"[Warning] Failed to compute weighted_variance for file: {file_path}")
            feature_names.append('weighted_variance')
            feature_values.append(0)

        # 3. lag1_autocorr
        try:
            feature_names.append('lag1_autocorr')
            feature_values.append(lag1_autocorrelation(df['RATE']))
        except Exception:
            print(f"[Warning] Failed to compute lag1_autocorr for file: {file_path}")
            feature_names.append('lag1_autocorr')
            feature_values.append(0)

        # 4. hurst_exp
        try:
            feature_names.append('hurst_exp')
            feature_values.append(hurst_exponent(df['RATE']))
        except Exception:
            print(f"[Warning] Failed to compute hurst_exp for file: {file_path}")
            feature_names.append('hurst_exp')
            feature_values.append(0.5)

        # 5. mean_rise_fall_ratio
        try:
            feature_names.append('mean_rise_fall_ratio')
            feature_values.append(rise_fall_ratio_over_time(df['RATE']))
        except Exception:
            print(f"[Warning] Failed to compute mean_rise_fall_ratio for file: {file_path}")
            feature_names.append('mean_rise_fall_ratio')
            feature_values.append(1.0)

        # Instantiate light_curve package objects
        beyond1std_fn = Light_Curve_Package.BeyondNStd(nstd=1)
        stetson_k_fn  = Light_Curve_Package.StetsonK()
        mean_var_fn   = Light_Curve_Package.MeanVariance()

        # 6. beyond1std
        try:
            val = beyond1std_fn(times, rates, errors, sorted=True, check=False)
            feature_names.append('beyond1std')
            feature_values.append(float(val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute beyond1std for file: {file_path}, Error: {e}")
            feature_names.append('beyond1std')
            feature_values.append(0)

        # 7. stetson_k
        try:
            val = stetson_k_fn(times, rates, errors, sorted=True, check=False)
            feature_names.append('stetson_k')
            feature_values.append(float(val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute stetson_k for file: {file_path}, Error: {e}")
            feature_names.append('stetson_k')
            feature_values.append(0)

        # 8. excess_var (Normalized Excess Variance)
        # sig_nev_value is initialized here so a failure doesn't cause a NameError at return
        sig_nev_value = 0.0
        try:
            R_bar = np.mean(rates)
            N = len(rates)
            sigma_obs_sq = np.sum((rates - R_bar) ** 2) / (N - 1)
            sigma_err_sq_mean = np.mean(errors ** 2)
            sigma_xs_sq = sigma_obs_sq - sigma_err_sq_mean
            nev = max(sigma_xs_sq / (R_bar ** 2), 0.001)
            f_var = np.sqrt(nev)

            feature_names.append('excess_var')
            feature_values.append(nev)

            sig_nev = np.sqrt(
                2 / N * (sigma_err_sq_mean / R_bar ** 2) ** 2
                + sigma_err_sq_mean / N * (2 * f_var / R_bar) ** 2
            )
            sig_nev_value = nev / sig_nev if sig_nev != 0 else 0.0
        except Exception as e:
            print(f"[Warning] Failed to compute excess_var for file: {file_path}, Error: {e}")
            feature_names.append('excess_var')
            feature_values.append(0)

        # 9. bexvar (Bayesian excess variance via in-process nested sampling)
        try:
            band = 1
            fullpath = os.path.join(DEFAULT_DATA_DIR, df.attrs['FILE_NAME'])

            tab = Table.read(fullpath, hdu='RATE', format='fits')
            lc  = tab[tab['FRACEXP'][:, band] > 0.1]
            c   = lc['COUNTS'][:, band]
            bc  = lc['BACK_COUNTS'][:, band]
            bgarea    = 1.0 / lc['BACKRATIO']
            fe        = lc['FRACEXP'][:, band]
            rate_conv = fe * lc['TIMEDEL']

            log_grid = lscg_gen(c, bc, bgarea, rate_conv, density_gp=100)
            pdfs = np.vstack([
                estimate_source_cr_marginalised(log_grid, ci, bci, bga, rc)
                for ci, bci, bga, rc in zip(c, bc, bgarea, rate_conv)
            ])
            log_mean_samps, log_sigma_samps = _bexvar_nested(log_grid, pdfs)
            qs = norm().cdf([-1, 0, 1])
            _lo, med, _hi = mstats.mquantiles(log_sigma_samps, prob=qs)

            feature_names.append('bexvar')
            feature_values.append(med)
        except Exception as e:
            print(f"[Warning] Failed to compute bexvar for file: {file_path}, Error: {e}")
            feature_names.append('bexvar')
            feature_values.append(0)

        # 10. mean_var
        try:
            val = mean_var_fn(times, rates, errors, sorted=True, check=False)
            feature_names.append('mean_var')
            feature_values.append(float(val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute mean_var for file: {file_path}, Error: {e}")
            feature_names.append('mean_var')
            feature_values.append(0)

        # 11. ampl_sig (amplitude significance)
        # compute_ampl_sig requires SYM_ERR which was set above
        try:
            feature_names.append('ampl_sig')
            feature_values.append(compute_ampl_sig(df))
        except Exception as e:
            print(f"[Warning] Failed to compute ampl_sig for file: {file_path}, Error: {e}")
            feature_names.append('ampl_sig')
            feature_values.append(0.0)

        result = pd.DataFrame({
            'file_path':     [file_path],
            'feature_names': [feature_names],
            'feature_values': [np.array(feature_values, dtype=np.float64)],
            'light_curve':   [light_curve],
        })

        return result, sig_nev_value

    except Exception as e:
        print(f"[Error] Failed to process file {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch processing helpers (used by PATH A only; PATH B handles its own loop)
# ---------------------------------------------------------------------------

def df_process_all_light_curves_error(light_curves):
    """
    Process a list of light curve DataFrames sequentially and combine results.
    Also saves SIG_NEV_mappings.pkl alongside FEATURES_FILE.

    Note: This function is called by PATH A (run_feature_extraction.py) but NOT
    by PATH B (which uses process_chunk per-curve saving instead).

    Parameters:
        light_curves (list of pd.DataFrame)

    Returns:
        pd.DataFrame: Combined features DataFrame.
    """
    print(f"\nProcessing {len(light_curves)} light curves for feature extraction...")
    start_time = time.time()
    features_list = []
    sig_nev_mappings = {}

    for i, lc in enumerate(light_curves):
        if len(lc) < 3:
            print(f"  → Skipping curve {i} ({len(lc)} points)")
            continue
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(light_curves) - i) / rate if rate > 0 else 0
            print(f"Progress: {i}/{len(light_curves)} ({i/len(light_curves)*100:.1f}%) "
                  f"- {rate:.1f} curves/sec - Est. remaining: {remaining:.1f} sec")

        features, sig_nev = df_extract_statistical_features_error(lc)
        features_list.append(features)

        try:
            names = list(features['feature_names'][0])
            ev_idx = names.index('excess_var')
            sig_nev_mappings[features['file_path'][0]] = {
                'sig_nev':    sig_nev,
                'excess_var': features['feature_values'][0][ev_idx]
            }
        except Exception as e:
            print(f"Failed to extract sig_nev for {features['file_path'][0]}: {e}")

    result = pd.concat(features_list, ignore_index=True)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

    try:
        sig_nev_file = os.path.join(os.path.dirname(FEATURES_FILE), 'SIG_NEV_mappings.pkl')
        with open(sig_nev_file, 'wb') as f:
            pickle.dump(sig_nev_mappings, f)
        print(f"Saved SIG_NEV mappings to {sig_nev_file}")
    except Exception as e:
        print(f"Error saving SIG_NEV mappings: {e}")

    return result


def chunked(iterable, size):
    """Yield successive size-length chunks as (chunk, start_index) tuples."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size], i
