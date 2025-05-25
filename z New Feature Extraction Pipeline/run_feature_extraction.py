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

from config import (
    FEATURES_FILE,
    DEFAULT_BAND,
    KNOWN_LIGHT_CURVES,
    LOAD_SIZE
)

from helper import (
    load_light_curve,
    load_n_light_curves,
    load_all_fits_files,
    create_dataframe_of_light_curves,
    read_inaccessible_lightcurves
)

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

        # Check enough data points
        if len(df) < 3:
            print(f"[Error] Not enough data points in file: {file_path}")
            return None

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
        excess_var = Light_Curve_Package.ExcessVariance()
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
            ex_var_val = excess_var(times, rates, errors, sorted=True, check=False)
            feature_names.append('excess_var')
            feature_values.append(float(ex_var_val[0]))
        except Exception as e:
            print(f"[Warning] Failed to compute excess_var for file: {file_path}, Error: {str(e)}")
            feature_names.append('excess_var')
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

        return result

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

    # Add progress tracking
    for i, lc in enumerate(light_curves):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(light_curves) - i) / rate if rate > 0 else 0
            print(f"Progress: {i}/{len(light_curves)} ({i/len(light_curves)*100:.1f}%) - {rate:.1f} curves/sec - Est. remaining: {remaining:.1f} sec")
        features_list.append(df_extract_statistical_features_error(lc))

    # Concatenate all rows into a single DataFrame
    result = pd.concat(features_list, ignore_index=True)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return result

def main():
    """Main function to process all light curves and save features."""
    print("Starting feature extraction pipeline...")

    # Load all light curves
    print("Loading light curves...")
    try:
        fits_files  = load_all_fits_files()
        if not fits_files:
            raise ValueError("No fits files were loaded successfully")
    except Exception as e:
        print(f"Error loading fits files: {str(e)}")
        return

    light_curves = load_n_light_curves(LOAD_SIZE, fits_files, band = DEFAULT_BAND)

    print(f"Successfully loaded {len(light_curves)} light curves")

    # Process light curves and extract features
    try:
        features_df = df_process_all_light_curves_error(light_curves)
        if len(features_df) == 0:
            raise ValueError("No features were extracted successfully")
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        return

    # Save features to file
    print(f"\nSaving features to {FEATURES_FILE}...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
        features_df.to_pickle(FEATURES_FILE)
    except Exception as e:
        print(f"Error saving features: {str(e)}")
        return

    print(f"\nFeature extraction completed successfully:")
    print(f"- Processed {len(features_df)} light curves")
    print(f"- Features saved to {FEATURES_FILE}")

    # Print sample of the features
    if len(features_df) > 0:
        print("\nSample of extracted features:")
        sample = features_df.iloc[0]
        print(f"File: {sample['file_path']}")
        for name, value in zip(sample['feature_names'], sample['feature_values']):
            print(f"- {name}: {value:.6f}")

        # Print statistics about feature extraction
        success_rate = len(features_df) / len(light_curves) * 100
        print(f"\nFeature extraction success rate: {success_rate:.1f}%")

        if success_rate < 100:
            print("Note: Some light curves were skipped due to errors. Check the log for details.")

if __name__ == "__main__":
    main()
