import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from astropy.io import fits
import random
import math
import time
from datetime import datetime
import seaborn as sns
from scipy.stats import linregress
import light_curve as Light_Curve_Package

from helper import (
    load_light_curve,
    load_n_light_curves,
    load_all_fits_files,
    create_dataframe_of_light_curves,
    DEFAULT_DATA_DIR
)

# known interesting light curves
known_light_curves = ["em01_211120_020_LightCurve_00007_c010_rebinned.fits",
                        "em01_039135_020_LightCurve_00058_c010_rebinned.fits",
                        "em01_038099_020_LightCurve_00005_c010_rebinned.fits"]

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
    # Print file being processed
    # print(f"Processing file: {df.attrs.get('FILE_NAME', 'Unknown')}")
    """
    Extract statistical features from the light curve, accounting for measurement errors,
    and return as a single-row DataFrame.

    Parameters:
        df (pd.DataFrame): Light curve DataFrame with 'RATE', 'ERRM', and 'ERRP'.

    Returns:
        pd.DataFrame: A DataFrame containing 'file_path', 'feature_names', and 'feature_values'.
    """
    # Weighted Mean (Symmetric Errors)
    weights = 1 / df['SYM_ERR']**2  # Inverse variance weighting
    weighted_mean = np.sum(df['RATE'] * weights) / np.sum(weights)

    # Weighted Variance
    weighted_variance = np.sum(weights * (df['RATE'] - weighted_mean)**2) / np.sum(weights)

    # Robust Statistics
    median = np.median(df['RATE'])
    # iqr = np.percentile(df['RATE'], 75) - np.percentile(df['RATE'], 25)

    # Beyond 1 sigma (error-aware)
    # beyond_1_sigma = np.sum(np.abs(df['RATE'] - weighted_mean) > np.sqrt(weighted_variance)) / len(df['RATE'])

    # Median Absolute Deviation (x)
    # mad = stats.median_abs_deviation(df['RATE'])

    # Skewness and Kurtosis
    # skewness = ((df['RATE'] - weighted_mean)**3 * weights).sum() / (weights.sum() * weighted_variance**1.5)
    # kurtosis = ((df['RATE'] - weighted_mean)**4 * weights).sum() / (weights.sum() * weighted_variance**2)

    # Additional Features
    max_rate = df['RATE'].max()
    min_rate = df['RATE'].min()
    # max_amp = max_rate - min_rate

    # Flux Percentile Ratio (95th - 5th percentile)
    # flux_percentile_ratio = np.percentile(df['RATE'], 95) - np.percentile(df['RATE'], 5)

    # Time series analysis features
    lag1_autocorr = lag1_autocorrelation(df['RATE'])
    hurst_exp = hurst_exponent(df['RATE'])
    mean_rise_fall_ratio = rise_fall_ratio_over_time(df['RATE'])

    # Light_Curve_Package
    mag_ratio = Light_Curve_Package.MagnitudePercentageRatio()
    bey_1std = Light_Curve_Package.BeyondNStd(nstd=1)
    # perc_amp = Light_Curve_Package.PercentAmplitude()
    # max_slope = Light_Curve_Package.MaximumSlope()
    stetson_k = Light_Curve_Package.StetsonK()
    excess_var = Light_Curve_Package.ExcessVariance()
    mean_var = Light_Curve_Package.MeanVariance()

    count = 3
    feature_n = ["mag_ratio", "beyond1std", "stetson_k", "excess_var", "mean_var"]

    try:
        mr = mag_ratio(np.asarray(df["TIME"].values, dtype=np.float64),
                            np.asarray(df["RATE"].values, dtype=np.float64),
                            np.asarray(df["SYM_ERR"].values, dtype=np.float64),
                            sorted=True,
                            check=False)[0]
    except:
        print(f"[Warning] Failed to compute magnitude_ratio for file: {df.attrs.get('FILE_NAME', 'Unknown')}")
        mr = 0

    try:
        beyond_1std = bey_1std(np.asarray(df["TIME"].values, dtype=np.float64),
                            np.asarray(df["RATE"].values, dtype=np.float64),
                            np.asarray(df["SYM_ERR"].values, dtype=np.float64),
                            sorted=True,
                         check=False)[0]
    except:
        print(f"[Warning] Failed to compute beyond_1std for file: {df.attrs.get('FILE_NAME', 'Unknown')}")
        beyond_1std = 0

    # try:
    #     maximum_slope = max_slope(np.asarray(df["TIME"].values, dtype=np.float64),
    #                         np.asarray(df["RATE"].values, dtype=np.float64),
    #                         np.asarray(df["SYM_ERR"].values, dtype=np.float64),
    #                         sorted=True,
    #                      check=False)
    # except:
    #     maximum_slope = 0

    try:
        sk = stetson_k(np.asarray(df["TIME"].values, dtype=np.float64),
                            np.asarray(df["RATE"].values, dtype=np.float64),
                            np.asarray(df["SYM_ERR"].values, dtype=np.float64),
                            sorted=True,
                         check=False)[0]
    except:
        print(f"[Warning] Failed to compute stetson_k for file: {df.attrs.get('FILE_NAME', 'Unknown')}")
        sk = 0

    try:
        ex_var = excess_var(np.asarray(df["TIME"].values, dtype=np.float64),
                            np.asarray(df["RATE"].values, dtype=np.float64),
                            np.asarray(df["SYM_ERR"].values, dtype=np.float64),
                            sorted=True,
                         check=False)[0]
    except:
        print(f"[Warning] Failed to compute excess_var for file: {df.attrs.get('FILE_NAME', 'Unknown')}")
        ex_var = 0

    try:
        m_var = mean_var(np.asarray(df["TIME"].values, dtype=np.float64),
                            np.asarray(df["RATE"].values, dtype=np.float64),
                            np.asarray(df["SYM_ERR"].values, dtype=np.float64),
                            sorted=True,
                         check=False)[0]
    except:
        print(f"[Warning] Failed to compute mean_variance for file: {df.attrs.get('FILE_NAME', 'Unknown')}")
        m_var = 0





    mr = float(mr)
    beyond_1std = float(beyond_1std)
    # maximum_slope = float(maximum_slope[0])
    sk = float(sk)
    ex_var = float(ex_var)
    m_var = float(m_var)
    features = [mr, beyond_1std, sk, ex_var, m_var]


    # print(features)
    # print('\n'.join(f"{name} = {value:.2f}" for name, value in zip(extractor.names, features)))


    # Create a row with features
    feature_names = [
        "weighted_mean", "weighted_variance", "median",
        "lag1_autocorr", "hurst_exp",
        "mean_rise_fall_ratio", "mag_ratio", "beyond1std", "stetson_k", "excess_var", "mean_var"
    ]

    feature_values = np.array([
        weighted_mean, weighted_variance, median,
        lag1_autocorr, hurst_exp,
        mean_rise_fall_ratio, mr, beyond_1std, sk, ex_var, m_var
    ])

    # Create the DataFrame row
    return pd.DataFrame({
        "file_path": [df.attrs['FILE_NAME']],
        "feature_names": [feature_names],
        "feature_values": [feature_values]
    })

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

# This method works with a pandas dataframe consisting of more information
def df_detect_outliers_extreme(df, contamination=0.05, known_light_curves=known_light_curves):
    print(f"\nStarting outlier detection with contamination={contamination}...")
    start_time = time.time()
    """
    Detect outliers in rows of a DataFrame, where each row represents a light curve.
    Each row contains:
      - 'file_path': The file path of the light curve
      - 'feature_names': List of feature names
      - 'feature_values': Numpy array of numerical feature values

    Args:
    - df (pd.DataFrame): Input DataFrame with light curves.
    - contamination (float): The proportion of data expected to be outliers.

    Returns:
    - pd.DataFrame: Original DataFrame with additional columns:
        - 'is_outlier': Boolean flag indicating if the row is an outlier.
        - 'scaled_features': Scaled feature values for further analysis.
        - 'iso_score': Anomaly score from Isolation Forest.
        - 'lof_score': Negative outlier factor from LOF.
        - 'iqr_outlier': Boolean indicating IQR-based outlier.
        - 'z_score_outlier': Boolean indicating Z-score-based outlier.
        - 'combined_outlier': Combined outlier flag across methods.
        - 'outlier_rank': Rank of combined outliers (1 = most outlier-like).
    """
    # Ensure required columns are present
    required_columns = ['file_path', 'feature_names', 'feature_values']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare features for outlier detection
    feature_matrix = []
    for idx, row in df.iterrows():
        feature_values = row['feature_values']
        if not isinstance(feature_values, np.ndarray):
            raise ValueError(f"Row {idx}: 'feature_values' must be a numpy array.")
        feature_matrix.append(feature_values)

    # Convert to numpy array for processing
    feature_matrix = np.vstack(feature_matrix)

    # Scale features
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Isolation Forest
    print("Running Isolation Forest algorithm...")
    iso_start = time.time()
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        max_samples='auto',
        n_estimators=250
    )
    iso_forest.fit(scaled_features)
    iso_pred = iso_forest.predict(scaled_features)
    iso_scores = iso_forest.decision_function(scaled_features)
    print(f"Isolation Forest completed in {time.time() - iso_start:.2f} seconds")\

    # Local Outlier Factor
    print("Running Local Outlier Factor algorithm...")
    lof_start = time.time()
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        metric='euclidean',
        p=2,
        novelty=False
    )
    lof_scores = lof.fit_predict(scaled_features)

    lof_neg_scores = -lof.negative_outlier_factor_  # Higher = more anomalous
    print(f"Local Outlier Factor completed in {time.time() - lof_start:.2f} seconds")

    # Calculate feature importance using LOF
    print("Calculating feature importance...")
    feat_imp_start = time.time()
    feature_importances = []
    num_features = scaled_features.shape[1]
    for i in range(num_features):
        # Remove one feature at a time
        reduced_features = np.delete(scaled_features, i, axis=1)

        # Recompute LOF on reduced feature set
        lof_reduced = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            metric='euclidean',
            p=2,
            novelty=False
        )
        lof_reduced.fit(reduced_features)
        lof_reduced_neg_scores = -lof_reduced.negative_outlier_factor_

        # Measure the difference in LOF scores
        score_difference = np.mean(np.abs(lof_neg_scores - lof_reduced_neg_scores))
        # if (i+1) % 2 == 0:
        #     print(f"  Feature importance progress: {i+1}/{num_features}")
        feature_importances.append((df['feature_names'][0][i], score_difference))

    # Rank features by importance
    sorted_feature_importances = sorted(
        feature_importances, key=lambda x: x[1], reverse=True
    )
    print("Feature importance ranking:")
    for feat, imp in sorted_feature_importances[:5]:
        print(f"  - {feat}: {imp:.4f}")

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame(sorted_feature_importances, columns=['feature', 'importance'])

    # IQR method
    iqr_mult = 5
    q1 = np.percentile(scaled_features, 25, axis=0)
    q3 = np.percentile(scaled_features, 75, axis=0)
    iqr = q3 - q1
    iqr_outliers = np.any(
        (scaled_features < (q1 - iqr_mult * iqr)) |
        (scaled_features > (q3 + iqr_mult * iqr)),
        axis=1
    )

    # Z-score method
    z_scores = np.abs((scaled_features - np.mean(scaled_features, axis=0)) /
                      np.std(scaled_features, axis=0))
    z_score_outliers = np.any(z_scores > 4, axis=1)

    # Combine outlier predictions
    combined_outliers = (
        ((iso_pred == -1) &
        (lof_scores == -1)) |
        (iqr_outliers &
        z_score_outliers)
    )

    # Compute outlier ranks based on Isolation Forest scores for outliers
    outlier_scores = iso_scores[combined_outliers]
    outlier_ranks = pd.Series(outlier_scores).rank(ascending=True).astype(int)

    # Add results to DataFrame
    df['scaled_features'] = list(scaled_features)
    df['iso_score'] = iso_scores
    df['lof_score'] = lof_scores  # Negative outlier factor (lower = more anomalous)
    df['iqr_outlier'] = iqr_outliers
    df['z_score_outlier'] = z_score_outliers
    df['combined_outlier'] = combined_outliers
    df['outlier_rank'] = None  # Initialize column with None
    df.loc[combined_outliers, 'outlier_rank'] = outlier_ranks.values

    # Store feature importance in the DataFrame
    df['feature_importance'] = [feature_importance_df] * len(df)

    # -----------------------------
    # Similarity-weighted re-ranking (NEW) # CHANGE HERE !!
    # -----------------------------
    if known_light_curves:
        df['basename'] = df['file_path'].apply(lambda p: os.path.basename(p))
        df['is_known'] = df['basename'].isin(known_light_curves)

        if df['is_known'].any():
            known_features = np.vstack(df[df['is_known']]['feature_values'].values)
            similarities = cosine_similarity(scaled_features, known_features)
            max_sim_to_known = similarities.max(axis=1)
            df['max_similarity_to_known'] = max_sim_to_known

            # Normalize
            iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
            cos_norm = (max_sim_to_known - max_sim_to_known.min()) / (max_sim_to_known.max() - max_sim_to_known.min())

            # flip bc how it works
            cos_norm = 1 - cos_norm

            # Blend scores
            alpha = 0.3
            df['custom_outlier_score'] = (1 - alpha) * iso_norm + alpha * cos_norm
            df['custom_outlier_rank'] = df['custom_outlier_score'].rank(ascending=True).astype(int)
        else:
            print("Warning: None of the known light curves were found in this dataset.")
            df['custom_outlier_score'] = df['iso_score']
            df['custom_outlier_rank'] = df['iso_score'].rank(ascending=True).astype(int)
    else:
        print("No known_light_curves provided — skipping similarity reweighting.")
        df['custom_outlier_score'] = df['iso_score']
        df['custom_outlier_rank'] = df['iso_score'].rank(ascending=True).astype(int)

    return df


def df_visualize_clusters(scaled_features, combined_outliers, output_file=None):
    """
    Visualize clusters in 2D using PCA, highlighting normal points and outliers.

    Args:
    - scaled_features (np.ndarray): Scaled features of the dataset.
    - combined_outliers (np.ndarray): Boolean array indicating outliers.
    - output_file (str): Path to save the plot (if None, a default name will be used)
    """
    print("\nVisualizing clusters using PCA...")
    start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_clusters.png")

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        features_2d[~combined_outliers, 0], features_2d[~combined_outliers, 1],
        c='blue', label='Normal', alpha=0.4
    )
    plt.scatter(
        features_2d[combined_outliers, 0], features_2d[combined_outliers, 1],
        c='red', label='Outliers', alpha=0.6
    )
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.title('Light Curves in 2D Feature Space')

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Also display the plot
    plt.show()
    print(f"Visualization completed in {time.time() - start_time:.2f} seconds")

def df_analyze_light_curves(light_curves, contamination=0.05, vis=True, output_file=None, processed_light_curves=None):
    print("\nStarting light curve analysis pipeline...")
    total_start_time = time.time()
    """
    Analyze light curves to extract features, detect outliers, and visualize results.

    Args:
    - light_curves (pd.DataFrame): DataFrame containing light curves.
    - contamination (float): Proportion of data expected to be outliers.
    - vis (bool): Whether to visualize the results.

    Returns:
    - pd.DataFrame: DataFrame with extracted features and outlier analysis results.
    """
    # Detect outliers using the provided function
    features = processed_light_curves
    results = df_detect_outliers_extreme(features, contamination=contamination)

    scaled_features = np.vstack(results['scaled_features'].values)
    combined_outliers = results['combined_outlier'].values

    if vis:
        # Visualize results and save to file
        df_visualize_clusters(scaled_features, combined_outliers, output_file)

    return results

# This method works with a pandas dataframe consisting of more information
def extract_features_from_lc(light_curve, output_file=None, band=1):
    """
    Extract features from a single light curve and output the results to a text file.

    Parameters:
        file_path (str): Path to the FITS file containing the light curve.
        output_file (str, optional): Path to the output text file. If None, a default name will be used.
        band (int, optional): Energy band index to load data for (default: 1 for medium band).

    Returns:
        str: Path to the output text file.
    """

    # Extract features
    features_df = df_extract_statistical_features_error(light_curve)

    # Create a readable output format
    feature_names = features_df['feature_names'].iloc[0]
    feature_values = features_df['feature_values'].iloc[0]

    output_file = f"{output_file}_features.txt"

    # Write to text file
    with open(output_file, 'w') as f:
        f.write("Feature examples for 1 light curve:")
        f.write(f"Band: {band} (0=low, 1=medium, 2=high)\n")
        f.write("-" * 50 + "\n")

        # Write each feature and its value
        for name, value in zip(feature_names, feature_values):
            f.write(f"{name}: {value:.6f}\n")

    print(f"Features successfully written to: {output_file}")
    return output_file

def extract_features_to_file(file_path, output_file=None, band=1):
    """
    Extract features from a single light curve and output the results to a text file.

    Parameters:
        file_path (str): Path to the FITS file containing the light curve.
        output_file (str, optional): Path to the output text file. If None, a default name will be used.
        band (int, optional): Energy band index to load data for (default: 1 for medium band).

    Returns:
        str: Path to the output text file.
    """
    print(f"Extracting features from: {file_path}")

    # Load the light curve
    light_curve = load_light_curve(file_path, band=band)

    if light_curve is None:
        print(f"Error: Could not load light curve from {file_path}")
        return None

    # Extract features
    features_df = df_extract_statistical_features_error(light_curve)

    # Create a readable output format
    feature_names = features_df['feature_names'].iloc[0]
    feature_values = features_df['feature_values'].iloc[0]

    # Create output file name if not provided
    if output_file is None:
        base_name = os.path.basename(file_path).replace('.fits', '')
        output_file = f"{base_name}_features.txt"

    # Write to text file
    with open(output_file, 'w') as f:
        f.write(f"Features for light curve: {os.path.basename(file_path)}\n")
        f.write(f"Band: {band} (0=low, 1=medium, 2=high)\n")
        f.write("-" * 50 + "\n")

        # Write each feature and its value
        for name, value in zip(feature_names, feature_values):
            f.write(f"{name}: {value:.6f}\n")

    print(f"Features successfully written to: {output_file}")
    return output_file

# Define functions for grid plotting
def plot_light_curve(ax, lc, title=None, is_outlier=False):
    """Plot a single light curve on the given axis"""
    ax.errorbar(lc['TIME'], lc['RATE'],
                yerr=[lc['ERRM'], lc['ERRP']],
                fmt='o', markersize=2,
                elinewidth=0.5, capsize=0,
                color='red' if is_outlier else 'blue')

    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('Time', fontsize=7)
    ax.set_ylabel('Rate', fontsize=7)

def create_grid_plots(light_curves, results, output_dir, timestamp, plot_num=200, per_file=25):
    """Create grid plots of outliers and regular light curves"""
    print("\nCreating grid plots of outliers and regular light curves...")
    grid_start_time = time.time()

    # Get indices of outliers and regular light curves
    outlier_indices = np.where(results['combined_outlier'])[0]
    regular_indices = np.where(~results['combined_outlier'])[0]

    # Select 25 outliers and 25 regular light curves
    # If there are fewer than 25 outliers, select all available
    n_outliers = min(plot_num, len(outlier_indices))
    n_regular = min(25, len(regular_indices))

    # CHANGE HERE !!
    # Sort outliers by outlier rank (most extreme first)
    if 'custom_outlier_rank' in results.columns:
        # Lower rank means more extreme outlier
        outlier_indices = outlier_indices[np.argsort(results.loc[outlier_indices, 'custom_outlier_rank'])]
    elif 'outlier_rank' in results.columns:
        # Lower rank means more extreme outlier
        outlier_indices = outlier_indices[np.argsort(results.loc[outlier_indices, 'outlier_rank'])]

    # Randomly select from regular if there are too many
    if len(regular_indices) > n_regular:
        regular_indices = np.random.choice(regular_indices, n_regular, replace=False)

    n_files = math.ceil(n_outliers / per_file)

    for file_idx in range(n_files):
        start = file_idx * per_file
        end   = start + per_file
        chunk = outlier_indices[start:end]
        n_in_chunk = len(chunk)

        # build figure
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes = axes.flatten()

        for i in range(per_file):
            ax = axes[i]
            if i < n_in_chunk:
                idx = chunk[i]
                lc  = light_curves[idx]
                file_name = os.path.basename(results.iloc[idx]['file_path'])
                rank      = results.iloc[idx].get('custom_outlier_rank', 'N/A')
                title     = f"#{start + i + 1} (Rank {rank})\n{file_name[:-18]}"
                plot_light_curve(ax, lc, title, is_outlier=True)
            else:
                ax.axis('off')

        plt.tight_layout()

        outlier_grid_file = os.path.join(
            output_dir,
            f"outlier_grid_{file_idx+1}.png"
        )
        fig.savefig(outlier_grid_file, dpi=300)
        plt.close(fig)
        print(f"Saved {outlier_grid_file}")



    # Create a 5x5 grid for outliers
    # if n_outliers > 0:
    #     fig_outliers, axes_outliers = plt.subplots(5, 5, figsize=(15, 15))
    #     axes_outliers = axes_outliers.flatten()

    #     for i in range(25):
    #         if i < n_outliers:
    #             idx = outlier_indices[i]
    #             lc = light_curves[idx]
    #             file_name = os.path.basename(results.iloc[idx]['file_path'])
    #             rank = results.iloc[idx].get('outlier_rank', 'N/A')
    #             title = f"Outlier #{i+1} (Rank: {rank})\n{file_name[:-18]}"
    #             plot_light_curve(axes_outliers[i], lc, title, is_outlier=True)
    #         else:
    #             axes_outliers[i].axis('off')  # Hide unused subplots

    #     plt.tight_layout()
    #     fig_outliers.savefig(outlier_grid_file, dpi=300)
    #     plt.close(fig_outliers)
    #     # print(f"Outlier grid plot saved to: {outlier_grid_file}")
    # else:
    #     print("No outliers found to plot")

    # Create a 5x5 grid for regular light curves
    if n_regular > 0:
        fig_regular, axes_regular = plt.subplots(5, 5, figsize=(15, 15))
        axes_regular = axes_regular.flatten()

        for i in range(25):
            if i < n_regular:
                idx = regular_indices[i]
                lc = light_curves[idx]
                file_name = os.path.basename(results.iloc[idx]['file_path'])
                title = f"Regular #{i+1}\n{file_name[:-18]}"
                plot_light_curve(axes_regular[i], lc, title)
            else:
                axes_regular[i].axis('off')  # Hide unused subplots

        plt.tight_layout()
        regular_grid_file = os.path.join(output_dir, f"regular_grid.png")
        fig_regular.savefig(regular_grid_file, dpi=300)
        plt.close(fig_regular)
        # print(f"Regular grid plot saved to: {regular_grid_file}")
    else:
        print("No regular light curves found to plot")

    print(f"Grid plots created in {time.time() - grid_start_time:.2f} seconds")
    return outlier_grid_file, regular_grid_file

# =============================================================================
# HDBSCAN Clustering Pipeline
# =============================================================================

def run_umap_clustering(light_curves, n_neighbors=15, min_dist=0.1, n_components=2, output_file=None):
    """
    Run UMAP (Uniform Manifold Approximation and Transformation) for dimensionality reduction
    and clustering on the statistical features of light curves.

    Parameters:
        light_curves (list of pd.DataFrame): List of light curve DataFrames.
        n_neighbors (int): The number of neighbors to consider for each point in UMAP.
        min_dist (float): The minimum distance between points in the embedding.
        n_components (int): The number of dimensions in the embedding.
        output_file (str): Path to save the plot (if None, a default name will be used).
    Returns:
        tuple: (cluster_labels, feature_matrix, umap_embedding)
    """
    # Import UMAP here to ensure it's available when needed
    # Note: You need to install umap-learn package first: pip install umap-learn
    try:
        import umap
    except ImportError:
        print("UMAP (umap-learn) is not installed. Please install it using:")
        print("pip install umap-learn")
        print("or")
        print("conda install -c conda-forge umap-learn")
        return None, None, None

    # For clustering after UMAP reduction
    from sklearn.cluster import DBSCAN

    print("\n" + "="*50)
    print("STARTING UMAP CLUSTERING PIPELINE")
    print("="*50)

    # Track execution time
    clustering_start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        output_file = os.path.join(UMAP_OUTPUT_DIR, f"umap_clusters.png")

def run_hdbscan_clustering(light_curves, min_cluster_size=5, min_samples=None, output_file=None, skip_noise=True, known_light_curves=None, processed_light_curves=None):
    """
    Run HDBSCAN clustering on the statistical features of light curves and visualize
    the clusters using PCA.

    Parameters:
        light_curves (list of pd.DataFrame): List of light curve DataFrames.
        min_cluster_size (int): The minimum size of clusters for HDBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        output_file (str): Path to save the plot (if None, a default name will be used).
        skip_noise (bool): Whether to skip noise points in visualization.
        known_light_curves (list): List of file paths for known light curves to highlight.

    Returns:
        tuple: (cluster_labels, feature_matrix, pca_result)
    """
    import hdbscan  # Import here to ensure it's available when needed

    print("\n" + "="*50)
    print("STARTING HDBSCAN CLUSTERING PIPELINE")
    print("="*50)

    # Track execution time
    clustering_start_time = time.time()

    # Create default output filename if none provided
    if output_file is None:
        if skip_noise:
            output_file = os.path.join(HDBSCAN_OUTPUT_DIR, f"hdbscan_clusters.png")
        else:
            output_file = os.path.join(HDBSCAN_OUTPUT_DIR, f"hdbscan_clusters_with_noise.png")

    # Extract statistical features
    print("Extracting statistical features for clustering...")
    features_df = processed_light_curves
    feature_matrix = np.stack(features_df['feature_values'].values, axis=0)
    feature_names = features_df['feature_names'].values[0]  # All rows have same feature names

    # Generate and save correlation matrix
    print("\nGenerating feature correlation matrix...")
    correlation_matrix_file = plot_correlation_matrix(
        feature_matrix,
        feature_names,
        os.path.join(HDBSCAN_OUTPUT_DIR, f"feature_correlation_matrix.png")
    )
    print(f"Feature correlation matrix saved to: {correlation_matrix_file}")

    # Scale the features
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Run HDBSCAN
    print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")
    hdbscan_start = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass
    )
    cluster_labels = clusterer.fit_predict(scaled_features)
    print(f"HDBSCAN completed in {time.time() - hdbscan_start:.2f} seconds")

    # Get cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")

    # Get cluster sizes
    if n_clusters > 0:
        # print("Cluster sizes:")
        for i in range(n_clusters):
            cluster_size = list(cluster_labels).count(i)
            # print(f"  - Cluster {i}: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.2f}%)")

    # Reduce to 2D using PCA for visualization
    print("Reducing dimensions with PCA for visualization...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)

    # Create a colormap for visualization
    # -1 (noise) will be black, other clusters will have distinct colors
    unique_labels = set(cluster_labels)
    n_clusters_with_noise = len(unique_labels)

    # Create a colorful palette for better cluster distinction
    # Use tab20 colormap which provides 20 distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, max(3, n_clusters)))

    # Plot the clusters
    plt.figure(figsize=(12, 10))

    # Plot each cluster separately for better control over colors
    if skip_noise:
        # Plot non-noise points
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(
                features_2d[mask, 0], features_2d[mask, 1],
                c=[colors[i]], label=f'Cluster {i}',
                alpha=0.8, s=50, edgecolors='white', linewidths=0.5
            )
    else:
        # Include noise points (black)
        noise_mask = cluster_labels == -1
        plt.scatter(
            features_2d[noise_mask, 0], features_2d[noise_mask, 1],
            c='black', label='Noise', alpha=0.2,
            s=50, edgecolors='white', linewidths=0.5
        )

        # Plot non-noise clusters
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(
                features_2d[mask, 0], features_2d[mask, 1],
                c=[colors[i]], label=f'Cluster {i}',
                alpha=0.8, s=50, edgecolors='white', linewidths=0.5
            )

    plt.xlabel('First Principal Component (PC1)')
    plt.ylabel('Second Principal Component (PC2)')
    plt.title('HDBSCAN Clustering of Light Curves in 2D Feature Space', pad=20)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Use log scale for both axes
    plt.xscale('symlog', linthresh=1e-2)
    plt.yscale('symlog', linthresh=1e-2)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Show the plot
    plt.show()

    # After the main clustering visualization
    print("\nGenerating individual cluster PCA plots...")
    plot_cluster_pcas(scaled_features, cluster_labels)
    print("Individual cluster PCA plots completed")

    print(f"HDBSCAN clustering completed in {time.time() - clustering_start_time:.2f} seconds")
    return cluster_labels, feature_matrix, features_2d

def plot_cluster_pcas(scaled_features, cluster_labels, output_dir=None):
    """
    Create separate PCA plots for each cluster identified by HDBSCAN.
    Parameters:
    scaled_features (np.ndarray): Scaled feature matrix
    cluster_labels (np.ndarray): Cluster labels from HDBSCAN
    output_dir (str): Directory to save the plots (if None, uses HDBSCAN_OUTPUT_DIR)
    """
    if output_dir is None:
        output_dir = HDBSCAN_OUTPUT_DIR

    # Create cluster PCA directory if it doesn't exist
    cluster_pca_dir = os.path.join(output_dir, "cluster_pcas")
    os.makedirs(cluster_pca_dir, exist_ok=True)

    # Get number of clusters (excluding noise)
    # n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_clusters = len(set(cluster_labels))

    # Process each cluster separately
    for i in range(n_clusters):
        # Get data for this cluster
        cluster_mask = cluster_labels == i - 1
        cluster_data = scaled_features[cluster_mask]

        # Skip if cluster is too small for PCA
        if len(cluster_data) < 2:
            print(f"Skipping PCA for cluster {i - 1} - too few points ({len(cluster_data)})")
            continue

        # Perform PCA on this cluster's data
        pca = PCA()
        cluster_pca = pca.fit_transform(cluster_data)

        # Create scree plot
        plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # explained_var = pca.explained_variance_ratio_
        # plt.plot(range(1, len(explained_var) + 1), np.cumsum(explained_var), 'bo-')
        # plt.xlabel('Number of Components')
        # plt.ylabel('Cumulative Explained Variance Ratio')
        # plt.title(f'Cluster {i} Screen Plot')

        # Plot first two components
        plt.subplot(122)
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], alpha=0.6)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f'Cluster {i - 1} PCA Plot')

        # # Add variance explained text
        # var_explained = explained_var[:2].sum()
        # plt.text(0.05, 0.95, f'Variance explained: {var_explained:.2%}',
        #         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(cluster_pca_dir, f'cluster_{i - 1}_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created PCA plot for cluster {i - 1} with {len(cluster_data)} points")

# Continue the UMAP function implementation
def continue_umap_clustering(light_curves, features_df, scaled_features, n_neighbors=15, min_dist=0.1, n_components=2, output_file=None, eps=0.5, min_samples=5, skip_noise=True):
    """
    Continue the UMAP clustering process with the extracted features.
    This is a helper function to be called after feature extraction.

    Parameters:
        light_curves (list): List of light curve DataFrames
        features_df (pd.DataFrame): DataFrame with features
        scaled_features (np.ndarray): Scaled feature matrix
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        n_components (int): Number of components for UMAP
        output_file (str): Output file path
        eps (float): Maximum distance between samples for DBSCAN
        min_samples (int): Minimum samples for DBSCAN

    Returns:
        tuple: (cluster_labels, feature_matrix, umap_embedding)
    """
    import umap
    from sklearn.cluster import DBSCAN

    # Track execution time
    umap_start = time.time()

    # Run UMAP for dimensionality reduction
    print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        random_state=42
    )
    umap_embedding = reducer.fit_transform(scaled_features)
    print(f"UMAP completed in {time.time() - umap_start:.2f} seconds")

    # Run DBSCAN on the UMAP embedding for clustering
    print(f"Running DBSCAN on UMAP embedding with eps={eps}, min_samples={min_samples}...")
    dbscan_start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(umap_embedding)
    print(f"DBSCAN completed in {time.time() - dbscan_start:.2f} seconds")

    # Get cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")

    # Get cluster sizes
    if n_clusters > 0:
        # print("Cluster sizes:")
        for i in range(n_clusters):
            cluster_size = list(cluster_labels).count(i)
            # print(f"  - Cluster {i}: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.2f}%)")

    # Create a colormap for visualization
    unique_labels = set(cluster_labels)
    n_clusters_with_noise = len(unique_labels)

    # Create a colormap
    cmap = plt.get_cmap('viridis', max(3, n_clusters))

    # Plot the clusters
    plt.figure(figsize=(12, 10))

    # Plot each cluster with a different color
    for label in unique_labels:
        mask = cluster_labels == label
        if label == -1:  # Noise points in gray with higher transparency
            if not skip_noise:
                plt.scatter(
                    umap_embedding[mask, 0], umap_embedding[mask, 1],
                    c='gray', marker='.', label='Noise', alpha=0.2, s=160
                )
        else:  # Regular clusters with colors from colormap and higher visibility
            plt.scatter(
                umap_embedding[mask, 0], umap_embedding[mask, 1],
                c=np.array([cmap(label)]), label=f'Cluster {label}', alpha=0.9, s=60,
                edgecolors='w', linewidths=0.5
            )

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP+DBSCAN Clustering of Light Curves')
    plt.xscale('symlog', linthresh=1e-2)
    plt.yscale('symlog', linthresh=1e-2)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Show the plot
    plt.show()

    return cluster_labels, scaled_features, umap_embedding

def run_umap_clustering_complete(light_curves, n_neighbors=15, min_dist=0.1, n_components=2, output_file=None, eps=0.5, min_samples=5, skip_noise=True, processed_light_curves=None):
    """
    Complete UMAP clustering pipeline, from feature extraction to visualization.

    Parameters:
        light_curves (list): List of light curve DataFrames
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        n_components (int): Number of components for UMAP
        output_file (str): Output file path
        eps (float): Maximum distance between samples for DBSCAN
        min_samples (int): Minimum samples for DBSCAN

    Returns:
        tuple: (cluster_labels, feature_matrix, umap_embedding)
    """
    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(UMAP_OUTPUT_DIR, f"umap_clusters.png")

    clustering_start_time = time.time()

    # Extract statistical features
    print("Extracting statistical features for clustering...")
    features_df = processed_light_curves

    # Prepare feature matrix
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # Scale features
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Run the UMAP clustering
    cluster_labels, _, umap_embedding = continue_umap_clustering(
        light_curves, features_df, scaled_features,
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        output_file=output_file, eps=eps, min_samples=min_samples, skip_noise=skip_noise
    )

    print(f"UMAP clustering completed in {time.time() - clustering_start_time:.2f} seconds")
    return cluster_labels, feature_matrix, umap_embedding

def plot_cluster_samples(light_curves, features_df, cluster_labels, output_dir, timestamp):
    """
    Create grid plots for each cluster detected by HDBSCAN.
    For each cluster, randomly sample up to 10 light curves and display them in a 2x5 grid.

    Parameters:
        light_curves (list): List of light curve DataFrames
        features_df (pd.DataFrame): DataFrame with features
        cluster_labels (np.array): Cluster labels from HDBSCAN
        output_dir (str): Directory to save the plots
        timestamp (str): Timestamp for the filenames
    """
    print("\nCreating cluster sample plots...")
    cluster_start_time = time.time()

    # Create a subdirectory for the cluster plots
    cluster_plots_dir = os.path.join(output_dir, f"cluster_samples_{timestamp}")
    os.makedirs(cluster_plots_dir, exist_ok=True)
    print(f"Created directory for cluster samples: {cluster_plots_dir}")

    # Get unique cluster labels (excluding noise which is -1)
    unique_clusters = sorted(list(set(cluster_labels)))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    # Also create a plot for noise points
    all_cluster_labels = sorted(list(set(cluster_labels)))

    # Define a colormap for consistency with the main plot
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap('viridis', max(3, n_clusters))

    # Process each cluster (including noise)
    for cluster_label in all_cluster_labels:
        # Get indices of light curves in this cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]

        # Skip if no light curves in this cluster (shouldn't happen)
        if len(cluster_indices) == 0:
            continue

        # Determine cluster name and color
        if cluster_label == -1:
            cluster_name = "Noise"
            color = 'gray'
        else:
            cluster_name = f"Cluster_{cluster_label}"
            color = cmap(cluster_label)

        # Sample up to 10 light curves from this cluster
        sample_size = min(10, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)

        # Create a 2x5 grid for the sampled light curves
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        # Plot each sampled light curve
        for i, ax_idx in enumerate(range(10)):
            ax = axes[ax_idx]

            if i < sample_size:
                # Get the light curve and its file path
                lc_idx = sampled_indices[i]
                lc = light_curves[lc_idx]
                file_path = features_df.iloc[lc_idx]['file_path']
                file_name = os.path.basename(file_path)

                # Plot the light curve
                ax.errorbar(lc['TIME'], lc['RATE'],
                           yerr=[lc['ERRM'], lc['ERRP']],
                           fmt='o', markersize=2,
                           elinewidth=0.5, capsize=0,
                           color=color)

                # Set title and labels
                ax.set_title(f"{file_name[-18:-9]}", fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=6)

                if ax_idx >= 5:  # Only add x-label to bottom row
                    ax.set_xlabel('Time', fontsize=7)
                if ax_idx % 5 == 0:  # Only add y-label to leftmost column
                    ax.set_ylabel('Rate', fontsize=7)
            else:
                # Hide unused subplots
                ax.axis('off')

        # Add a main title for the entire figure
        if cluster_label == -1:
            plt.suptitle(f"Noise Points (Sample of {sample_size} out of {len(cluster_indices)} points)",
                       fontsize=14, y=0.98)
        else:
            plt.suptitle(f"Cluster {cluster_label} (Sample of {sample_size} out of {len(cluster_indices)} points)",
                       fontsize=14, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

        # Save the plot
        cluster_file = os.path.join(cluster_plots_dir, f"{cluster_name}_samples.png")
        fig.savefig(cluster_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Created sample plot for {cluster_name} with {sample_size} light curves")

    print(f"Cluster sample plots created in {time.time() - cluster_start_time:.2f} seconds")
    return cluster_plots_dir

def plot_feature_corner_plot(feature_matrix, feature_names, cluster_labels, output_file=None):
    """
    Create a corner plot of features color-coded by HDBSCAN clusters using seaborn's pairplot.

    Parameters:
        feature_matrix (np.ndarray): Matrix of features where each row is a light curve and each column is a feature
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        cluster_labels (np.ndarray): Cluster labels from HDBSCAN clustering
        output_file (str): Path to save the plot (if None, a default name will be used)

    Returns:
        str: Path to the saved plot
    """

    # Create default output filename if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"corner_plot")

    # Get the output directory from the output file path
    output_dir = os.path.dirname(output_file)

    # Create a DataFrame with the features and cluster labels
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_features['cluster'] = cluster_labels

    # Convert cluster labels to string for better visualization
    df_features['cluster'] = df_features['cluster'].apply(lambda x: f'Cluster {x}' if x >= 0 else 'Noise')

    # Filter out noise points
    df_features_no_noise = df_features[df_features['cluster'] != 'Noise']

    # If after filtering there's no data, use the original dataframe
    if len(df_features_no_noise) == 0:
        print("Warning: All points were classified as noise. Showing all data points.")
        df_features_filtered = df_features
    else:
        df_features_filtered = df_features_no_noise
        print(f"Removed {len(df_features) - len(df_features_filtered)} noise points from the pairplot.")

    # Create a custom palette for the clusters (excluding noise)
    unique_clusters = df_features_filtered['cluster'].unique()
    n_clusters = len(unique_clusters)

    # Create a colormap for the clusters
    cluster_colors = plt.get_cmap('viridis', max(3, n_clusters))
    palette = {}

    # Assign colors to clusters
    for i, cluster in enumerate(unique_clusters):
        palette[cluster] = cluster_colors(i)

    # Create the corner plot
    print("Creating corner plot of features colored by clusters (excluding noise)...")

    output1 = output_file + "_reg.png"

    # Create the pairplot with cluster coloring
    corner_plot1 = sns.pairplot(
        df_features_filtered,
        hue='cluster',
        palette=palette,
        plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'none'},
        diag_kind='kde',
        corner=True,  # False - full pairplot, True - corner plot
    )

    # Set log scale for all axes
    for ax in corner_plot1.axes.flatten():
        if ax is not None:
            ax.set_xscale('symlog', linthresh=1e-2)
            ax.set_yscale('symlog', linthresh=1e-2)

    # Add a main title for the entire figure
    corner_plot1.fig.suptitle('Feature Relationships by Cluster (Noise Excluded)', fontsize=24, y=1.02)

    # Save the plot
    corner_plot1.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"Corner plot 1 saved to: {output1}")

    output2 = output_file + "_KD.png"

    # Create the KD pairplot with cluster coloring
    corner_plot2 = sns.pairplot(
        df_features_filtered,
        hue='cluster',
        palette=palette,
        plot_kws={'alpha': 0.7},
        kind='kde', # reduce the number of contours - levels/significatnce etc
        diag_kind='kde',
        corner=True,  # False - full pairplot, True - corner plot
    )

    # Set log scale for all axes
    for ax in corner_plot2.axes.flatten():
        if ax is not None:
            ax.set_xscale('symlog', linthresh=1e-2)
            ax.set_yscale('symlog', linthresh=1e-2)

    # Add a main title for the entire figure
    corner_plot2.fig.suptitle('Feature Relationships by Cluster (Noise Excluded)', fontsize=24, y=1.02)

    # Save the plot
    corner_plot2.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"Corner plot 1 saved to: {output2}")

    return output_file

def analyze_cluster_feature_importance(feature_matrix, feature_names, cluster_labels, output_dir=None):
    """
    Analyze which features are most important for distinguishing each cluster.

    Parameters:
        feature_matrix (np.ndarray): Matrix of features where each row is a light curve and each column is a feature
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        cluster_labels (np.ndarray): Cluster labels from clustering algorithm
        output_dir (str): Directory to save the output plots and text file

    Returns:
        dict: Dictionary mapping cluster names to DataFrames containing feature importance scores
    """
    # Set default output directory if none provided
    if output_dir is None:
        output_dir = HDBSCAN_OUTPUT_DIR

    # Create a DataFrame with features and cluster labels
    df = pd.DataFrame(feature_matrix, columns=feature_names)

    # Convert cluster labels to string format
    cluster_str_labels = np.array([f'Cluster {x}' if x >= 0 else 'Noise' for x in cluster_labels])
    df['cluster'] = cluster_str_labels

    # Get unique clusters (excluding noise)
    unique_clusters = [c for c in df['cluster'].unique() if c != 'Noise']

    if len(unique_clusters) <= 1:
        print("Not enough clusters to perform feature importance analysis.")
        return None

    # Dictionary to store feature importance for each cluster
    cluster_importance = {}

    # For each cluster, calculate how its feature distributions differ from other clusters
    for cluster in unique_clusters:
        # Create binary classification: this cluster vs all other clusters (excluding noise)
        cluster_mask = df['cluster'] == cluster
        other_mask = (df['cluster'] != cluster) & (df['cluster'] != 'Noise')

        # Skip if either group is too small
        if cluster_mask.sum() < 5 or other_mask.sum() < 5:
            continue

        # Calculate feature importance based on distribution differences
        importance_scores = []

        for feature in feature_names:
            # Get feature values for this cluster and other clusters
            cluster_values = df.loc[cluster_mask, feature].values
            other_values = df.loc[other_mask, feature].values

            # Calculate mean and std for both groups
            cluster_mean = np.mean(cluster_values)
            other_mean = np.mean(other_values)

            # Calculate pooled standard deviation
            n1 = len(cluster_values)
            n2 = len(other_values)
            s1 = np.std(cluster_values, ddof=1)
            s2 = np.std(other_values, ddof=1)

            # Avoid division by zero
            if s1 == 0 and s2 == 0:
                effect_size = 0
            else:
                # Calculate Cohen's d effect size - defaults to  1e-8 if std deviation is too close to 0
                pooled_std = np.sqrt(((n1-1)*(s1**2) + (n2-1)*(s2**2)) / (n1+n2-2))
                effect_size = abs(cluster_mean - other_mean) / (pooled_std if pooled_std > 0 else  1e-8)

            importance_scores.append({
                'feature': feature,
                'effect_size': effect_size,
                'cluster_mean': cluster_mean,
                'other_mean': other_mean,
                'percent_difference': abs(cluster_mean - other_mean) / (abs(other_mean) if other_mean != 0 else  1e-8) * 100
            })

        # Convert to DataFrame and sort by importance
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('effect_size', ascending=False)

        # Calculate relative importance as percentage
        total_effect = importance_df['effect_size'].sum()
        if total_effect > 0:
            importance_df['importance_percent'] = (importance_df['effect_size'] / total_effect * 100).round(2)
        else:
            importance_df['importance_percent'] = 0

        cluster_importance[cluster] = importance_df

    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to text file
    text_output_file = os.path.join(output_dir, f"cluster_feature_importance.txt")
    with open(text_output_file, 'w') as f:
        f.write("Summary of feature importance for each cluster:\n")
        f.write("="*50 + "\n\n")

        if cluster_importance:
            for cluster, importance_df in cluster_importance.items():
                f.write(f"{cluster} is distinguished by:\n")
                for _, row in importance_df.head(5).iterrows():
                    f.write(f"  - {row['feature']}: {row['importance_percent']:.1f}% importance\n")
                f.write("\n")
        else:
            f.write("No significant cluster feature importance found.\n")

    print(f"Cluster feature importance saved to text file: {text_output_file}")

    # Create a summary plot showing top features for each cluster
    if cluster_importance:
        plt.figure(figsize=(12, 8))

        # Number of top features to show
        n_top_features = min(5, len(feature_names))

        # For each cluster, plot the top features
        for i, (cluster, importance_df) in enumerate(cluster_importance.items()):
            plt.subplot(len(cluster_importance), 1, i+1)

            # Get top features
            top_features = importance_df.head(n_top_features)

            # Create horizontal bar chart
            bars = plt.barh(top_features['feature'], top_features['importance_percent'],
                           color=plt.get_cmap('viridis')(i/max(1, len(cluster_importance)-1)))

            # Add percentage labels
            for bar, value in zip(bars, top_features['importance_percent']):
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                         f"{value:.1f}%", va='center')

            plt.title(f"{cluster} - Top Distinguishing Features")
            plt.xlabel("Relative Importance (%)")
            plt.xlim(0, 100)
            plt.tight_layout()

        # Save the plot
        plot_output_file = os.path.join(output_dir, f"cluster_feature_importance.png")
        plt.tight_layout()
        plt.savefig(plot_output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Cluster feature importance visualization saved to: {plot_output_file}")

    return cluster_importance

def plot_correlation_matrix(feature_matrix, feature_names, output_file=None):
    """
    Create and visualize a correlation matrix for the extracted features.

    Parameters:
        feature_matrix (np.ndarray): Matrix of feature values (rows=samples, columns=features)
        feature_names (list): List of feature names corresponding to the columns in feature_matrix
        output_file (str): Path to save the plot (if None, a default name will be used)

    Returns:
        str: Path to the saved correlation matrix plot
    """
    # Create a DataFrame with the features
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)

    # Calculate the correlation matrix
    corr_matrix = df_features.corr()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle

    # Generate the heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,  # Show correlation values
        mask=mask,   # Only show the lower triangle
        cmap='coolwarm',  # Color map
        vmin=-1, vmax=1,  # Correlation range
        square=True,      # Make cells square
        linewidths=0.5,   # Width of cell borders
        fmt='.2f'         # Format for correlation values
    )

    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()

    # Save the plot
    if output_file is None:
        output_file = os.path.join(FEATURE_OUTPUT_DIR, f"correlation_matrix.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Correlation matrix saved to: {output_file}")

    return output_file


# print(light_curves[0])
# print(light_curves[0].iloc[0])
# # print(light_curves[0].attrs['file_path'])
# print(light_curves[0].iloc[0].attrs)
# print(light_curves[0].iloc[0].attrs['FILE_NAME'])

# Find cluster assignments for specified light curves
# sampled_paths = [os.path.basename(file_path) for file_path in [light_curves[0].iloc[i].attrs['FILE_NAME'] for i in range(len(light_curves[0])//2)]]

# =============================================================================
# Cluster Assignments Pipeline
# =============================================================================

def get_cluster_assignments(file_paths, features_df, cluster_labels):
    """
    Look up cluster assignments for specific light curves.

    Parameters:
        file_paths (list): List of file paths to look up
        features_df (pd.DataFrame): DataFrame containing the features and file paths
        cluster_labels (np.ndarray): Array of cluster labels from HDBSCAN

    Returns:
        dict: Dictionary mapping file paths to their cluster assignments
    """
    # Create a mapping of file paths to cluster labels
    file_to_cluster = {}

    # Get the file paths from the features DataFrame
    all_file_paths = features_df['file_path'].values
    # print(f"len all_file_paths: {len(all_file_paths)}")
    # for i in all_file_paths:
    #     print(i)
    print(f'sampled paths: {file_paths}')
    print(f'len sampled paths: {len(file_paths)}')
    print(f'len all_file_paths: {len(all_file_paths)}')

    for path in file_paths:
        # Find the index of this file path in the features DataFrame
        try:
            idx = np.where(all_file_paths == path)[0][0]
            file_to_cluster[path] = cluster_labels[idx]
        except IndexError:
            file_to_cluster[path] = None  # File not found in dataset

    return file_to_cluster

def plot_significant_curves_with_cluster(significant_curves, light_curves, features_df, cluster_labels, output_dir):
    """
    For each significant light curve, create a plot showing it alongside other members of its cluster.

    Parameters:
        significant_curves (list): List of file paths for significant light curves
        light_curves (list): List of all light curve DataFrames
        features_df (pd.DataFrame): DataFrame with features and file paths
        cluster_labels (np.ndarray): Cluster labels from HDBSCAN
        output_dir (str): Directory to save the plots
    """
    # First, get cluster assignments for significant curves
    sig_curve_clusters = get_cluster_assignments(significant_curves, features_df, cluster_labels)

    # Create output directory for these specific plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sig_plots_dir = os.path.join(output_dir, f"significant_curves")
    os.makedirs(sig_plots_dir, exist_ok=True)

    # Process each significant curve
    for sig_curve in significant_curves:
        cluster_num = sig_curve_clusters[sig_curve]
        if cluster_num is None:
            print(f"Warning: {sig_curve} not found in dataset")
            continue

        # Get all curves in this cluster
        cluster_mask = cluster_labels == cluster_num
        cluster_files = features_df.loc[cluster_mask, 'file_path'].values

        # Remove the significant curve from the pool
        other_curves = [f for f in cluster_files if f != sig_curve]

        # Randomly sample additional curves (up to 9 more for a total of 10)
        sample_size = min(9, len(other_curves))
        if sample_size > 0:
            sampled_curves = np.random.choice(other_curves, size=sample_size, replace=False)
            plot_curves = [sig_curve] + list(sampled_curves)
        else:
            plot_curves = [sig_curve]

        # Create the plot
        fig = plt.figure(figsize=(15, 10))

        for idx, curve_file in enumerate(plot_curves, 1):
            # Find the light curve data
            curve_idx = features_df[features_df['file_path'] == curve_file].index[0]
            lc = light_curves[curve_idx]

            ax = plt.subplot(2, 5, idx)

            # Plot the light curve
            time = lc['TIME'].values
            flux = lc['RATE'].values
            err_low = lc['ERRM'].values
            err_high = lc['ERRP'].values

            # Normalize time to start at 0
            time = time - time[0]

            ax.errorbar(time, flux, yerr=[err_low, err_high], fmt='k.', markersize=1, alpha=0.5)

            # Highlight the significant curve
            title = os.path.basename(curve_file)
            if curve_file == sig_curve:
                ax.set_title(title, color='red')
            else:
                ax.set_title(title)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Rate')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Significant Curve and Cluster {cluster_num} Members', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        plot_file = os.path.join(sig_plots_dir, f"significant_curve_{os.path.basename(sig_curve)}_cluster_{cluster_num}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Created plot for significant curve {sig_curve} (Cluster {cluster_num})")

    return sig_plots_dir

def plot_top_similar_curves(light_curves, processed_light_curves, known_light_curves, output_dir=None, n_similar=25):
    """
    Plot the top N most similar light curves for each known light curve based on cosine similarity.

    Parameters:
        light_curves (list): List of all light curve DataFrames
        known_light_curves (list): List of known light curve file names
        output_dir (str): Directory to save plots (default: current directory)
        n_similar (int): Number of most similar curves to plot (default: 25)
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # Process all light curves to get features
    print("Extracting features from all light curves...")
    processed_lcs = processed_light_curves

    # Extract feature matrix and scale it
    feature_matrix = np.array([lc['feature_values'] for lc in processed_lcs])
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Create mapping of filenames to indices
    file_to_idx = {os.path.basename(lc['file_path']): i for i, lc in enumerate(processed_lcs)}

    # For each known light curve
    for known_lc in known_light_curves:
        if known_lc not in file_to_idx:
            print(f"Warning: {known_lc} not found in dataset")
            continue

        known_idx = file_to_idx[known_lc]
        known_features = scaled_features[known_idx].reshape(1, -1)

        # Calculate cosine similarity with all other curves
        similarities = cosine_similarity(scaled_features, known_features).flatten()

        # Get indices of top N+1 similar curves (including itself)
        top_indices = np.argsort(similarities)[::-1][:n_similar + 1]

        # Create a grid plot
        n_rows = 5
        n_cols = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
        fig.suptitle(f'Top {n_similar} Most Similar Light Curves to {known_lc}', fontsize=16)

        # Plot each similar curve
        for i, idx in enumerate(top_indices[1:]):  # Skip first one (itself)
            row = i // n_cols
            col = i % n_cols
            sim_score = similarities[idx]

            # Plot the light curve
            plot_light_curve(axes[row, col], light_curves[idx])
            axes[row, col].set_title(f'Similarity: {sim_score:.3f}')

        # Remove empty subplots if any
        for i in range(len(top_indices), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_file = os.path.join(output_dir, f'similar_curves_{os.path.splitext(known_lc)[0]}.png')
        plt.savefig(output_file)
        plt.close()

        print(f"Saved plot for {known_lc} to {output_file}")


def plot_light_curve_from_file(filepath, is_outlier=False, title=None, savepath=None):
    lc = load_light_curve(filepath)
    fig, ax = plt.subplots()
    plot_light_curve(ax, lc, title=title, is_outlier=is_outlier)
    plt.tight_layout()
    plt.show()
    plt.savefig(savepath + "/" + os.path.basename(filepath) + ".png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":

    # File Loading Pipeline
    size = 30000
    # Create output directory for plots if it doesn't exist
    FILE_DIR = '/home/pdong/Astro UROP/plots/feature_extraction_plots_test' + f"/{size}_9"
    # Create directories for both HDBSCAN and UMAP outputs
    FEATURE_OUTPUT_DIR = FILE_DIR + "/FEATURES"
    HDBSCAN_OUTPUT_DIR = FILE_DIR + "/HDBSCAN"
    UMAP_OUTPUT_DIR = FILE_DIR + "/UMAP"
    OUTPUT_DIR = HDBSCAN_OUTPUT_DIR  # Default to HDBSCAN for backward compatibility
    os.makedirs(HDBSCAN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(UMAP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

    for i in known_light_curves:
        plot_light_curve_from_file(os.path.join(DEFAULT_DATA_DIR, i), title=i, is_outlier=True, savepath=FEATURE_OUTPUT_DIR)

    print("Starting feature extraction process...")
    print("Loading FITS files...")
    start_time = time.time()
    fits_files = load_all_fits_files()
    print(f"Loaded {len(fits_files)} FITS files in {time.time() - start_time:.2f} seconds")

    # light_curve_file = os.path.join(FILE_DIR, "fits_files.txt")

    # with open(light_curve_file, "w") as f:
    #     f.write(f"Size of fits files loaded: {len(fits_files)}\n")
    #     for file_path in fits_files:
    #         f.write(f"{file_path}\n")


    print("Loading light curves...")
    start_time = time.time()
    light_curves = load_n_light_curves(size, fits_files, band = "med")
    print(f"Loaded {len(light_curves)} light curves in {time.time() - start_time:.2f} seconds")


    # light_curve_file2 = os.path.join(FILE_DIR, "light_curves.txt")

    # with open(light_curve_file2, "w") as f:
    #     f.write(f"Size of light Curves loaded: {len(light_curves)}\n")
    #     for lc in light_curves:
    #         f.write(f"{lc.iloc[0].attrs['FILE_NAME']}\n")

    # print(Light_Curve_Package.__version__)
    # # Half-amplitude of magnitude
    # amplitude = Light_Curve_Package.Amplitude()
    # # Fraction of points beyond standard deviations from mean
    # beyond_std = Light_Curve_Package.BeyondNStd(nstd=1)
    # # Slope, its error and reduced chi^2 of linear fit
    # linear_fit = Light_Curve_Package.LinearFit()
    # # Feature extractor, it will evaluate all features in more efficient way
    # extractor = Light_Curve_Package.Extractor(amplitude, beyond_std, linear_fit)

    # test_lc = light_curves[0]
    # print(test_lc)
    # print(test_lc["TIME"].values)
    # print(test_lc["RATE"].values)
    # print(np.asarray(test_lc["SYM_ERR"].values, dtype=np.float64))
    # # features = extractor(test_lc["TIME"].values, test_lc["RATE"].values, test_lc["SYM_ERR"].values, sorted=True, check=False)
    # # print(features)
    # # print(help(Light_Curve_Package.Amplitude))
    # features = extractor(np.asarray(test_lc["TIME"].values, dtype=np.float64),
    #                      np.asarray(test_lc["RATE"].values, dtype=np.float64),
    #                      np.asarray(test_lc["SYM_ERR"].values, dtype=np.float64),
    #                      sorted=True,
    #                      check=False)
    # print(features)
    # print('\n'.join(f"{name} = {value:.2f}" for name, value in zip(extractor.names, features)))



    print("\n" + "="*50)
    print("STARTING MAIN EXECUTION")
    print("="*50)
    print("Filtering out small light curves")
    light_curves = [lc for lc in light_curves if len(lc) > 3]


    print("Writing example of features:")
    extract_features_from_lc(light_curves[0], output_file=FEATURE_OUTPUT_DIR)

    print("Processing All Features:")
    processed_lcs = df_process_all_light_curves_error(light_curves)

    # raise(NotImplementedError)

    # Track overall execution time
    overall_start_time = time.time()

    # Run the analysis and save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(FEATURE_OUTPUT_DIR, f"main_analysis.png")
    results = df_analyze_light_curves(light_curves, output_file=output_file, processed_light_curves=processed_lcs)

    # Print summary of results
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)


    print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")
    print(f"Processed {len(light_curves)} light curves")
    print(f"Detected {results['combined_outlier'].sum()} outliers ({results['combined_outlier'].sum()/len(results)*100:.2f}%)")
    # print("Top 5 most important features:")
    # if 'feature_importance' in results.columns:
    #     top_features = results['feature_importance'].iloc[0].head(5)
    #     for idx, row in top_features.iterrows():
    #         print(f"  - {row['feature']}: {row['importance']:.4f}")

    #     # Save feature importance to a CSV file
    #     importance_file = os.path.join(FEATURE_OUTPUT_DIR, f"feature_importance_{timestamp}.csv")
    #     results['feature_importance'].iloc[0].to_csv(importance_file, index=False)
    #     print(f"Feature importance saved to: {importance_file}")
    # else:
    #     print("  Feature importance information not available")
    print("="*50)

    # Create grid plots of outliers and regular light curves
    outlier_grid_file, regular_grid_file = create_grid_plots(light_curves, results, FEATURE_OUTPUT_DIR, timestamp)

    # Run the HDBSCAN clustering pipeline
    print("\n" + "="*50)
    print("RUNNING HDBSCAN CLUSTERING")
    print("="*50)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run HDBSCAN with default parameters
    cluster_labels, feature_matrix, pca_result = run_hdbscan_clustering(
        light_curves,
        min_cluster_size=10,  # Adjust based on dataset size
        skip_noise=False,
        known_light_curves=known_light_curves,
        processed_light_curves=processed_lcs
    )

    # Create a corner plot of features colored by clusters
    feature_names = [
        "weighted_mean", "weighted_variance", "median",
        "lag1_autocorr", "hurst_exp",
        "mean_rise_fall_ratio", "mag_ratio", "beyond1std", "stetson_k", "excess_var", "mean_var"
    ]
    corner_plot_file = plot_feature_corner_plot(
        feature_matrix,
        feature_names,
        cluster_labels,
        os.path.join(HDBSCAN_OUTPUT_DIR, f"feature_corner_plot_{timestamp}.png")
    )
    print(f"Feature corner plot saved to: {corner_plot_file}")

    # Create sample plots for each HDBSCAN cluster
    print("\nCreating sample plots for each HDBSCAN cluster...")
    features_df_orig = pd.DataFrame({
        "file_path": [lc.attrs['FILE_NAME'] for lc in light_curves],
    })
    hdbscan_cluster_samples_dir = plot_cluster_samples(
        light_curves,
        features_df_orig,
        cluster_labels,
        HDBSCAN_OUTPUT_DIR,
        timestamp
    )
    print(f"HDBSCAN cluster sample plots saved to: {hdbscan_cluster_samples_dir}")

    # Print summary of cluster feature importance
    print("\nSummary of feature importance for each cluster:")
    cluster_importance = analyze_cluster_feature_importance(feature_matrix, feature_names, cluster_labels, HDBSCAN_OUTPUT_DIR)
    if cluster_importance:
        for cluster, importance_df in cluster_importance.items():
            print(f"\n{cluster} is distinguished by:")
            for _, row in importance_df.head(3).iterrows():
                print(f"  - {row['feature']}: {row['importance_percent']:.1f}% importance")

    # Run the UMAP clustering pipeline
    print("\n" + "="*50)
    print("RUNNING UMAP CLUSTERING")
    print("="*50)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run UMAP with default parameters
    umap_cluster_labels, umap_feature_matrix, umap_embedding = run_umap_clustering_complete(
        light_curves,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        eps=0.5,
        min_samples=5,
        skip_noise=True,
        processed_light_curves=processed_lcs
    )

    # Create a corner plot of features colored by UMAP clusters
    umap_corner_plot_file = plot_feature_corner_plot(
        umap_feature_matrix,
        feature_names,
        umap_cluster_labels,
        os.path.join(UMAP_OUTPUT_DIR, f"umap_feature_corner_plot.png")
    )
    print(f"UMAP feature corner plot saved to: {umap_corner_plot_file}")

    # Create sample plots for each UMAP cluster
    print("\nCreating sample plots for each UMAP cluster...")
    umap_cluster_samples_dir = plot_cluster_samples(
        light_curves,
        features_df_orig,
        umap_cluster_labels,
        UMAP_OUTPUT_DIR,
        timestamp
    )
    print(f"UMAP cluster sample plots saved to: {umap_cluster_samples_dir}")

    # Print summary of UMAP cluster feature importance
    print("\nSummary of feature importance for UMAP clusters:")
    umap_cluster_importance = analyze_cluster_feature_importance(umap_feature_matrix, feature_names, umap_cluster_labels, UMAP_OUTPUT_DIR)
    if umap_cluster_importance:
        for cluster, importance_df in umap_cluster_importance.items():
            print(f"\n{cluster} is distinguished by:")
            for _, row in importance_df.head(3).iterrows():
                print(f"  - {row['feature']}: {row['importance_percent']:.1f}% importance")

    # Run cluster assignments pipeline

    # Write the results
    output_file = os.path.join(FEATURE_OUTPUT_DIR, "cluster_assignments.txt")

    with open(output_file, "w") as f:
        for file_path, cluster in get_cluster_assignments(known_light_curves, features_df_orig, cluster_labels).items():
            if cluster is not None:
                # print(f"Light curve {file_path} belongs to cluster {cluster}")
                f.write(f"Light curve {file_path} belongs to cluster {cluster}\n")
            else:
                # print(f"Light curve {file_path} was not found in the dataset")
                f.write(f"Light curve {file_path} was not found in the dataset\n")

    significant_clusters_dir = plot_significant_curves_with_cluster(
        known_light_curves,
        light_curves,
        features_df_orig,
        cluster_labels,
        FEATURE_OUTPUT_DIR
    )

    # 1) add a column with just the basename of each file_path
    results['basename'] = results['file_path'].apply(lambda p: os.path.basename(p))

    # 2) mark which rows correspond to your known list
    results['is_known'] = results['basename'].isin(known_light_curves)

    # 3) now filter to see which known files are flagged as outliers
    known_outliers = results.loc[
        results['is_known'] & results['combined_outlier'],
        ['basename', 'outlier_rank']
    ]

    with open(output_file, "w") as f:
        # known curves flagged as outliers?
        print("These known curves were flagged as outliers:")
        print(known_outliers)
        f.write("These known curves were flagged as outliers:")
        f.write(str(known_outliers))

    plot_top_similar_curves(
        light_curves=light_curves,  # Your list of light curve DataFrames
        processed_light_curve=processed_lcs
        known_light_curves=known_light_curves,  # Your list of known outlier filenames
        output_dir=FEATURE_OUTPUT_DIR,  # Optional output directory
        n_similar=25  # Optional number of similar curves to show,
    )


    print("Finished feature extraction and analysis.")
