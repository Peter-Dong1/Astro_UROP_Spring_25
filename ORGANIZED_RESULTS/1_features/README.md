# Extracted Features

## features.pkl

**Status**: ⚠️ TO BE DOWNLOADED FROM CLUSTER

- **Source**: Feature extraction from all eRASS1 light curves
- **Cluster Location**: `/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl`
- **Format**: Pickled pandas DataFrame
- **Expected Columns**:
  - `file_path` - Full path to FITS file
  - `file_name` - FITS filename
  - `light_curve` - Nested DataFrame with TIME, RATE, ERRM, ERRP
  - `weighted_mean` - Error-weighted mean flux
  - `weighted_variance` - Error-weighted variance
  - `lag1_autocorr` - Lag-1 autocorrelation (temporal correlation)
  - `hurst_exp` - Hurst exponent (persistence measure)
  - `mean_rise_fall_ratio` - Rise/fall asymmetry
  - `stetson_k` - Stetson K variability index
  - `bexvar` - Bayesian excess variance
  - `mean_var` - Mean variance
  - `ampl_sig` - Amplitude significance
  - (possibly: `excess_var`, `beyond1std`, `weighted_median`, `weighted_iqr`)

## feature_histograms_640/

✅ **Present** - Feature distribution histograms generated from 640 light curve sample.

**Command used**:
```bash
python bexvar_histograms.py \
  "data/640/processedbatch/feature.pkl" \
  --outdir "640features"
```

**Files** (10 histograms):
- `bexvar_hist.png` - Bayesian excess variance distribution
- `beyond1std_hist.png` - Fraction of points beyond 1σ
- `excess_var_hist.png` - Excess variance measure
- `hurst_exp_hist.png` - Hurst exponent (persistence measure)
- `lag1_autocorr_hist.png` - Lag-1 autocorrelation
- `mean_rise_fall_ratio_hist.png` - Rise/fall asymmetry
- `mean_var_hist.png` - Mean variance
- `stetson_k_hist.png` - Stetson K variability index
- `weighted_mean_hist.png` - Error-weighted mean flux
- `weighted_variance_hist.png` - Error-weighted variance

---

## Feature Descriptions

### Weighted Statistics
- **weighted_mean**: Mean flux weighted by inverse variance (more weight to precise measurements)
- **weighted_variance**: Variance accounting for measurement uncertainties

### Temporal Features
- **lag1_autocorr**: Correlation between consecutive time points (measures smoothness vs randomness)
- **hurst_exp**: H > 0.5 = persistent/trending, H < 0.5 = mean-reverting

### Variability Measures
- **stetson_k**: Kurtosis-based variability index (detects outlier-driven variability)
- **bexvar**: Bayesian excess variance (intrinsic variability beyond measurement noise)
- **excess_var**: Classical excess variance
- **beyond1std**: Fraction of points beyond 1 standard deviation

### Shape Features
- **mean_rise_fall_ratio**: Asymmetry of rising vs falling flux segments
- **mean_var**: Ratio of mean to variance
- **ampl_sig**: Amplitude significance relative to errors

---

## Loading Features

```python
import pandas as pd

# Load features
features_df = pd.read_pickle('features.pkl')

print(f"Total sources: {len(features_df)}")
print(f"Features: {[c for c in features_df.columns if c not in ['file_path', 'file_name', 'light_curve']]}")

# Access light curve data
first_lc = features_df.iloc[0]['light_curve']
print(f"Time points: {len(first_lc)}")
```

---

## Next Steps

1. Download `features.pkl` from cluster (see parent README)
2. Generate summary statistics with `generate_summary_stats.py`
3. Verify all expected features are present
