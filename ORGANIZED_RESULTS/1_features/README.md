# 1_features/

## feature_histograms_640/

Feature distribution histograms generated from a 640 light curve sample. 10 PNGs, one per feature.

| File | Feature |
|---|---|
| `weighted_mean_hist.png` | Error-weighted mean flux |
| `weighted_variance_hist.png` | Error-weighted variance |
| `lag1_autocorr_hist.png` | Lag-1 autocorrelation (temporal smoothness) |
| `hurst_exp_hist.png` | Hurst exponent — H>0.5 persistent, H<0.5 mean-reverting |
| `mean_rise_fall_ratio_hist.png` | Rise/fall asymmetry |
| `stetson_k_hist.png` | Kurtosis-based variability index |
| `bexvar_hist.png` | Bayesian excess variance (intrinsic variability beyond noise) |
| `mean_var_hist.png` | Ratio of mean to variance |
| `beyond1std_hist.png` | Fraction of points beyond 1σ |
| `excess_var_hist.png` | Classical normalized excess variance |

Generated with:
```bash
python pipeline/scripts/plot_feature_histograms.py \
  "pipeline/data/640/processedbatch/feature.pkl" \
  --outdir "640features"
```

---

## features.pkl

The full features file (~200k sources) lives on the cluster at:
```
/home/pdong/Astro UROP/pipeline/data/all/amp_max_features/features.pkl
```

**Format**: Pickled pandas DataFrame with columns:
`file_path`, `feature_names`, `feature_values`, `light_curve`

**All 11 features extracted**: `weighted_mean`, `weighted_variance`, `lag1_autocorr`,
`hurst_exp`, `mean_rise_fall_ratio`, `beyond1std`, `stetson_k`, `excess_var`,
`bexvar`, `mean_var`, `ampl_sig`

**9 used for clustering** (see `pipeline/config.py → SELECTED_FEATURES_FOR_CLUSTERING`):
excludes `beyond1std` and `excess_var`
