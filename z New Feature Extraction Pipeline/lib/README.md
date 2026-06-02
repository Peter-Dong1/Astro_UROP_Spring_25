# lib/

Shared feature extraction code imported by both pipeline paths.

## Files

### `feature_functions.py`

All feature computation logic. Imported by `batch_feature_extraction.py` (Path B step 2) and
`run_feature_extraction.py` (Path A). Keeping this logic here means the two extraction paths
stay identical in output.

**Key exports:**

| Symbol | Purpose |
|---|---|
| `df_extract_statistical_features_error(df)` | Extracts all 11 features from one light curve DataFrame. Returns `(result_df, sig_nev_value)` or `None` on failure. |
| `chunked(iterable, size)` | Yields `(chunk, start_index)` tuples for parallelizing over a list. |
| `compute_ampl_sig(df)` | Amplitude significance: `((R_max - σ_max) - (R_min + σ_min)) / sqrt(σ_max² + σ_min²)`. |
| `lag1_autocorrelation(ts)` | Lag-1 autocorrelation coefficient. |
| `hurst_exponent(ts)` | Hurst exponent via rescaled-range analysis. |
| `rise_fall_ratio_over_time(ts)` | Rise/fall ratio across consecutive steps, capped at 10. |
| `df_process_all_light_curves_error(light_curves)` | Sequential batch processor (Path A only). Also saves `SIG_NEV_mappings.pkl`. |

**Features extracted (in order):**
`weighted_mean`, `weighted_variance`, `lag1_autocorr`, `hurst_exp`, `mean_rise_fall_ratio`,
`beyond1std`, `stetson_k`, `excess_var`, `bexvar`, `mean_var`, `ampl_sig`

**Dependencies:** `numpy`, `scipy`, `pandas`, `astropy`, `light_curve` package, `ultranest`
(via `bexvar_ero.py`), `config.py`, `helper.py`
