# eROSITA Light Curve Data Processing

## Data Structure Overview

### Input Data (Raw Format)
- **Source**: eROSITA FITS files from `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned`
- **Format**: FITS files containing light curve data across three energy bands
- **Energy Bands**:
  - Low Band (0.2-0.6 keV)
  - Medium Band (0.6-2.3 keV)
  - High Band (2.3-5.0 keV)

### Raw Data Structure
Each FITS file contains:
- `TIME`: Observation timestamps
- `TIMEDEL`: Time delta
- `RATE`: Light curve intensity for each band (0, 1, and 2)
- `RATE_ERRM`: Negative error measurements for each band (0, 1, and 2)
- `RATE_ERRP`: Positive error measurements for each band (0, 1, and 2)

### Processed Data Structure
After processing through `helper.py`, the data is transformed into:

1. **DataFrame Format**:
   - Each light curve is converted into a Pandas DataFrame with columns:
     - `TIME`: Observation time
     - `TIMEDEL`: Time interval
     - `RATE`: Light curve intensity
     - `ERRM`: Negative error
     - `ERRP`: Positive error
     - `SYM_ERR`: Symmetric error approximation ((ERRM + ERRP)/2)

2. **Data Organization**:
   - Light curves are truncated to a maximum of 20 data points
   - Each light curve DataFrame includes metadata:
     - `FILE_NAME`: Original FITS file name
     - `OUTLIER`: Boolean flag for outlier detection

3. **Multi-band Processing**:
   - Data can be loaded for:
     - Single specific band ('low', 'med', or 'high')
     - All bands simultaneously
   - When loading all bands, data is organized in a DataFrame of DataFrames with columns:
     - `file_name`: Source file identifier
     - `low_band`: Low energy band light curve
     - `medium_band`: Medium energy band light curve
     - `high_band`: High energy band light curve

### Data Loading Functions
Key processing functions in `helper.py`:
- `load_light_curve()`: Loads individual FITS files
- `load_n_light_curves()`: Loads specified number of light curves
- `create_dataframe_of_light_curves()`: Creates structured DataFrame of light curves
- `load_all_fits_files()`: Loads all available FITS files from directory

### Data Partitioning
For machine learning purposes, data can be partitioned into:
- Training set
- Validation set (optional)
- Test set

Using the `partition_data()` function with customizable split ratios.
