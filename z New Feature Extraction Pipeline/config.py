import os

LOAD_SIZE = 1000

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") + '/' + str(LOAD_SIZE)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Feature extraction output directory
number = 1
PLOT_DIR = os.path.join(BASE_DIR, "plots")
FILE_PLOT_DIR = os.path.join(PLOT_DIR,(str(LOAD_SIZE)) + str(number))
FEATURE_OUTPUT_DIR = os.path.join(FILE_PLOT_DIR, "features")
HDBSCAN_OUTPUT_DIR = os.path.join(FILE_PLOT_DIR, "hdbscan")
UMAP_OUTPUT_DIR = os.path.join(FILE_PLOT_DIR, "umap")


# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, PLOT_DIR, FILE_PLOT_DIR,
                 FEATURE_OUTPUT_DIR, HDBSCAN_OUTPUT_DIR, UMAP_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Known interesting light curves
KNOWN_LIGHT_CURVES = [
    "em01_211120_020_LightCurve_00007_c010_rebinned.fits",
    "em01_039135_020_LightCurve_00058_c010_rebinned.fits",
    "em01_038099_020_LightCurve_00005_c010_rebinned.fits"
]

# Feature extraction parameters
DEFAULT_BAND = "med"  # Default energy band for light curves

# Analysis pipeline parameters
DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_N_COMPONENTS = 2
DEFAULT_MIN_CLUSTER_SIZE = 5

# File for storing extracted features
FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "light_curve_features.pkl")
