import pandas as pd
import numpy as np
import os
import pickle
from astropy.io import fits
from astropy.table import Table
from helper import DEFAULT_DATA_DIR, load_light_curve  # adjust if needed

# === CONFIGURATION ===
INPUT_PICKLE = "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/bexvar_features/features.pkl"
OUTPUT_PICKLE = "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/all/amp_max_features/features.pkl"
FITS_DATA_DIR = DEFAULT_DATA_DIR  # or a hardcoded string path

NEW_FEATURE_NAME = "ampl_sig"

# === Feature function ===
def compute_new_feature_from_fits(fits_path):
    try:
        df = load_light_curve(fits_path, band=1)
        if df is None or len(df) < 3:
            return 0

        rates = df['RATE'].values
        errors = df['SYM_ERR'].values

        idx_min = np.argmin(rates)
        idx_max = np.argmax(rates)

        r_min = rates[idx_min]
        r_max = rates[idx_max]

        sigma_min = errors[idx_min]
        sigma_max = errors[idx_max]

        ampl_max = (r_max - sigma_max) - (r_min + sigma_min)
        denominator = np.sqrt(sigma_max**2 + sigma_min**2)

        if denominator == 0:
            return 0

        ampl_sig = ampl_max / denominator
        return ampl_sig

    except Exception as e:
        print(f"❌ Failed to compute {NEW_FEATURE_NAME} from {fits_path}: {e}")
        return 0

# === Main ===
def main():
    print(f"📥 Loading existing features from: {INPUT_PICKLE}")
    df = pd.read_pickle(INPUT_PICKLE)
    print(f"✅ Loaded DataFrame with {len(df)} rows")

    print(f"\n🔍 Appending new feature: {NEW_FEATURE_NAME}")
    for i, row in df.iterrows():
        if i % 10000 == 0 and i > 0:
            print(f"  → Processed {i}/{len(df)}")

        fits_path = os.path.join(FITS_DATA_DIR, row["file_path"])
        new_val = compute_new_feature_from_fits(fits_path)

        # Append to feature list
        row['feature_names'].append(NEW_FEATURE_NAME)
        row['feature_values'] = np.append(row['feature_values'], new_val)

    print(f"\n💾 Saving updated DataFrame to: {OUTPUT_PICKLE}")
    os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
    df.to_pickle(OUTPUT_PICKLE)
    print("✅ Done.")

if __name__ == "__main__":
    main()
