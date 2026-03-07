"""
plot_feature_histograms.py

Load a FEATURES_FILE pickle that contains columns:
 - 'feature_names': list of names
 - 'feature_values': np.ndarray of shape (n_features,)

and produce one histogram per feature.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        description="Plot histograms of each feature from a pickled features DataFrame"
    )
    p.add_argument(
        "features_pkl",
        help="Path to the pickled DataFrame (e.g. FEATURES_FILE)"
    )
    p.add_argument(
        "--outdir",
        default="feature_histograms",
        help="Directory to save histogram PNGs"
    )
    args = p.parse_args()

    # 1) Load
    df = pd.read_pickle(args.features_pkl)
    if df.empty:
        raise ValueError("Loaded DataFrame is empty!")

    # 2) Unpack into a matrix
    # assume every row has the same feature_names order
    names = df["feature_names"].iloc[0]
    X = np.vstack(df["feature_values"].values)  # shape (n_samples, n_features)
    features = pd.DataFrame(X, columns=names)

    # 3) Make output dir
    os.makedirs(args.outdir, exist_ok=True)

    # 4) Plot one histogram per feature
    for feat in features.columns:
        plt.figure(figsize=(6,4))
        plt.hist(features[feat].dropna(), bins=50, alpha=0.8)
        plt.title(f"Histogram of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Count")
        plt.tight_layout()
        fname = os.path.join(args.outdir, f"{feat}_hist.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Wrote {fname}")

    print("All histograms written to:", args.outdir)

if __name__ == "__main__":
    main()
