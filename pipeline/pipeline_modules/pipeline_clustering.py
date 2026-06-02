import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
import hdbscan
import umap
import os
import time
from datetime import datetime

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (
    HDBSCAN_OUTPUT_DIR, UMAP_OUTPUT_DIR, KNOWN_LIGHT_CURVES,
    DEFAULT_CONTAMINATION, DEFAULT_N_NEIGHBORS, DEFAULT_MIN_DIST,
    DEFAULT_N_COMPONENTS, number
)
from pipeline_modules.pipeline_io import build_index_maps

DEFAULT_MIN_CLUSTER_SIZE = 24 # Smaller TODO: Realisitcally 20+
DEFAULT_EPSILON = 0.11
DEFAULT_EOM = 'leaf'
DEFAULT_MIN_SAMPLES = 12

def detect_outliers(df, contamination=0.05, known_light_curves=KNOWN_LIGHT_CURVES):
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
    print(f"\nStarting outlier detection with contamination={contamination}...")
    start_time = time.time()

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
    scaled_features =  np.nan_to_num(scaled_features, nan=0.0)

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

def _discrete_cmap_for_labels(labels, base_cmap='tab20', noise_label=-1):
    """
    Build a discrete ListedColormap + BoundaryNorm + mapping from raw cluster labels
    to contiguous indices, so that scatter + colorbar stay perfectly in sync.

    Returns:
        label_to_idx: dict mapping raw label -> 0..K-1
        idx_to_label: list, idx -> raw label
        cmap: ListedColormap of length K
        norm: BoundaryNorm for 0..K
        ticks: list of tick positions (at bin centers)
        ticklabels: list of raw label values for display on colorbar
    """
    uniq = sorted(set(int(x) for x in labels))
    # put noise at the end so it's visually separated
    if noise_label in uniq:
        uniq = [u for u in uniq if u != noise_label] + [noise_label]

    K = len(uniq)
    base = plt.get_cmap(base_cmap, max(K, 3))
    colors = [base(i) for i in range(K)]

    # make noise a light gray if present
    if noise_label in uniq:
        colors[-1] = (0.7, 0.7, 0.7, 1.0)

    cmap = mpl.colors.ListedColormap(colors, name=f"{base_cmap}_disc_{K}")
    bounds = list(range(K + 1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    idx_to_label = uniq
    ticks = [i + 0.5 for i in range(K)]
    ticklabels = [str(l) for l in uniq]
    return label_to_idx, idx_to_label, cmap, norm, ticks, ticklabels

def run_hdbscan(
    features,
    min_cluster_size,
    min_samples,
    cluster_selection_method,
    cluster_selection_epsilon
):
    """Wrap a single HDBSCAN fit/predict so we can call it twice."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(features)
    return clusterer, labels

def two_stage_hdbscan(
    normalized_features,
    first_pass_params: dict,
    second_pass_params: dict,
    size_threshold: int = 10000
):
    """
    1) Run HDBSCAN with the "loose" (first_pass_params) settings
    2) For any cluster > size_threshold, rerun HDBSCAN *on just that subset*
       with (tighter) second_pass_params
    3) Remap all of those little subclusters back into a single label array
    """
    # 1st pass
    _, labels = run_hdbscan(normalized_features, **first_pass_params)

    # We'll fill this with the final label for every point:
    final_labels = labels.copy()
    next_label = labels.max() + 1

    # look for any cluster that's "too big"
    for cid in set(labels):
        if cid < 0:
            continue   # skip noise
        idxs = np.where(labels == cid)[0]
        if len(idxs) <= size_threshold:
            continue

        # 2nd pass on the subset
        subset = normalized_features[idxs]
        _, sublabels = run_hdbscan(subset, **second_pass_params)

        # sublabels runs from -1,0,1,...  remap each 0,1,2… to new global labels
        for local in set(sublabels):
            if local < 0:
                # keep as noise
                final_labels[idxs[sublabels == local]] = cid
            else:
                final_labels[idxs[sublabels == local]] = next_label
                next_label += 1

    return final_labels

def run_hdbscan_clustering(
    features_df,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    min_samples=None,
    output_file=None,
    two_stage: bool = True,           # ← new toggle
    size_threshold: int = 10000       # ← optional threshold
):
    """
    Run HDBSCAN (1- or 2-stage) on the features.
    two_stage=True will do the "subclustering" pass on large clusters.
    """
    print("\nStarting HDBSCAN clustering...")
    start_time = time.time()

    # 1) prepare output path
    if output_file is None:
        output_file = os.path.join(HDBSCAN_OUTPUT_DIR, "hdbscan_clusters.png")

    # 2) build feature array
    feature_matrix = np.vstack(features_df['feature_values'].values)

    # 3) scale & normalize
    scaler = RobustScaler()
    scaled = scaler.fit_transform(feature_matrix)
    scaled =  np.nan_to_num(scaled, nan=0.0)
    from sklearn.preprocessing import normalize
    normalized = normalize(scaled, norm='l2')

    # 4) pick params
    first_pass = {
        'min_cluster_size': min_cluster_size,
        'min_samples':       min_samples or min_cluster_size,
        'cluster_selection_method': DEFAULT_EOM,
        'cluster_selection_epsilon': DEFAULT_EPSILON
    }
    second_pass = {
        # tighten up however you like
        'min_cluster_size': max(1, int(min_cluster_size/1.5)),
        'min_samples':      max(1, int((min_samples or min_cluster_size)/1.5)),
        'cluster_selection_method': 'leaf',
        'cluster_selection_epsilon': DEFAULT_EPSILON / 3
    }

    # 5) run HDBSCAN (1- or 2-stage)
    if two_stage:
        cluster_labels = two_stage_hdbscan(
            normalized,
            first_pass_params=first_pass,
            second_pass_params=second_pass,
            size_threshold=size_threshold
        )
    else:
        _, cluster_labels = run_hdbscan(
            normalized,
            **first_pass
        )

    # 6) PCA for plotting
    pca = PCA(n_components=2)
    proj2 = pca.fit_transform(normalized)

    # 7) plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        proj2[:, 0], proj2[:, 1],
        c=cluster_labels, cmap='Spectral',
        alpha=0.8, s=50, edgecolors='white', linewidths=0.5
    )
    plt.colorbar(scatter)
    plt.xscale('symlog', linthresh=1e-2)
    plt.yscale('symlog', linthresh=1e-2)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    title = '2-Stage HDBSCAN' if two_stage else 'HDBSCAN'
    plt.title(f'{title} Clusters')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=15,    # you can tweak
        min_dist=0.1,      # ditto
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(normalized)
    file_paths = features_df['file_path'].values

    # 3) Identify known indices
    known_idxs = [i for i, p in enumerate(file_paths)
                    if any(k == os.path.basename(p) for k in KNOWN_LIGHT_CURVES)]

        # --- Discrete label colormap + mapping ---
    label_to_idx, idx_to_label, cmap, norm, ticks, ticklabels = _discrete_cmap_for_labels(
        cluster_labels, base_cmap='tab20', noise_label=-1
    )
    mapped = np.array([label_to_idx[int(l)] for l in cluster_labels], dtype=int)

    # --- Figure/Axes (OO style) ---
    fig, ax = plt.subplots(figsize=(12, 8))

    reg = [i for i in range(len(embedding)) if i not in known_idxs]
    sc = None
    if reg:
        sc = ax.scatter(
            embedding[reg, 0], embedding[reg, 1],
            c=mapped[reg], cmap=cmap, norm=norm,
            alpha=0.75, s=40, edgecolors='white', linewidths=0.4,
            label='Light curves'
        )

    # Known light curves
    for idx in known_idxs:
        ax.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=[mapped[idx]], cmap=cmap, norm=norm,
            marker='*', s=280, edgecolors='black', linewidths=1.2,
            label=f"Known: {os.path.basename(file_paths[idx])}"
        )
        ax.annotate(
            os.path.basename(file_paths[idx]),
            (embedding[idx, 0], embedding[idx, 1]),
            xytext=(6, 6), textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85)
        )

    # --- Colorbar: prefer the real scatter; otherwise attach a dummy mappable to THIS fig/ax ---
    if sc is None:
        dummy = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        dummy.set_array(np.arange(len(idx_to_label)))  # attach data so colorbar is happy
        cbar = fig.colorbar(dummy, ax=ax)
    else:
        cbar = fig.colorbar(sc, ax=ax)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label('HDBSCAN cluster')

    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title('HDBSCAN clusters (UMAP projection)')
    ax.grid(alpha=0.3)

    # dedupe legend
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    ax.legend(by_label.values(), by_label.keys(), loc='best', bbox_to_anchor=(1,1))

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote UMAP plot → {output_file}")

    print(f"HDBSCAN clustering completed in {time.time() - start_time:.2f} seconds")
    return cluster_labels, feature_matrix, embedding

def analyze_cluster_feature_importance(feature_matrix, feature_names, cluster_labels, output_dir=None):
    """
    Analyze feature importance for each cluster.

    Args:
        feature_matrix (np.ndarray): Feature matrix
        feature_names (list): Feature names
        cluster_labels (np.ndarray): Cluster labels
        output_dir (str): Directory to save output
    """
    if output_dir is None:
        output_dir = HDBSCAN_OUTPUT_DIR

    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_output_file = os.path.join(output_dir, f"cluster_feature_importance_{timestamp}.txt")

    cluster_importance = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        if cluster != -1:  # Skip noise points
            cluster_mask = cluster_labels == cluster
            other_mask = cluster_labels != cluster

            # Calculate feature importance using mean difference
            cluster_means = np.mean(feature_matrix[cluster_mask], axis=0)
            other_means = np.mean(feature_matrix[other_mask], axis=0)
            importance = np.abs(cluster_means - other_means)

            # Normalize importance scores
            importance = importance / np.sum(importance) * 100

            # Store results
            cluster_importance[f"Cluster {cluster}"] = pd.DataFrame({
                'feature': feature_names,
                'importance_percent': importance
            }).sort_values('importance_percent', ascending=False)

    # Write results to file
    with open(text_output_file, 'w') as f:
        f.write("Cluster Feature Importance Analysis\n")
        f.write("=" * 50 + "\n\n")

        if cluster_importance:
            for cluster, importance_df in cluster_importance.items():
                f.write(f"{cluster} is distinguished by:\n")
                for _, row in importance_df.head(5).iterrows():
                    f.write(f"  - {row['feature']}: {row['importance_percent']:.1f}% importance\n")
                f.write("\n")
        else:
            f.write("No significant cluster feature importance found.\n")

    print(f"Cluster feature importance saved to: {text_output_file}")

def save_cluster_size_histogram(file_paths, labels, outdir, tag):
    """
    Given an array of cluster labels,
     1) save 'tag_cluster_sizes.txt' with each label and its count
     2) save 'tag_cluster_sizes.png' bar‐plot of that distribution
    """
    os.makedirs(outdir, exist_ok=True)

    # compute counts
    counts = pd.Series(labels).value_counts().sort_index()

    # figure out which clusters contain known curves
    # map basename -> cluster
    basename_to_cluster = {
        os.path.basename(fp): lbl
        for fp, lbl in zip(file_paths, labels)
    }
    known_clusters = { basename_to_cluster[lc]
                       for lc in KNOWN_LIGHT_CURVES
                       if lc in basename_to_cluster }

    # 1) dump to text
    txtfile = os.path.join(outdir, f"{tag}_cluster_sizes.txt")
    with open(txtfile, 'w') as f:
        f.write("cluster_label\tcount\tknown?\n")
        for lbl, cnt in counts.items():
            mark = "*" if lbl in known_clusters else ""
            f.write(f"{lbl}\t{cnt}\t{mark}\n")
    print(f"→ Wrote cluster counts to {txtfile}")

    # 2) bar‐plot with hatching for known clusters
    plt.figure(figsize=(12,5))
    bars = plt.bar(
        counts.index.astype(str),
        counts.values,
        log=True
    )
    # hatch the bars that correspond to known clusters
    for bar, lbl in zip(bars, counts.index):
        if lbl in known_clusters:
            bar.set_hatch("///")

    plt.xlabel("Cluster label")
    plt.ylabel("Number of light curves")
    plt.title(f"Cluster‐size distribution ({tag})")
    plt.tight_layout()

    pngfile = os.path.join(outdir, f"{tag}_cluster_sizes.png")
    plt.savefig(pngfile, dpi=150)
    plt.close()
    print(f"→ Wrote cluster histogram to {pngfile}")

def plot_subcluster_histograms(orig_labels, new_labels,
                               threshold, outdir):
    """
    For every original cluster in orig_labels with size > threshold,
    build and save a bar‐histogram of how its points got reassigned
    in new_labels.

    Args:
        orig_labels (array‐like of int): 1st‐pass cluster IDs
        new_labels  (array‐like of int): 2nd‐pass cluster IDs (same length)
        threshold   (int): minimum number of points in orig cluster to plot
        outdir      (str): directory to save PNGs
    """
    os.makedirs(outdir, exist_ok=True)
    orig = pd.Series(orig_labels, name="orig")
    new  = pd.Series(new_labels, name="sub")

    # group by original cluster
    for cluster_id, idxs in orig.groupby(orig).groups.items():
        size = len(idxs)
        if size <= threshold:
            continue

        # count how many points went into each subcluster
        subcounts = new.iloc[list(idxs)].value_counts().sort_index()

        # plot
        plt.figure(figsize=(8,4))
        subcounts.plot(kind='bar', log=True)
        plt.title(f"Orig cluster {cluster_id} (n={size}) → {len(subcounts)} subclusters")
        plt.xlabel("Subcluster label")
        plt.ylabel("Count (log scale)")
        plt.tight_layout()
        fname = os.path.join(outdir, f"subcluster_hist_{cluster_id}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Wrote {fname}")
