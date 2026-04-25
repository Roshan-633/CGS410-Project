"""
Step 3: Clustering Analysis & Visualizations
============================================
Reads features.csv + metadata.csv, then produces:

  1.  PCA biplot (PC1 vs PC2, colored by word order)
  2.  t-SNE plot (colored by word order)
  3.  K-Means clustering (k=3) + silhouette analysis
  4.  Agglomerative (Hierarchical) clustering dendrogram
  5.  DBSCAN clustering plot
  6.  Feature correlation heatmap
  7.  Feature importance (loadings) bar chart
  8.  Cluster agreement vs known typology table
  9.  Per-cluster feature means table
  10. Silhouette score vs k plot (elbow / optimal k)

All figures saved to ./results/figures/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             adjusted_rand_score)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR  = "./results"
FIG_DIR  = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Colour palettes ──────────────────────────────────────────────────────────
WO_COLORS = {"SVO": "#2196F3", "SOV": "#F44336", "VSO": "#4CAF50",
             "VOS": "#FF9800", "OVS": "#9C27B0", "OSV": "#795548",
             "UNK": "#9E9E9E"}

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (10, 7)
SELECTED_FEATURES = [
    "nsubj_before_v",
    "obj_before_v",
    "aux_before_v",
    "head_dir_ratio",
    "verb_dir",
    "noun_dir",
    "adp_dir",
    "mean_dep_len",
    "mean_depth",
    "mean_arity",
    "projective"
]
K_VALUES_TO_TRY = [2, 3, 4, 5]


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Load data
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    feat_path = os.path.join(OUT_DIR, "features.csv")
    meta_path = os.path.join(OUT_DIR, "metadata.csv")

    feat_df = pd.read_csv(feat_path, index_col="language")
    meta_df = pd.read_csv(meta_path, index_col="language")

    print("\n[DEBUG] features.csv shape:", feat_df.shape)
    print("[DEBUG] metadata.csv shape:", meta_df.shape)
    print("[DEBUG] Any NaN before coercion?:", feat_df.isna().sum().sum())

    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

    print("[DEBUG] Any NaN after coercion?:", feat_df.isna().sum().sum())

    if feat_df.isna().sum().sum() > 0:
        print("\n[DEBUG] NaN count by column:")
        print(feat_df.isna().sum()[feat_df.isna().sum() > 0])

        print("\n[DEBUG] Rows containing NaN:")
        print(feat_df[feat_df.isna().any(axis=1)])

    # Align indices
    langs = feat_df.index.intersection(meta_df.index)
    feat_df = feat_df.loc[langs]
    meta_df = meta_df.loc[langs]

    #003 
    # Keep only selected useful features
    feat_df = feat_df[SELECTED_FEATURES]
    print(f"[INFO] Using selected features: {list(feat_df.columns)}")

    # Standardise original features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(feat_df.values)
    X_scaled = pd.DataFrame(
        X_scaled_array,
        index=feat_df.index,
        columns=feat_df.columns
    )

    print("[DEBUG] Any NaN in X_scaled?:", np.isnan(X_scaled.values).sum())
    print("[DEBUG] Any +inf/-inf in X_scaled?:", np.isinf(X_scaled.values).sum())
    print("[DEBUG] X_scaled min:", np.nanmin(X_scaled.values))
    print("[DEBUG] X_scaled max:", np.nanmax(X_scaled.values))

    if np.isnan(X_scaled.values).any() or np.isinf(X_scaled.values).any():
        bad_rows, bad_cols = np.where(~np.isfinite(X_scaled.values))
        print("\n[DEBUG] Non-finite entries in X_scaled:")
        for r, c in zip(bad_rows, bad_cols):
            print(f"  row={X_scaled.index[r]}, col={X_scaled.columns[c]}, value={X_scaled.iat[r, c]}")

    word_orders = meta_df["word_order"]
    print(f"Loaded {len(langs)} languages × {feat_df.shape[1]} features")
    return feat_df, X_scaled, word_orders, scaler

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PCA Biplot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pca(X_scaled, word_orders, feat_df):
    print("[DEBUG] plot_pca input NaN count:", np.isnan(X_scaled.values).sum())
    print("[DEBUG] plot_pca input inf count:", np.isinf(X_scaled.values).sum())

    pca   = PCA(n_components=10)
    comps = pca.fit_transform(X_scaled)
    ev    = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: scatter PC1 vs PC2 ──────────────────────────
    ax = axes[0]
    for lang, row in zip(X_scaled.index, comps):
        wo  = word_orders.get(lang, "UNK")
        col = WO_COLORS.get(wo, "#9E9E9E")
        ax.scatter(row[0], row[1], color=col, s=100, zorder=3)
        ax.annotate(lang, (row[0], row[1]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    legend_patches = [mpatches.Patch(color=c, label=wo)
                      for wo, c in WO_COLORS.items()
                      if wo in word_orders.values]
    ax.legend(handles=legend_patches, title="Word Order", loc="best")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
    ax.set_title("PCA — PC1 vs PC2 (coloured by word order)")

    # ── Right: scree plot ─────────────────────────────────
    ax2 = axes[1]
    ax2.bar(range(1, 11), ev * 100, color="#5C6BC0")
    ax2.plot(range(1, 11), np.cumsum(ev) * 100, "o-", color="#E53935")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Scree Plot + Cumulative Variance")
    ax2.axhline(80, linestyle="--", color="#9E9E9E", label="80% threshold")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_pca_biplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

    # Also return PC coords and PCA object for reuse
    return pd.DataFrame(comps, index=X_scaled.index,
                        columns=[f"PC{i+1}" for i in range(10)]), pca


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PCA Loading Plot (feature contributions)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loadings(pca, feat_df):
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=feat_df.columns,
        columns=["PC1", "PC2"]
    )
    # Show top 15 most influential features
    top = (loadings["PC1"].abs() + loadings["PC2"].abs()).nlargest(15).index
    sub = loadings.loc[top]

    fig, ax = plt.subplots(figsize=(10, 6))
    sub[["PC1", "PC2"]].plot(kind="bar", ax=ax, color=["#5C6BC0", "#EF5350"])
    ax.set_title("Top-15 Feature Loadings on PC1 and PC2")
    ax.set_ylabel("Loading coefficient")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_pca_loadings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  t-SNE Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne(X_scaled, word_orders, pca_df):
    # Use top PCA components as input to t-SNE (reduces noise)
    n_langs = len(X_scaled)
    perplexity = min(10, n_langs - 1)

    tsne  = TSNE(n_components=2, perplexity=perplexity,
                 random_state=42, max_iter=2000)
    emb   = tsne.fit_transform(pca_df.iloc[:, :10].values)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, lang in enumerate(X_scaled.index):
        wo  = word_orders.get(lang, "UNK")
        col = WO_COLORS.get(wo, "#9E9E9E")
        ax.scatter(emb[i, 0], emb[i, 1], color=col, s=100, zorder=3)
        ax.annotate(lang, (emb[i, 0], emb[i, 1]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    legend_patches = [mpatches.Patch(color=c, label=wo)
                      for wo, c in WO_COLORS.items()
                      if wo in word_orders.values]
    ax.legend(handles=legend_patches, title="Word Order", loc="best")
    ax.set_title("t-SNE Projection (coloured by word order)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  K-Means Clustering (k=3) + Silhouette
# ═══════════════════════════════════════════════════════════════════════════════

# def plot_kmeans(X_scaled, pca_df, word_orders, k=3):
#     km = KMeans(n_clusters=k, random_state=42, n_init=20)
#     km_labels = km.fit_predict(X_scaled.values)
#     sil = silhouette_score(X_scaled.values, km_labels)
#     print(f"K-Means (k={k}) silhouette score: {sil:.3f}")

#     cluster_colors = ["#E91E63", "#2196F3", "#4CAF50",
#                       "#FF9800", "#9C27B0", "#00BCD4"]

#     fig, axes = plt.subplots(1, 2, figsize=(16, 7))

#     # Left: PCA scatter coloured by K-Means cluster
#     ax = axes[0]
#     for i, lang in enumerate(X_scaled.index):
#         c   = cluster_colors[km_labels[i] % len(cluster_colors)]
#         wo  = word_orders.get(lang, "UNK")
#         ax.scatter(pca_df.iloc[i, 0], pca_df.iloc[i, 1],
#                    color=c, s=100, zorder=3)
#         ax.annotate(f"{lang}\n({wo})", (pca_df.iloc[i, 0], pca_df.iloc[i, 1]),
#                     fontsize=7, ha="center", va="bottom",
#                     xytext=(0, 5), textcoords="offset points")

#     legend_patches = [mpatches.Patch(color=cluster_colors[j],
#                                      label=f"Cluster {j+1}")
#                       for j in range(k)]
#     ax.legend(handles=legend_patches, title="K-Means Cluster")
#     ax.set_xlabel(f"PC1"); ax.set_ylabel(f"PC2")
#     ax.set_title(f"K-Means (k={k}) on PCA space  |  silhouette={sil:.3f}")

#     # Right: silhouette plot
#     ax2 = axes[1]
#     sil_vals = silhouette_samples(X_scaled.values, km_labels)
#     y_lower   = 10
#     for j in range(k):
#         vals = np.sort(sil_vals[km_labels == j])
#         y_upper = y_lower + len(vals)
#         ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
#                           color=cluster_colors[j], alpha=0.7)
#         y_lower = y_upper + 5
#     ax2.axvline(sil, color="red", linestyle="--", label=f"Mean = {sil:.3f}")
#     ax2.set_title("Silhouette Plot (K-Means)")
#     ax2.set_xlabel("Silhouette coefficient")
#     ax2.set_ylabel("Language (sorted within cluster)")
#     ax2.legend()

#     plt.tight_layout()
#     path = os.path.join(FIG_DIR, "04_kmeans.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[Saved] {path}")
#     return pd.Series(km_labels, index=X_scaled.index, name="kmeans_cluster")

#003
def plot_kmeans(X_scaled, pca_df, word_orders, feat_df, k=3):
    k_fig_dir = os.path.join(FIG_DIR, f"k_{k}")
    os.makedirs(k_fig_dir, exist_ok=True)

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km_labels = km.fit_predict(X_scaled.values)
    sil = silhouette_score(X_scaled.values, km_labels)
    print(f"K-Means (k={k}) silhouette score: {sil:.3f}")

    cluster_colors = ["#E91E63", "#2196F3", "#4CAF50",
                      "#FF9800", "#9C27B0", "#00BCD4"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for i, lang in enumerate(X_scaled.index):
        c = cluster_colors[km_labels[i] % len(cluster_colors)]
        wo = word_orders.get(lang, "UNK")
        ax.scatter(pca_df.iloc[i, 0], pca_df.iloc[i, 1],
                   color=c, s=100, zorder=3)
        ax.annotate(f"{lang}\n({wo})",
                    (pca_df.iloc[i, 0], pca_df.iloc[i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    legend_patches = [
        mpatches.Patch(color=cluster_colors[j], label=f"Cluster {j+1}")
        for j in range(k)
    ]
    ax.legend(handles=legend_patches, title="K-Means Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K-Means k={k} on PCA space | silhouette={sil:.3f}")

    ax2 = axes[1]
    sil_vals = silhouette_samples(X_scaled.values, km_labels)
    y_lower = 10

    for j in range(k):
        vals = np.sort(sil_vals[km_labels == j])
        y_upper = y_lower + len(vals)
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                          color=cluster_colors[j % len(cluster_colors)],
                          alpha=0.7)
        y_lower = y_upper + 5

    ax2.axvline(sil, color="red", linestyle="--", label=f"Mean = {sil:.3f}")
    ax2.set_title(f"Silhouette Plot — K-Means k={k}")
    ax2.set_xlabel("Silhouette coefficient")
    ax2.set_ylabel("Language")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(k_fig_dir, f"04_kmeans_k{k}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

    km_series = pd.Series(km_labels, index=X_scaled.index, name="kmeans_cluster")

    # Save cluster assignments for this k
    df_out = pd.DataFrame({
        "language": km_series.index,
        "km_cluster": km_series.values + 1,
        "word_order": [word_orders.get(l, "UNK") for l in km_series.index],
    })
    df_out.to_csv(os.path.join(OUT_DIR, f"cluster_assignments_k{k}.csv"), index=False)

    # Save feature means for this k
    feat_subset = feat_df.copy()
    feat_subset["cluster"] = km_series.values + 1
    means = feat_subset.groupby("cluster").mean().round(3)
    means.to_csv(os.path.join(OUT_DIR, f"cluster_feature_means_k{k}.csv"))

    return km_series, sil

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Silhouette Score vs k  (elbow / optimal k)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_silhouette_k(X_scaled):
    n_langs = len(X_scaled)
    k_range = range(2, min(n_langs, 10))
    scores  = []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=20)
        lbl = km.fit_predict(X_scaled.values)
        scores.append(silhouette_score(X_scaled.values, lbl))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), scores, "o-", color="#5C6BC0", linewidth=2)
    best_k = list(k_range)[np.argmax(scores)]
    ax.axvline(best_k, linestyle="--", color="#E53935",
               label=f"Best k={best_k} (score={max(scores):.3f})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k  (optimal clustering)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_silhouette_vs_k.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    print(f"  Best k = {best_k} (silhouette = {max(scores):.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Hierarchical (Agglomerative) Clustering — Dendrogram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dendrogram(X_scaled, word_orders):
    Z = linkage(X_scaled.values, method="ward")

    # Colour leaf labels by word order
    lang_list = list(X_scaled.index)
    label_colors = [WO_COLORS.get(word_orders.get(l, "UNK"), "#9E9E9E")
                    for l in lang_list]

    fig, ax = plt.subplots(figsize=(14, 7))
    dn = dendrogram(Z, labels=lang_list, ax=ax,
                    color_threshold=0.7 * max(Z[:, 2]),
                    above_threshold_color="#BDBDBD",
                    leaf_rotation=45)

    # Colour x-tick labels by word order
    for lbl_obj, lang in zip(ax.get_xticklabels(), dn["ivl"]):
        wo  = word_orders.get(lang, "UNK")
        lbl_obj.set_color(WO_COLORS.get(wo, "#9E9E9E"))
        lbl_obj.set_fontsize(9)

    legend_patches = [mpatches.Patch(color=c, label=wo)
                      for wo, c in WO_COLORS.items()
                      if wo in word_orders.values]
    ax.legend(handles=legend_patches, title="Word Order", loc="upper right")
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)\n"
                 "Leaf labels coloured by known word order")
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "06_dendrogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DBSCAN Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dbscan(X_scaled, pca_df, word_orders):
    # Tune eps via nearest-neighbour distances
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3).fit(X_scaled.values)
    dists, _ = nbrs.kneighbors(X_scaled.values)
    median_dist = np.median(dists[:, -1])

    db = DBSCAN(eps=median_dist * 1.5, min_samples=2)
    db_labels = db.fit_predict(X_scaled.values)

    unique_labels = set(db_labels)
    palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, lang in enumerate(X_scaled.index):
        lbl = db_labels[i]
        col = "#BDBDBD" if lbl == -1 else palette[lbl % len(palette)]
        ax.scatter(pca_df.iloc[i, 0], pca_df.iloc[i, 1],
                   color=col, s=100, zorder=3,
                   marker="x" if lbl == -1 else "o")
        ax.annotate(lang, (pca_df.iloc[i, 0], pca_df.iloc[i, 1]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    # Legend
    legend_handles = []
    for lbl in sorted(unique_labels):
        col  = "#BDBDBD" if lbl == -1 else palette[lbl % len(palette)]
        name = "Noise" if lbl == -1 else f"Cluster {lbl+1}"
        legend_handles.append(mpatches.Patch(color=col, label=name))
    ax.legend(handles=legend_handles, title="DBSCAN", loc="best")
    ax.set_title(f"DBSCAN Clustering  (eps={median_dist*1.5:.2f}, min_samples=2)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "07_dbscan.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Feature Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation(feat_df):
    # Select non-POS features for clarity
    non_pos = [c for c in feat_df.columns if not c.startswith("pos_")]
    corr = feat_df[non_pos].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Matrix (non-POS features)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "08_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Key Feature Comparison by Word Order (Box plots)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_by_wordorder(feat_df, word_orders):
    key_features = ["mean_dep_len", "head_dir_ratio", "mean_depth",
                    "mean_arity", "adp_dir", "nsubj_before_v",
                    "obj_before_v", "projective"]
    feat_labels = {
        "mean_dep_len":    "Mean Dep. Length",
        "head_dir_ratio":  "Head-Direction Ratio\n(1=all right-headed)",
        "mean_depth":      "Mean Tree Depth",
        "mean_arity":      "Mean Branching Factor",
        "adp_dir":         "ADP Direction Ratio\n(1=preposition-like)",
        "nsubj_before_v":  "Prop. nsubj before VERB\n(S before V indicator)",
        "obj_before_v":    "Prop. obj before VERB\n(O before V indicator)",
        "projective":      "Prop. Projective Sentences",
    }

    combined = feat_df[key_features].copy()
    combined["word_order"] = word_orders

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    palette = {wo: WO_COLORS[wo] for wo in WO_COLORS}

    for i, feat in enumerate(key_features):
        ax = axes[i]
        wo_groups = combined.groupby("word_order")[feat]
        group_data = [v.values for _, v in wo_groups]
        labels_    = [k for k, _ in wo_groups]
        bplot = ax.boxplot(group_data, patch_artist=True, labels=labels_)
        for patch, lbl in zip(bplot["boxes"], labels_):
            patch.set_facecolor(WO_COLORS.get(lbl, "#9E9E9E"))
        # Overlay individual points
        for j, (vals, lbl) in enumerate(zip(group_data, labels_)):
            ax.scatter(np.random.normal(j + 1, 0.05, len(vals)), vals,
                       color=WO_COLORS.get(lbl, "#9E9E9E"),
                       alpha=0.7, zorder=3, s=40)
        ax.set_title(feat_labels[feat], fontsize=9)
        ax.set_ylabel(feat)

    plt.suptitle("Key Dependency Features by Word-Order Family",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "09_features_by_wordorder.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Cluster Summary Table + ARI
# ═══════════════════════════════════════════════════════════════════════════════

def print_cluster_analysis(km_labels, word_orders, feat_df):
    df_out = pd.DataFrame({
        "language":   km_labels.index,
        "km_cluster": km_labels.values + 1,
        "word_order": [word_orders.get(l, "UNK") for l in km_labels.index],
    })
    df_out = df_out.sort_values(["km_cluster", "word_order"])

    print("\n" + "=" * 55)
    print("K-MEANS CLUSTER ASSIGNMENTS")
    print("=" * 55)
    for cluster_id in sorted(df_out["km_cluster"].unique()):
        rows = df_out[df_out["km_cluster"] == cluster_id]
        print(f"\n  Cluster {cluster_id}:")
        for _, row in rows.iterrows():
            print(f"    {row['language']:<20} [{row['word_order']}]")

    #002
    # # Map word_order → numeric for ARI
    # wo_map = {"SVO": 0, "SOV": 1, "VSO": 2, "VOS": 3, "OVS": 4,
    #           "OSV": 5, "UNK": -1}
    # wo_numeric = [wo_map.get(word_orders.get(l, "UNK"), -1)
    #               for l in km_labels.index]
    # valid = [(a, b) for a, b in zip(km_labels.values, wo_numeric) if b != -1]
    # if valid:
    #     pred, true = zip(*valid)
    #     ari = adjusted_rand_score(true, pred)
    #     print(f"\n  Adjusted Rand Index (K-Means vs known word order): {ari:.3f}")
    #     print("  (ARI=1 → perfect match, ARI=0 → random, ARI<0 → worse than random)")

        # Clean ARI computation: use only valid, trusted labels
    

    wo_map = {"SVO": 0, "SOV": 1, "VSO": 2}

    valid_langs = []
    true_labels = []
    pred_labels = []

    for i, lang in enumerate(km_labels.index):
        wo = word_orders.get(lang, "UNK")

        # Ignore anything not in the clean label set
        if wo not in wo_map:
            continue

        valid_langs.append(lang)
        true_labels.append(wo_map[wo])
        pred_labels.append(km_labels.iloc[i])

    if len(set(true_labels)) > 1:
        ari = adjusted_rand_score(true_labels, pred_labels)
        print(f"\n  Adjusted Rand Index (clean labels only): {ari:.3f}")
        print("  (ARI=1 → perfect match, ARI=0 → random, ARI<0 → worse than random)")
    else:
        print("\n  ARI not computed: not enough label variety.")

    # Per-cluster feature means
    feat_subset = feat_df[["mean_dep_len", "head_dir_ratio",
                            "mean_depth", "mean_arity",
                            "adp_dir", "nsubj_before_v"]].copy()
    feat_subset["cluster"] = km_labels.values + 1
    means = feat_subset.groupby("cluster").mean().round(3)
    print("\n  Per-cluster feature means:")
    print(means.to_string())

    # Save tables
    df_out.to_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"), index=False)
    means.to_csv( os.path.join(OUT_DIR, "cluster_feature_means.csv"))
    print(f"\n  [Saved] cluster_assignments.csv, cluster_feature_means.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("CLUSTERING ANALYSIS — Language Dependency Graph Features")
    print("=" * 60)

    feat_df, X_scaled, word_orders, scaler = load_data()

    print("\n[1/9] PCA biplot ...")
    pca_df, pca_obj = plot_pca(X_scaled, word_orders, feat_df)

    print("\n[2/9] PCA loadings ...")
    plot_loadings(pca_obj, feat_df)

    print("\n[3/9] t-SNE ...")
    plot_tsne(X_scaled, word_orders, pca_df)

    # print("\n[4/9] K-Means clustering (k=3) ...")

    # 002
    # km_labels = plot_kmeans(X_scaled, pca_df, word_orders, k=3)

    # Find best k
    # best_k = 3
    # best_score = -1

    # for k in range(2, min(len(X_scaled), 8)):
    #     km = KMeans(n_clusters=k, random_state=42, n_init=20)
    #     labels = km.fit_predict(X_scaled.values)
    #     score = silhouette_score(X_scaled.values, labels)

    #     if score > best_score:
    #         best_score = score
    #         best_k = k

    # print(f"\nUsing optimal k = {best_k}")

    # km_labels = plot_kmeans(X_scaled, pca_df, word_orders, k=best_k)

    #003
    print("\n[4/9] K-Means clustering for multiple k values ...")

    all_k_results = []

    best_k = None
    best_score = -1
    best_labels = None

    for k in K_VALUES_TO_TRY:
        print(f"\nRunning K-Means for k={k} ...")
        km_labels_k, sil_k = plot_kmeans(X_scaled, pca_df, word_orders, feat_df, k=k)

        all_k_results.append({
            "k": k,
            "silhouette_score": sil_k
        })

        if sil_k > best_score:
            best_score = sil_k
            best_k = k
            best_labels = km_labels_k

    km_results_df = pd.DataFrame(all_k_results)
    km_results_df.to_csv(os.path.join(OUT_DIR, "kmeans_k_comparison.csv"), index=False)

    print("\nK-Means comparison:")
    print(km_results_df.to_string(index=False))
    print(f"\nBest k = {best_k} with silhouette = {best_score:.3f}")

    km_labels = best_labels

    print("\n[5/9] Silhouette vs k ...")
    plot_silhouette_k(X_scaled)

    print("\n[6/9] Hierarchical dendrogram ...")
    plot_dendrogram(X_scaled, word_orders)

    print("\n[7/9] DBSCAN ...")
    plot_dbscan(X_scaled, pca_df, word_orders)

    print("\n[8/9] Correlation heatmap ...")
    plot_correlation(feat_df)

    print("\n[9/9] Feature box-plots by word order ...")
    plot_feature_by_wordorder(feat_df, word_orders)

    print_cluster_analysis(km_labels, word_orders, feat_df)

    print(f"\n{'='*60}")
    print(f"ALL DONE — figures saved in: {os.path.abspath(FIG_DIR)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
