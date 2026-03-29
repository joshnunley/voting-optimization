#!/usr/bin/env python3
"""
Cluster raw NKalpha profile-matrix feature vectors and compare the discovered
clusters to the hand-identified K0 regimes.

This script uses the output from analyze_raw_profile_matrices.py:
    raw_profile_matrix_analysis.npz

It performs:
1. Feature standardization
2. PCA projection (for visualization)
3. Hierarchical clustering
4. K-means style clustering using scipy.cluster.vq.kmeans2
5. Cluster/regime contingency summaries
6. Per-cluster feature summaries and simple discriminative-feature ranking
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist


def parse_args():
    p = argparse.ArgumentParser(description="Cluster raw profile-matrix features by complexity regime")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/raw_profile_matrix_analysis.npz",
        help="Input feature dataset from analyze_raw_profile_matrices.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/clustering",
        help="Directory for clustering outputs",
    )
    p.add_argument("--n_clusters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return (X - mu) / sigma, mu, sigma


def pca_projection(X, n_components=2):
    Xz, _, _ = standardize(X)
    _, svals, vt = np.linalg.svd(Xz, full_matrices=False)
    coords = Xz @ vt[:n_components].T
    var = np.square(svals)
    var_ratio = var / np.sum(var)
    return coords, var_ratio[:n_components]


def regime_from_k0(k0):
    if 3 <= k0 <= 8:
        return "low"
    if 9 <= k0 <= 14:
        return "mid"
    if 15 <= k0 <= 20:
        return "high"
    return "other"


def contingency(labels, categories, label_names, category_names):
    table = np.zeros((len(label_names), len(category_names)), dtype=int)
    label_idx = {name: i for i, name in enumerate(label_names)}
    cat_idx = {name: i for i, name in enumerate(category_names)}
    for lab, cat in zip(labels, categories):
        table[label_idx[lab], cat_idx[cat]] += 1
    return table


def greedy_cluster_to_regime_map(table):
    """
    Greedy mapping from cluster rows to regime cols for quick interpretability.
    """
    table = table.copy()
    mapping = {}
    used_rows = set()
    used_cols = set()
    while len(used_rows) < table.shape[0] and len(used_cols) < table.shape[1]:
        best = None
        best_val = -1
        for i in range(table.shape[0]):
            if i in used_rows:
                continue
            for j in range(table.shape[1]):
                if j in used_cols:
                    continue
                if table[i, j] > best_val:
                    best_val = table[i, j]
                    best = (i, j)
        if best is None:
            break
        i, j = best
        mapping[i] = j
        used_rows.add(i)
        used_cols.add(j)
    return mapping


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_pca(coords, labels, k0, outpath, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=k0, cmap="viridis", s=40, alpha=0.85)
    axes[0].set_title("Colored by K0")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    cb0 = plt.colorbar(sc0, ax=axes[0])
    cb0.set_label("K0")

    palette = np.array(["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"])
    colors = palette[labels % len(palette)]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=colors, s=40, alpha=0.85)
    axes[1].set_title("Colored by cluster")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(Z, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_xlabel("sample")
    ax.set_ylabel("distance")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_feature_heatmap(cluster_means, feature_names, outpath):
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(feature_names)), 3.8))
    im = ax.imshow(cluster_means, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(cluster_means.shape[0]))
    ax.set_yticklabels([f"cluster {i}" for i in range(cluster_means.shape[0])])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_title("Cluster mean feature z-scores")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    X = d["feature_matrix"].astype(float)
    feature_names = list(d["feature_names"])
    k0 = d["K0_per_sample"].astype(int)
    sample_idx = np.arange(len(k0))
    regimes = np.array([regime_from_k0(x) for x in k0], dtype=object)

    Xz, mu, sigma = standardize(X)
    coords, var_ratio = pca_projection(X)

    # Hierarchical clustering
    Z = linkage(Xz, method="ward")
    h_labels = fcluster(Z, t=args.n_clusters, criterion="maxclust") - 1

    # Kmeans clustering
    np.random.seed(args.seed)
    _, k_labels = kmeans2(Xz, args.n_clusters, minit="points", iter=50)

    regime_names = ["low", "mid", "high"]
    h_cluster_names = [f"cluster_{i}" for i in range(args.n_clusters)]
    k_cluster_names = [f"cluster_{i}" for i in range(args.n_clusters)]

    core_mask = regimes != "other"

    h_table = contingency(
        [f"cluster_{x}" for x in h_labels[core_mask]],
        list(regimes[core_mask]),
        h_cluster_names,
        regime_names,
    )
    k_table = contingency(
        [f"cluster_{x}" for x in k_labels[core_mask]],
        list(regimes[core_mask]),
        k_cluster_names,
        regime_names,
    )

    h_map = greedy_cluster_to_regime_map(h_table)
    k_map = greedy_cluster_to_regime_map(k_table)

    h_acc = sum(h_table[i, j] for i, j in h_map.items()) / np.sum(h_table)
    k_acc = sum(k_table[i, j] for i, j in k_map.items()) / np.sum(k_table)

    # Use kmeans clusters for feature summaries.
    cluster_means = np.zeros((args.n_clusters, Xz.shape[1]))
    cluster_sizes = np.zeros(args.n_clusters, dtype=int)
    for c in range(args.n_clusters):
        mask = (k_labels == c)
        cluster_sizes[c] = int(mask.sum())
        if np.any(mask):
            cluster_means[c] = Xz[mask].mean(axis=0)

    discriminative = np.max(cluster_means, axis=0) - np.min(cluster_means, axis=0)
    top_order = np.argsort(-discriminative)

    sample_rows = []
    for i in range(len(k0)):
        sample_rows.append({
            "sample_idx": int(sample_idx[i]),
            "K0": int(k0[i]),
            "regime": str(regimes[i]),
            "hier_cluster": int(h_labels[i]),
            "kmeans_cluster": int(k_labels[i]),
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]),
        })

    summary_rows = []
    for c in range(args.n_clusters):
        mask = (k_labels == c)
        row = {"cluster": int(c), "size": int(mask.sum())}
        for reg in regime_names:
            row[f"frac_{reg}"] = float(np.mean(regimes[mask] == reg)) if np.any(mask) else np.nan
        summary_rows.append(row)

    top_feature_rows = []
    for rank, idx in enumerate(top_order[:15], start=1):
        row = {
            "rank": rank,
            "feature": feature_names[idx],
            "spread": float(discriminative[idx]),
        }
        for c in range(args.n_clusters):
            row[f"cluster_{c}_zmean"] = float(cluster_means[c, idx])
        top_feature_rows.append(row)

    write_csv(
        outdir / "sample_cluster_assignments.csv",
        sample_rows,
        ["sample_idx", "K0", "regime", "hier_cluster", "kmeans_cluster", "pc1", "pc2"],
    )
    write_csv(
        outdir / "kmeans_cluster_summary.csv",
        summary_rows,
        ["cluster", "size", "frac_low", "frac_mid", "frac_high"],
    )
    write_csv(
        outdir / "top_discriminative_features.csv",
        top_feature_rows,
        ["rank", "feature", "spread"] + [f"cluster_{c}_zmean" for c in range(args.n_clusters)],
    )

    np.savez_compressed(
        outdir / "clustering_results.npz",
        Xz=Xz,
        pca_coords=coords,
        pca_var_ratio=var_ratio,
        hierarchical_labels=h_labels,
        kmeans_labels=k_labels,
        k0=k0,
        regimes=regimes,
        hierarchical_table=h_table,
        kmeans_table=k_table,
        cluster_means=cluster_means,
        cluster_sizes=cluster_sizes,
        discriminative_spread=discriminative,
        feature_names=np.array(feature_names, dtype=object),
    )

    plot_pca(
        coords, h_labels, k0,
        outdir / "pca_hierarchical_clusters.pdf",
        title=f"PCA of raw U features with hierarchical clusters (var={var_ratio[0]:.2f}, {var_ratio[1]:.2f})",
    )
    plot_pca(
        coords, k_labels, k0,
        outdir / "pca_kmeans_clusters.pdf",
        title=f"PCA of raw U features with k-means clusters (var={var_ratio[0]:.2f}, {var_ratio[1]:.2f})",
    )
    plot_dendrogram(Z, outdir / "hierarchical_dendrogram.pdf")
    plot_cluster_feature_heatmap(cluster_means, feature_names, outdir / "kmeans_cluster_feature_heatmap.pdf")

    print("Hierarchical cluster x regime table:")
    print(h_table)
    print(f"Greedy hierarchical alignment accuracy: {h_acc:.3f}")
    print("Kmeans cluster x regime table:")
    print(k_table)
    print(f"Greedy kmeans alignment accuracy: {k_acc:.3f}")
    print("Top discriminative features:")
    for row in top_feature_rows[:10]:
        print(f"  {row['rank']:2d}. {row['feature']} (spread={row['spread']:.3f})")
    print(f"saved {outdir / 'clustering_results.npz'}")


if __name__ == "__main__":
    main()
