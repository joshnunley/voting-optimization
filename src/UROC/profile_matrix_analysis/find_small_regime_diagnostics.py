#!/usr/bin/env python3
"""
Search for small feature sets that best separate the low / mid / high
complexity regimes in the raw NKalpha profile-matrix feature dataset.

This is a supervised diagnostic step: the regimes are the hand-identified
bands
    low  = K0 3..8
    mid  = K0 9..14
    high = K0 15..20

For all 2-feature and 3-feature combinations, we measure how well a simple
nearest-centroid classifier separates the regimes under leave-one-out
cross-validation. The goal is not final prediction, but identifying a very
small, interpretable set of raw-matrix diagnostics that could later drive a
self-calibrating diffusion rule.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Find small raw-matrix diagnostics for low/mid/high regime separation")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/raw_profile_matrix_analysis.npz",
        help="Input feature dataset from analyze_raw_profile_matrices.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/small_diagnostics",
        help="Directory for outputs",
    )
    p.add_argument(
        "--max_feature_pool",
        type=int,
        default=15,
        help="Restrict the search to the top-N candidate features from the initial clustering pass",
    )
    return p.parse_args()


def regime_from_k0(k0):
    if 3 <= k0 <= 8:
        return 0  # low
    if 9 <= k0 <= 14:
        return 1  # mid
    if 15 <= k0 <= 20:
        return 2  # high
    return -1


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return (X - mu) / sigma, mu, sigma


def loo_nearest_centroid_accuracy(X, y):
    n = X.shape[0]
    correct = 0
    preds = np.full(n, -1, dtype=int)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        Xtr = X[train_mask]
        ytr = y[train_mask]
        Xte = X[i]

        mu = Xtr.mean(axis=0)
        sigma = Xtr.std(axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        Xtrz = (Xtr - mu) / sigma
        Xtez = (Xte - mu) / sigma

        centroids = np.vstack([Xtrz[ytr == c].mean(axis=0) for c in range(3)])
        dists = np.sum((centroids - Xtez[None, :]) ** 2, axis=1)
        pred = int(np.argmin(dists))
        preds[i] = pred
        if pred == y[i]:
            correct += 1
    return correct / float(n), preds


def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    return cm


def read_top_feature_pool(dataset_path: Path, max_feature_pool: int):
    top_csv = dataset_path.parent / "clustering" / "top_discriminative_features.csv"
    if top_csv.exists():
        rows = list(csv.DictReader(top_csv.open()))
        feats = [r["feature"] for r in rows[:max_feature_pool]]
        if feats:
            return feats
    return None


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_best_2d(X, y, feature_names, combo, outpath, title):
    X2 = X[:, combo]
    labels = np.array(["low", "mid", "high"], dtype=object)[y]
    colors = {"low": "#377eb8", "mid": "#4daf4a", "high": "#e41a1c"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in ["low", "mid", "high"]:
        mask = labels == lab
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=colors[lab], label=lab, s=45, alpha=0.8
        )
    ax.set_xlabel(feature_names[combo[0]])
    ax.set_ylabel(feature_names[combo[1]])
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    X_all = d["feature_matrix"].astype(float)
    feature_names = list(d["feature_names"])
    k0 = d["K0_per_sample"].astype(int)

    y_all = np.array([regime_from_k0(v) for v in k0], dtype=int)
    mask = y_all >= 0
    X = X_all[mask]
    y = y_all[mask]

    pool_names = read_top_feature_pool(Path(args.input_npz), args.max_feature_pool)
    if pool_names is None:
        pool_names = feature_names[:args.max_feature_pool]
    pool_indices = [feature_names.index(name) for name in pool_names]

    pair_rows = []
    best_pair = None
    best_pair_preds = None
    best_pair_acc = -1.0
    for combo in itertools.combinations(pool_indices, 2):
        acc, preds = loo_nearest_centroid_accuracy(X[:, combo], y)
        row = {
            "feature_1": feature_names[combo[0]],
            "feature_2": feature_names[combo[1]],
            "accuracy": float(acc),
        }
        pair_rows.append(row)
        if acc > best_pair_acc:
            best_pair_acc = acc
            best_pair = combo
            best_pair_preds = preds
    pair_rows.sort(key=lambda r: (-r["accuracy"], r["feature_1"], r["feature_2"]))

    triple_rows = []
    best_triple = None
    best_triple_preds = None
    best_triple_acc = -1.0
    for combo in itertools.combinations(pool_indices, 3):
        acc, preds = loo_nearest_centroid_accuracy(X[:, combo], y)
        row = {
            "feature_1": feature_names[combo[0]],
            "feature_2": feature_names[combo[1]],
            "feature_3": feature_names[combo[2]],
            "accuracy": float(acc),
        }
        triple_rows.append(row)
        if acc > best_triple_acc:
            best_triple_acc = acc
            best_triple = combo
            best_triple_preds = preds
    triple_rows.sort(key=lambda r: (-r["accuracy"], r["feature_1"], r["feature_2"], r["feature_3"]))

    write_csv(
        outdir / "top_pairs.csv",
        pair_rows[:50],
        ["feature_1", "feature_2", "accuracy"],
    )
    write_csv(
        outdir / "top_triples.csv",
        triple_rows[:50],
        ["feature_1", "feature_2", "feature_3", "accuracy"],
    )

    pair_cm = confusion_matrix(y, best_pair_preds, n_classes=3)
    triple_cm = confusion_matrix(y, best_triple_preds, n_classes=3)
    np.savez_compressed(
        outdir / "small_diagnostic_search.npz",
        best_pair=np.array(best_pair),
        best_pair_accuracy=np.array(best_pair_acc),
        best_pair_confusion=pair_cm,
        best_triple=np.array(best_triple),
        best_triple_accuracy=np.array(best_triple_acc),
        best_triple_confusion=triple_cm,
        pool_feature_names=np.array(pool_names, dtype=object),
    )

    plot_best_2d(
        X, y, feature_names, best_pair,
        outdir / "best_pair_scatter.pdf",
        title=f"Best 2-feature regime diagnostic (acc={best_pair_acc:.3f})",
    )

    regime_names = ["low", "mid", "high"]
    print("Best 2-feature combination:")
    print("  ", [feature_names[i] for i in best_pair], f"acc={best_pair_acc:.3f}")
    print(pair_cm)
    print("Best 3-feature combination:")
    print("  ", [feature_names[i] for i in best_triple], f"acc={best_triple_acc:.3f}")
    print(triple_cm)
    print("Top 10 pairs:")
    for row in pair_rows[:10]:
        print(f"  {row['feature_1']} | {row['feature_2']} : {row['accuracy']:.3f}")
    print("Top 10 triples:")
    for row in triple_rows[:10]:
        print(f"  {row['feature_1']} | {row['feature_2']} | {row['feature_3']} : {row['accuracy']:.3f}")
    print(f"saved {outdir / 'small_diagnostic_search.npz'}")
    print("Regime order in confusion matrices:", regime_names)


if __name__ == "__main__":
    main()
