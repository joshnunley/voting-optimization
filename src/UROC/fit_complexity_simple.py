#!/usr/bin/env python3
"""
Fit parabolic iso-regime curves: K = a*α² + b

The complexity parameter is C = K - a*α², which equals b along each curve.
Find `a` such that the regime label (rank-1 method) is maximally constant
along iso-C lines.

Single parameter a controls the curvature; different regimes are separated
by different values of C (= the K-intercept b).
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]

K_FULL = list(range(1, 21))
A_FULL = [round(a * 0.05, 2) for a in range(21)]

REGIME_METHODS = ["total_score", "plurality", "irv", "borda", "star"]
REGIME_NAMES   = ["Total Score", "Plurality", "IRV", "Borda", "STAR"]
REGIME_COLORS  = ['#999999', '#ff7f00', '#4daf4a', '#377eb8', '#f781bf']
N_TERMINAL = 10


def load_data():
    """Load terminal mean fitness. Try npz first, fall back to mmap."""
    npz_path = "results/direct_results.npz"
    if os.path.exists(npz_path):
        print("Loading from npz ...", flush=True)
        return np.load(npz_path)['mean']

    print("Loading from raw files ...", flush=True)
    nK, nA, nM = len(K_FULL), len(A_FULL), len(METHODS)
    mean = np.full((nK, nA, nM), np.nan)
    t0 = time.time()
    for i, K in enumerate(K_FULL):
        for j, a in enumerate(A_FULL):
            for k, m in enumerate(METHODS):
                p = f"results/direct/K{K}_a{a:.2f}/{m}/mean_history.npy"
                if os.path.exists(p):
                    arr = np.load(p, mmap_mode='r')
                    mean[i, j, k] = float(np.mean(arr[:, -N_TERMINAL:]))
        if (i + 1) % 5 == 0:
            print(f"  K={K} ({time.time()-t0:.0f}s)", flush=True)
    return mean


def regime_labels(mean):
    """Rank-1 method among regime methods only, at each (K, alpha)."""
    regime_idx = [METHODS.index(m) for m in REGIME_METHODS]
    nK, nA, _ = mean.shape
    labels = np.full((nK, nA), np.nan)
    for i in range(nK):
        for j in range(nA):
            vals = np.array([mean[i, j, k] for k in regime_idx])
            if not np.all(np.isnan(vals)):
                labels[i, j] = np.nanargmax(vals)
    return labels


def regime_constancy(labels, a):
    """
    For C = K - a*α², measure how well sorted the regime labels are along C.

    Score = fraction of adjacent pairs (sorted by C) that have the same label
    minus a penalty for the number of distinct transitions.
    Higher is better.
    """
    nK, nA = labels.shape
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if not np.isnan(labels[i, j]):
                C = K - a * alpha ** 2
                points.append((C, int(labels[i, j])))

    points.sort(key=lambda x: x[0])
    Cs     = np.array([p[0] for p in points])
    labs   = np.array([p[1] for p in points])

    # Metric 1: fraction of adjacent pairs with same label
    same = np.sum(labs[1:] == labs[:-1])
    adj_frac = same / (len(labs) - 1)

    # Metric 2: number of regime transitions when sweeping C
    transitions = np.sum(labs[1:] != labs[:-1])

    return adj_frac, transitions


def sweep_a(labels, a_range=np.linspace(-100, 100, 2001)):
    """Sweep `a` to find the curvature that maximizes regime constancy."""
    best_a = 0
    best_score = 0
    best_trans = 999
    results = []

    for a in a_range:
        score, trans = regime_constancy(labels, a)
        results.append((a, score, trans))
        if score > best_score or (score == best_score and trans < best_trans):
            best_score = score
            best_trans = trans
            best_a = a

    return best_a, best_score, best_trans, np.array(results)


def plot_sweep(results, best_a, fname="complexity_a_sweep.pdf"):
    """Plot constancy score vs curvature parameter a."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    a_vals = results[:, 0]
    scores = results[:, 1]
    trans  = results[:, 2]

    ax1.plot(a_vals, scores, 'b-', linewidth=1)
    ax1.axvline(best_a, color='red', linestyle='--', label=f'Best a = {best_a:.1f}')
    ax1.set_ylabel("Adjacent agreement fraction", fontsize=11)
    ax1.set_title("Regime constancy vs curvature parameter a", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(a_vals, trans, 'g-', linewidth=1)
    ax2.axvline(best_a, color='red', linestyle='--')
    ax2.set_ylabel("Number of regime transitions", fontsize=11)
    ax2.set_xlabel("a  (K = a·α² + b)", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def plot_iso_C_parabolas(labels, a, fname="complexity_parabolas.pdf"):
    """Plot the winner map with iso-C parabolic contours overlaid."""
    nK, nA = labels.shape

    # Build color image
    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            if not np.isnan(labels[i, j]):
                img[i, j] = to_rgb(REGIME_COLORS[int(labels[i, j])])

    # Compute C grid
    C_grid = np.full((nK, nA), np.nan)
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            C_grid[i, j] = K - a * alpha ** 2

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])

    # Iso-C contours
    C_min, C_max = np.nanmin(C_grid), np.nanmax(C_grid)
    levels = np.linspace(C_min, C_max, 15)[1:-1]
    cs = ax.contour(C_grid, levels=levels, colors='white', linewidths=1.2,
                    extent=[-0.5, nA-0.5, -0.5, nK-0.5], origin='lower')
    ax.clabel(cs, fontsize=6, fmt='%.1f')

    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)], rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title(f"Iso-complexity curves:  C = K − {a:.1f}·α²\n"
                 f"(white contours = constant C)", fontsize=13)

    handles = [Patch(facecolor=c, label=n) for c, n in zip(REGIME_COLORS, REGIME_NAMES)]
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def plot_C_scatter(labels, a, fname="complexity_C_scatter.pdf"):
    """1D scatter: sort all cells by C, color by regime."""
    nK, nA = labels.shape
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if not np.isnan(labels[i, j]):
                C = K - a * alpha ** 2
                points.append((C, int(labels[i, j])))
    points.sort(key=lambda x: x[0])
    Cs   = np.array([p[0] for p in points])
    labs = np.array([p[1] for p in points])

    fig, ax = plt.subplots(figsize=(12, 4))
    for regime_idx, (name, color) in enumerate(zip(REGIME_NAMES, REGIME_COLORS)):
        mask = (labs == regime_idx)
        if mask.sum() > 0:
            jitter = np.random.normal(0, 0.1, mask.sum())
            ax.scatter(Cs[mask], labs[mask] + jitter, c=color, s=12,
                       alpha=0.6, label=name, edgecolors='none')

    # Add vertical lines at estimated regime boundaries
    transitions = np.where(labs[1:] != labs[:-1])[0]
    for t in transitions:
        ax.axvline(0.5 * (Cs[t] + Cs[t+1]), color='gray', alpha=0.15, linewidth=0.5)

    ax.set_xlabel(f"C = K − {a:.1f}·α²", fontsize=12)
    ax.set_ylabel("Regime", fontsize=12)
    ax.set_yticks(range(len(REGIME_NAMES)))
    ax.set_yticklabels(REGIME_NAMES)
    ax.legend(fontsize=9, loc='center right')
    ax.set_title(f"Regime separation along C = K − {a:.1f}·α²", fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print("=== Simple Parabolic Complexity Fit ===\n")

    mean = load_data()
    labels = regime_labels(mean)

    print(f"\nRegime counts:")
    for i, name in enumerate(REGIME_NAMES):
        print(f"  {name}: {np.sum(labels == i):.0f} cells")

    # Sweep a
    print("\nSweeping a in [-100, 100] ...")
    best_a, best_score, best_trans, results = sweep_a(labels)
    print(f"\n  Best a = {best_a:.1f}")
    print(f"  Adjacent agreement = {best_score:.4f}")
    print(f"  Regime transitions = {best_trans}")

    # Fine sweep around best
    print(f"\nFine sweep a in [{best_a-10:.0f}, {best_a+10:.0f}] ...")
    fine_range = np.linspace(best_a - 10, best_a + 10, 2001)
    best_a2, best_score2, best_trans2, results_fine = sweep_a(labels, fine_range)
    print(f"  Refined a = {best_a2:.2f}")
    print(f"  Adjacent agreement = {best_score2:.4f}")
    print(f"  Regime transitions = {best_trans2}")

    # Plots
    print("\nGenerating plots ...")
    plot_sweep(results, best_a)
    plot_iso_C_parabolas(labels, best_a2)
    plot_C_scatter(labels, best_a2)

    # Print regime boundaries
    nK, nA = labels.shape
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if not np.isnan(labels[i, j]):
                C = K - best_a2 * alpha ** 2
                points.append((C, int(labels[i, j])))
    points.sort(key=lambda x: x[0])
    Cs   = np.array([p[0] for p in points])
    labs = np.array([p[1] for p in points])

    print(f"\nEstimated regime boundaries (C = K − {best_a2:.2f}·α²):")
    prev = labs[0]
    for idx in range(1, len(labs)):
        if labs[idx] != prev:
            boundary = 0.5 * (Cs[idx-1] + Cs[idx])
            print(f"  {REGIME_NAMES[prev]} → {REGIME_NAMES[int(labs[idx])]}: C ≈ {boundary:.2f}")
            prev = labs[idx]

    print(f"\nFinal form: C = K − {best_a2:.2f}·α²")
    print("Done.")


if __name__ == "__main__":
    main()
