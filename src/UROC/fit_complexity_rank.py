#!/usr/bin/env python3
"""
Find A, B such that C = K - A*α² - B*α is a complexity parameter
along which the FULL rank ordering of all 8 methods is maximally uniform.

Metric: for all pairs of (K,α) cells within ε in C-space, compute
the Kendall tau correlation of their 8-method rank vectors. Maximize
the mean correlation over A, B.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from scipy.optimize import minimize
from scipy.stats import kendalltau

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]
METHOD_LABELS = {m: l for m, l in zip(METHODS, [
    "Approval", "Borda", "IRV", "Minimax", "Plurality",
    "Rand. Dict.", "STAR", "Total Score"])}
METHOD_COLORS = {
    "approval": "#e41a1c", "borda": "#377eb8", "irv": "#4daf4a",
    "minimax": "#984ea3", "plurality": "#ff7f00",
    "random_dictator": "#a65628", "star": "#f781bf", "total_score": "#999999",
}

K_FULL = np.arange(1, 21, dtype=float)
A_FULL = np.round(np.arange(0, 1.05, 0.05), 2)


def load_data():
    d = np.load("results/direct_results.npz")
    return d['mean']  # (20, 21, 8)


def compute_ranks(mean):
    """Rank vector at each (K, α). Shape (20, 21, 8), rank 1=best."""
    nK, nA, nM = mean.shape
    ranks = np.full_like(mean, np.nan)
    for i in range(nK):
        for j in range(nA):
            vals = mean[i, j, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                for r, idx in enumerate(order):
                    ranks[i, j, idx] = r + 1
    return ranks


def rank_uniformity(ranks, A, B, eps=0.5):
    """
    Mean Kendall tau between rank vectors of cell pairs within ε in C-space.
    C = K - A*α² - B*α.  Higher = better.
    """
    nK, nA, nM = ranks.shape

    # Compute C for all cells, flatten
    Cs = np.zeros(nK * nA)
    rank_vecs = np.zeros((nK * nA, nM))
    idx = 0
    for i in range(nK):
        for j in range(nA):
            K = K_FULL[i]
            alpha = A_FULL[j]
            Cs[idx] = K - A * alpha**2 - B * alpha
            rank_vecs[idx] = ranks[i, j, :]
            idx += 1

    # Sort by C for efficient neighbor search
    order = np.argsort(Cs)
    Cs = Cs[order]
    rank_vecs = rank_vecs[order]
    n = len(Cs)

    # For each cell, compare with neighbors within ε
    total_corr = 0.0
    total_pairs = 0
    for i in range(n):
        # Binary search for right boundary
        j = i + 1
        while j < n and Cs[j] - Cs[i] <= eps:
            # Compute rank agreement (fraction of pairwise comparisons that agree)
            # Faster than full Kendall tau: count concordant pairs
            r1 = rank_vecs[i]
            r2 = rank_vecs[j]
            diff1 = r1[:, None] - r1[None, :]  # (8, 8)
            diff2 = r2[:, None] - r2[None, :]
            # Concordant: same sign (both positive or both negative)
            concordant = np.sum((diff1 * diff2) > 0)
            # Total comparable pairs (exclude ties and diagonal)
            total_comp = nM * (nM - 1)  # 56 for 8 methods
            tau = concordant / total_comp  # fraction concordant
            total_corr += tau
            total_pairs += 1
            j += 1

    return total_corr / total_pairs if total_pairs > 0 else 0


def rank_uniformity_fast(ranks, A, B, eps=0.5):
    """
    Faster version: use mean squared rank difference instead of Kendall tau.
    Lower = better (more uniform).
    """
    nK, nA, nM = ranks.shape
    n = nK * nA

    Cs = np.zeros(n)
    rvecs = np.zeros((n, nM))
    idx = 0
    for i in range(nK):
        for j in range(nA):
            Cs[idx] = K_FULL[i] - A * A_FULL[j]**2 - B * A_FULL[j]
            rvecs[idx] = ranks[i, j, :]
            idx += 1

    order = np.argsort(Cs)
    Cs = Cs[order]
    rvecs = rvecs[order]

    total_msd = 0.0
    total_pairs = 0
    for i in range(n):
        j = i + 1
        while j < n and Cs[j] - Cs[i] <= eps:
            total_msd += np.mean((rvecs[i] - rvecs[j])**2)
            total_pairs += 1
            j += 1

    return total_msd / total_pairs if total_pairs > 0 else 999


def optimize(ranks, eps=0.5):
    """Grid search + refinement for A, B minimizing rank MSD."""
    print(f"Grid search (eps={eps}) ...", flush=True)
    best = (0, 0, 999)

    # Coarse grid
    t0 = time.time()
    As = np.linspace(-60, 60, 61)
    Bs = np.linspace(-60, 60, 61)
    for A in As:
        for B in Bs:
            msd = rank_uniformity_fast(ranks, A, B, eps)
            if msd < best[2]:
                best = (A, B, msd)
    print(f"  Coarse ({time.time()-t0:.1f}s): A={best[0]:.1f}, B={best[1]:.1f}, "
          f"MSD={best[2]:.4f}")

    # Fine grid
    A_c, B_c = best[0], best[1]
    for A in np.linspace(A_c - 5, A_c + 5, 101):
        for B in np.linspace(B_c - 5, B_c + 5, 101):
            msd = rank_uniformity_fast(ranks, A, B, eps)
            if msd < best[2]:
                best = (A, B, msd)
    print(f"  Fine: A={best[0]:.2f}, B={best[1]:.2f}, MSD={best[2]:.4f}")

    # Scipy polish
    res = minimize(lambda p: rank_uniformity_fast(ranks, p[0], p[1], eps),
                   [best[0], best[1]], method='Nelder-Mead',
                   options={'xatol': 0.01, 'fatol': 1e-5})
    msd_s = rank_uniformity_fast(ranks, res.x[0], res.x[1], eps)
    if msd_s < best[2]:
        best = (res.x[0], res.x[1], msd_s)
    print(f"  Scipy: A={best[0]:.2f}, B={best[1]:.2f}, MSD={best[2]:.4f}")

    return best


def plot_results(mean, ranks, A, B, eps):
    nK, nA, nM = ranks.shape

    # C grid
    C_grid = np.zeros((nK, nA))
    for i in range(nK):
        for j in range(nA):
            C_grid[i, j] = K_FULL[i] - A * A_FULL[j]**2 - B * A_FULL[j]

    # Winner map
    winner = np.argmax(mean, axis=2)

    # ── 1. Winner map + iso-C contours ──
    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            img[i, j] = to_rgb(METHOD_COLORS[METHODS[winner[i, j]]])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])
    levels = np.arange(np.floor(C_grid.min()), np.ceil(C_grid.max()) + 1, 1)
    cs = ax.contour(C_grid, levels=levels, colors='white', linewidths=0.8,
                    extent=[-0.5, nA-0.5, -0.5, nK-0.5], origin='lower')
    ax.clabel(cs, fontsize=6, fmt='%.0f')
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                       rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(int(K_FULL[i])) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title(f"C = K − {A:.2f}α² − {B:.2f}α  (ε={eps})\n"
                 f"Iso-C contours over rank-1 winner map", fontsize=13)
    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in METHODS]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "rank_uniformity_winner.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 2. Rank heatmaps + iso-C contours ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()
    cmap = plt.cm.RdYlGn_r
    for ax, (midx, method) in zip(axes, enumerate(METHODS)):
        im = ax.imshow(ranks[:, :, midx], aspect='auto', origin='lower',
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA-0.5, -0.5, nK-0.5])
        ax.contour(C_grid, levels=levels, colors='black', linewidths=0.5,
                   extent=[-0.5, nA-0.5, -0.5, nK-0.5], origin='lower', alpha=0.4)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, nA, 4))
        ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 4)],
                           rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(0, nK, 2))
        ax.set_yticklabels([str(int(K_FULL[i])) for i in range(0, nK, 2)], fontsize=8)
        ax.set_xlabel("α", fontsize=9)
        ax.set_ylabel("K", fontsize=9)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1,2,3,4,5,6,7,8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"Rank heatmaps with iso-C contours: C = K − {A:.2f}α² − {B:.2f}α",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGDIR, "rank_uniformity_heatmaps.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 3. Mean rank vs C for each method (the money plot) ──
    # Bin cells by C, compute mean rank of each method per bin
    all_C = C_grid.ravel()
    C_min, C_max = all_C.min(), all_C.max()
    n_bins = 30
    bin_edges = np.linspace(C_min, C_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(12, 6))
    for midx, method in enumerate(METHODS):
        rank_flat = ranks[:, :, midx].ravel()
        bin_means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = (all_C >= bin_edges[b]) & (all_C < bin_edges[b+1])
            if mask.sum() > 0:
                bin_means[b] = np.mean(rank_flat[mask])
        valid = ~np.isnan(bin_means)
        ax.plot(bin_centers[valid], bin_means[valid],
                color=METHOD_COLORS[method], linewidth=2,
                label=METHOD_LABELS[method], marker='o', markersize=4)

    ax.set_xlabel(f"C = K − {A:.2f}α² − {B:.2f}α", fontsize=12)
    ax.set_ylabel("Mean rank (1=best, 8=worst)", fontsize=12)
    ax.set_title("Method rank vs complexity parameter C", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(FIGDIR, "rank_vs_complexity.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print("=== Full-Rank Uniformity Fit ===\n")
    mean = load_data()
    ranks = compute_ranks(mean)

    # Optimize for different epsilon values
    for eps in [0.5, 1.0]:
        print(f"\n--- eps = {eps} ---")
        A, B, msd = optimize(ranks, eps=eps)
        print(f"\n  Result: C = K − {A:.2f}α² − {B:.2f}α")
        if A != 0:
            alpha_peak = -B / (2 * A)
            g_peak = A * alpha_peak**2 + B * alpha_peak
            print(f"  g(α) = {A:.2f}α² + {B:.2f}α")
            print(f"  g peaks at α = {alpha_peak:.3f}, g(α*) = {g_peak:.2f}")
        print(f"  Mean squared rank difference within ε: {msd:.4f}")

        # Baseline: C = K (no α correction)
        msd0 = rank_uniformity_fast(ranks, 0, 0, eps)
        print(f"  Baseline (C=K): MSD = {msd0:.4f}")
        print(f"  Improvement: {(msd0 - msd) / msd0 * 100:.1f}%")

    # Use eps=1.0 result for plots
    A, B, _ = optimize(ranks, eps=1.0)
    print(f"\nGenerating plots with A={A:.2f}, B={B:.2f} ...")
    plot_results(mean, ranks, A, B, eps=1.0)
    print("\nDone.")


if __name__ == "__main__":
    main()
