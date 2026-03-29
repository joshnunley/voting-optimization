#!/usr/bin/env python3
"""
Fit a 1-parameter family of quadratic curves by interpolating between
two endpoint curves.

Curve at complexity t:
  K = a1(t)*α² + a2(t)*α + a3(t)
  where ai(t) = ai_lo + t*(ai_hi - ai_lo)

For each (K, α): t* = (K - L(α)) / D(α)
  where L(α) = a1_lo*α² + a2_lo*α + a3_lo
        D(α) = Δa1*α² + Δa2*α + Δa3

Optimize 6 params (a1_lo, a2_lo, a3_lo, a1_hi, a2_hi, a3_hi)
to maximize rank uniformity along iso-t curves.

Excludes approval and random_dictator.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from scipy.optimize import minimize, differential_evolution

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

ALL_METHODS = ["approval", "borda", "irv", "minimax", "plurality",
               "random_dictator", "star", "total_score"]
METHODS = ["borda", "irv", "minimax", "plurality", "star", "total_score"]
METHOD_LABELS = {"borda": "Borda", "irv": "IRV", "minimax": "Minimax",
                 "plurality": "Plurality", "star": "STAR", "total_score": "Total Score"}
METHOD_COLORS = {"borda": "#377eb8", "irv": "#4daf4a", "minimax": "#984ea3",
                 "plurality": "#ff7f00", "star": "#f781bf", "total_score": "#999999"}

K_FULL = np.arange(1, 21, dtype=float)
A_FULL = np.round(np.arange(0, 1.05, 0.05), 2)
nK, nA = len(K_FULL), len(A_FULL)


def load_data():
    d = np.load("results/direct_results.npz")
    all_m = list(d['methods'])
    keep = [all_m.index(m) for m in METHODS]
    return d['mean'][:, :, keep]


def compute_ranks(mean):
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


def compute_t_map(params):
    """
    For each (K, α), compute t* = (K - L(α)) / D(α).
    Returns (nK, nA) array of t values.
    """
    a1_lo, a2_lo, a3_lo, a1_hi, a2_hi, a3_hi = params
    da1 = a1_hi - a1_lo
    da2 = a2_hi - a2_lo
    da3 = a3_hi - a3_lo

    t_map = np.full((nK, nA), np.nan)
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            L = a1_lo * alpha**2 + a2_lo * alpha + a3_lo
            D = da1 * alpha**2 + da2 * alpha + da3
            if abs(D) > 1e-6:
                t_map[i, j] = (K - L) / D
            else:
                t_map[i, j] = np.nan
    return t_map


def rank_msd_score(params, ranks, eps_t=0.05):
    """
    For all pairs of cells within eps_t in t-space,
    compute mean squared rank difference. Lower = better.
    Also penalize cells outside [0, 1] in t-space (poor coverage).
    """
    t_map = compute_t_map(params)
    nM = ranks.shape[2]

    # Flatten valid cells
    valid = ~np.isnan(t_map)
    ts = t_map[valid]
    rvecs = ranks[valid]

    if len(ts) < 10:
        return 999.0

    # Penalize cells outside [0, 1]
    out_of_range = np.mean((ts < 0) | (ts > 1))
    if out_of_range > 0.3:
        return 999.0

    # Sort by t
    order = np.argsort(ts)
    ts = ts[order]
    rvecs = rvecs[order]
    n = len(ts)

    # Compute MSD for neighbors within eps_t
    total_msd = 0.0
    n_pairs = 0
    for i in range(n):
        j = i + 1
        while j < n and ts[j] - ts[i] <= eps_t:
            total_msd += np.mean((rvecs[i] - rvecs[j])**2)
            n_pairs += 1
            j += 1

    if n_pairs == 0:
        return 999.0

    msd = total_msd / n_pairs
    # Add penalty for poor coverage
    coverage_penalty = out_of_range * 5.0
    return msd + coverage_penalty


def optimize(ranks):
    print("Optimizing 6 params via differential evolution ...", flush=True)
    t0 = time.time()

    # Bounds: curves should span K=0..25 ish
    bounds = [
        (-30, 30),   # a1_lo
        (-30, 30),   # a2_lo
        (-5, 25),    # a3_lo (low-end K intercept)
        (-30, 30),   # a1_hi
        (-30, 30),   # a2_hi
        (-5, 25),    # a3_hi (high-end K intercept)
    ]

    def objective(params):
        return rank_msd_score(params, ranks, eps_t=0.05)

    result = differential_evolution(objective, bounds, seed=42,
                                     maxiter=200, tol=1e-4,
                                     popsize=20, polish=True)

    params = result.x
    score = result.fun
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Score (MSD): {score:.4f}")
    print(f"  a1_lo={params[0]:.2f}, a2_lo={params[1]:.2f}, a3_lo={params[2]:.2f}")
    print(f"  a1_hi={params[3]:.2f}, a2_hi={params[4]:.2f}, a3_hi={params[5]:.2f}")

    # Baseline: flat (no α correction)
    flat_params = [0, 0, 0, 0, 0, 20]
    flat_score = rank_msd_score(flat_params, ranks, eps_t=0.05)
    print(f"  Baseline (flat C=K): {flat_score:.4f}")
    print(f"  Improvement: {(flat_score - score) / flat_score * 100:.1f}%")

    return params, score


def plot_results(mean, ranks, params):
    nM = len(METHODS)
    t_map = compute_t_map(params)
    a1_lo, a2_lo, a3_lo, a1_hi, a2_hi, a3_hi = params

    # ── 1. t-map heatmap ──
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(t_map, aspect='auto', origin='lower', cmap='viridis',
                   vmin=0, vmax=1,
                   extent=[-0.5, nA-0.5, -0.5, nK-0.5])
    plt.colorbar(im, ax=ax, label='Complexity t')
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                       rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(int(K_FULL[i])) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title("Complexity parameter t across (K, α) space", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "interp_t_map.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 2. Iso-t curves on winner map ──
    winner = np.argmax(mean, axis=2)
    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            img[i, j] = to_rgb(METHOD_COLORS[METHODS[winner[i, j]]])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])

    # Draw iso-t curves
    alpha_fine = np.linspace(0, 1, 200)
    for t in np.linspace(0, 1, 15):
        a1 = a1_lo + t * (a1_hi - a1_lo)
        a2 = a2_lo + t * (a2_hi - a2_lo)
        a3 = a3_lo + t * (a3_hi - a3_lo)
        K_curve = a1 * alpha_fine**2 + a2 * alpha_fine + a3
        x_plot = alpha_fine / 0.05
        y_plot = K_curve - 1
        mask = (y_plot >= -0.5) & (y_plot <= nK - 0.5)
        if mask.sum() > 1:
            ax.plot(x_plot[mask], y_plot[mask], 'w-', linewidth=1.2, alpha=0.7)

    ax.set_xlim(-0.5, nA - 0.5)
    ax.set_ylim(-0.5, nK - 0.5)
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                       rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(int(K_FULL[i])) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title("Iso-t curves (interpolated family) on winner map", fontsize=13)
    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in METHODS]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "interp_winner_map.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 3. Mean rank vs t (the key plot) ──
    valid = ~np.isnan(t_map) & (t_map >= 0) & (t_map <= 1)
    ts_flat = t_map[valid]
    n_bins = 25
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(12, 6))
    for midx, method in enumerate(METHODS):
        rank_flat = ranks[:, :, midx][valid]
        bin_means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = (ts_flat >= bin_edges[b]) & (ts_flat < bin_edges[b+1])
            if mask.sum() > 0:
                bin_means[b] = np.mean(rank_flat[mask])
        v = ~np.isnan(bin_means)
        ax.plot(bin_centers[v], bin_means[v], color=METHOD_COLORS[method],
                linewidth=2.5, label=METHOD_LABELS[method], marker='o', markersize=5)

    ax.set_xlabel("Complexity t", fontsize=12)
    ax.set_ylabel("Mean rank (1=best, 6=worst)", fontsize=12)
    ax.set_title("Method rank vs interpolated complexity parameter", fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(FIGDIR, "interp_rank_vs_t.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 4. Rank heatmaps + iso-t contours ──
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    cmap = plt.cm.RdYlGn_r
    t_levels = np.linspace(0, 1, 11)

    for ax, (midx, method) in zip(axes, enumerate(METHODS)):
        im = ax.imshow(ranks[:, :, midx], aspect='auto', origin='lower',
                       cmap=cmap, vmin=1, vmax=6,
                       extent=[-0.5, nA-0.5, -0.5, nK-0.5])
        t_clipped = np.clip(t_map, 0, 1)
        ax.contour(t_clipped, levels=t_levels, colors='black', linewidths=0.5,
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
        cb.set_ticks([1, 2, 3, 4, 5, 6])
        cb.ax.tick_params(labelsize=7)

    fig.suptitle("Rank heatmaps with iso-t contours", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGDIR, "interp_rank_heatmaps.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 5. Print curve equations at t=0, 0.5, 1 ──
    print("\nCurve family:")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        a1 = a1_lo + t * (a1_hi - a1_lo)
        a2 = a2_lo + t * (a2_hi - a2_lo)
        a3 = a3_lo + t * (a3_hi - a3_lo)
        print(f"  t={t:.2f}: K = {a1:.2f}α² + {a2:.2f}α + {a3:.2f}")


def main():
    print("=== Interpolated Curve Family ===\n")
    mean = load_data()
    ranks = compute_ranks(mean)

    params, score = optimize(ranks)
    print("\nGenerating plots ...")
    plot_results(mean, ranks, params)
    print("\nDone.")


if __name__ == "__main__":
    main()
