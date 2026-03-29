#!/usr/bin/env python3
"""
Find a quadratic g(α) = p₁α + p₂α² such that C = K − g(α) is a
complexity parameter that maximally predicts which voting method is
rank 1.

Iso-regime curves: K = C + p₁α + p₂α² (same shape, shifted by C).

Metric: sort all (K, α) cells by C; count regime transitions.
Fewer transitions = C better separates regimes.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from scipy.optimize import minimize

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]
METHOD_LABELS = {
    "approval": "Approval", "borda": "Borda", "irv": "IRV",
    "minimax": "Minimax", "plurality": "Plurality",
    "random_dictator": "Rand. Dict.", "star": "STAR",
    "total_score": "Total Score",
}
METHOD_COLORS = {
    "approval": "#e41a1c", "borda": "#377eb8", "irv": "#4daf4a",
    "minimax": "#984ea3", "plurality": "#ff7f00",
    "random_dictator": "#a65628", "star": "#f781bf", "total_score": "#999999",
}

K_FULL = list(range(1, 21))
A_FULL = [round(a * 0.05, 2) for a in range(21)]
N_TERMINAL = 10


def load_data():
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


def winner_map(mean):
    """Rank-1 method index at each (K, α) among ALL 8 methods."""
    nK, nA, _ = mean.shape
    w = np.full((nK, nA), -1, dtype=int)
    for i in range(nK):
        for j in range(nA):
            vals = mean[i, j, :]
            if not np.all(np.isnan(vals)):
                w[i, j] = int(np.nanargmax(vals))
    return w


def count_transitions(winner, p1, p2):
    """
    Sort cells by C = K - p1*α - p2*α², count how many adjacent pairs
    have different rank-1 methods. Lower = better.
    """
    nK, nA = winner.shape
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if winner[i, j] >= 0:
                C = K - p1 * alpha - p2 * alpha ** 2
                points.append((C, winner[i, j]))
    points.sort()
    labs = [p[1] for p in points]
    return sum(1 for k in range(len(labs)-1) if labs[k] != labs[k+1])


def adjacent_agreement(winner, p1, p2):
    """Fraction of adjacent (by C) pairs with same rank-1 method."""
    nK, nA = winner.shape
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if winner[i, j] >= 0:
                C = K - p1 * alpha - p2 * alpha ** 2
                points.append((C, winner[i, j]))
    points.sort()
    labs = [p[1] for p in points]
    same = sum(1 for k in range(len(labs)-1) if labs[k] == labs[k+1])
    return same / (len(labs) - 1)


def epsilon_uniformity(winner, p1, p2, eps=0.5):
    """
    For each cell, find all other cells within ε in C-space.
    Fraction that share the same rank-1 method, averaged over all cells.
    """
    nK, nA = winner.shape
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if winner[i, j] >= 0:
                C = K - p1 * alpha - p2 * alpha ** 2
                points.append((C, winner[i, j]))
    points.sort()
    Cs = np.array([p[0] for p in points])
    labs = np.array([p[1] for p in points])
    n = len(Cs)

    total_agree = 0
    total_pairs = 0
    for i in range(n):
        # Binary search for neighbors within eps
        lo = np.searchsorted(Cs, Cs[i] - eps, side='left')
        hi = np.searchsorted(Cs, Cs[i] + eps, side='right')
        neighbors = hi - lo - 1  # exclude self
        if neighbors > 0:
            same = np.sum(labs[lo:hi] == labs[i]) - 1  # exclude self
            total_agree += same
            total_pairs += neighbors

    return total_agree / total_pairs if total_pairs > 0 else 0


def optimize(winner):
    """Find p1, p2 minimizing transitions."""
    print("\nGrid search over p1, p2 ...", flush=True)
    best = (0, 0, 999)

    # Coarse grid — needs wide range; general quadratic showed ~42α(1-α)
    for p1 in np.linspace(-100, 100, 201):
        for p2 in np.linspace(-100, 100, 201):
            t = count_transitions(winner, p1, p2)
            if t < best[2]:
                best = (p1, p2, t)

    p1_c, p2_c, t_c = best
    print(f"  Coarse: p1={p1_c:.1f}, p2={p2_c:.1f}, transitions={t_c}")

    # Fine grid around best
    for p1 in np.linspace(p1_c - 2, p1_c + 2, 81):
        for p2 in np.linspace(p2_c - 2, p2_c + 2, 81):
            t = count_transitions(winner, p1, p2)
            if t < best[2]:
                best = (p1, p2, t)

    p1_f, p2_f, t_f = best
    print(f"  Fine:   p1={p1_f:.2f}, p2={p2_f:.2f}, transitions={t_f}")

    # Also try scipy refinement
    res = minimize(lambda p: count_transitions(winner, p[0], p[1]),
                   [p1_f, p2_f], method='Nelder-Mead',
                   options={'xatol': 0.01, 'fatol': 0.5})
    p1_s, p2_s = res.x
    t_s = count_transitions(winner, p1_s, p2_s)
    print(f"  Scipy:  p1={p1_s:.2f}, p2={p2_s:.2f}, transitions={t_s}")

    if t_s <= t_f:
        return p1_s, p2_s, t_s
    return p1_f, p2_f, t_f


def plot_results(mean, winner, p1, p2):
    """Generate diagnostic plots."""
    nK, nA = winner.shape

    # ── 1. Winner map with iso-C contours ──
    C_grid = np.full((nK, nA), np.nan)
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            C_grid[i, j] = K - p1 * alpha - p2 * alpha ** 2

    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            if winner[i, j] >= 0:
                img[i, j] = to_rgb(METHOD_COLORS[METHODS[winner[i, j]]])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])

    levels = np.arange(np.floor(np.nanmin(C_grid)), np.ceil(np.nanmax(C_grid)) + 1, 1)
    cs = ax.contour(C_grid, levels=levels, colors='white', linewidths=0.8,
                    extent=[-0.5, nA-0.5, -0.5, nK-0.5], origin='lower')
    ax.clabel(cs, fontsize=6, fmt='%.0f')

    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                       rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title(f"C = K − ({p1:.2f})α − ({p2:.2f})α²\n"
                 f"Iso-C contours over winner map", fontsize=13)

    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in METHODS if m != "random_dictator"]
    handles.append(Patch(facecolor=METHOD_COLORS["random_dictator"],
                         label=METHOD_LABELS["random_dictator"]))
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "iso_regime_winner_map.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 2. 1D C-axis: regime separation scatter ──
    points = []
    for i, K in enumerate(K_FULL):
        for j, alpha in enumerate(A_FULL):
            if winner[i, j] >= 0:
                C = K - p1 * alpha - p2 * alpha ** 2
                points.append((C, winner[i, j]))
    points.sort()
    Cs = np.array([p[0] for p in points])
    labs = np.array([p[1] for p in points])

    fig, ax = plt.subplots(figsize=(14, 5))
    for midx, m in enumerate(METHODS):
        mask = (labs == midx)
        if mask.sum() > 0:
            jitter = np.random.normal(0, 0.12, mask.sum())
            ax.scatter(Cs[mask], midx + jitter, c=METHOD_COLORS[m],
                       s=15, alpha=0.7, label=METHOD_LABELS[m], edgecolors='none')

    ax.set_xlabel(f"C = K − {p1:.2f}α − {p2:.2f}α²", fontsize=12)
    ax.set_ylabel("Rank-1 method", fontsize=12)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
    ax.set_title(f"Regime separation along complexity axis C\n"
                 f"(transitions = {count_transitions(winner, p1, p2)})", fontsize=13)
    ax.legend(fontsize=8, loc='center right', ncol=1)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "iso_regime_scatter.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 3. Rank heatmaps with iso-C overlay ──
    ranks = np.full((nK, nA, len(METHODS)), np.nan)
    for i in range(nK):
        for j in range(nA):
            vals = mean[i, j, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                for r, idx in enumerate(order):
                    ranks[i, j, idx] = r + 1

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()
    cmap = plt.cm.RdYlGn_r

    for ax, method in zip(axes, METHODS):
        midx = METHODS.index(method)
        im = ax.imshow(ranks[:, :, midx], aspect='auto', origin='lower',
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA-0.5, -0.5, nK-0.5])
        cs = ax.contour(C_grid, levels=levels, colors='black', linewidths=0.6,
                        extent=[-0.5, nA-0.5, -0.5, nK-0.5], origin='lower',
                        alpha=0.5)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, nA, 4))
        ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 4)],
                           rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(0, nK, 2))
        ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)], fontsize=8)
        ax.set_xlabel("α", fontsize=9)
        ax.set_ylabel("K", fontsize=9)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"Mean fitness rank with iso-C contours: C = K − {p1:.2f}α − {p2:.2f}α²",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGDIR, "iso_regime_rank_heatmaps.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── 4. Print regime boundaries ──
    print(f"\nRegime boundaries along C = K − {p1:.2f}α − {p2:.2f}α²:")
    prev = labs[0]
    boundaries = []
    for k in range(1, len(labs)):
        if labs[k] != prev:
            boundary_C = 0.5 * (Cs[k-1] + Cs[k])
            fr = METHODS[prev]
            to = METHODS[int(labs[k])]
            boundaries.append((boundary_C, fr, to))
            prev = labs[k]

    # Deduplicate: find the dominant transitions
    from collections import Counter
    trans_pairs = Counter()
    for _, fr, to in boundaries:
        trans_pairs[(fr, to)] += 1

    print(f"  Total transitions: {len(boundaries)}")
    print(f"  Unique transition types:")
    for (fr, to), ct in trans_pairs.most_common():
        print(f"    {METHOD_LABELS[fr]} → {METHOD_LABELS[to]}: {ct}×")

    # ε-uniformity
    for eps in [0.25, 0.5, 1.0]:
        u = epsilon_uniformity(winner, p1, p2, eps=eps)
        print(f"\n  ε-uniformity (ε={eps}): {u:.4f}")


def main():
    print("=== Iso-Regime Curve Fitting ===\n")
    mean = load_data()
    w = winner_map(mean)

    p1, p2, trans = optimize(w)
    print(f"\n  OPTIMAL: C = K − ({p1:.2f})α − ({p2:.2f})α²")
    print(f"  = K − {p1:.2f}α − {p2:.2f}α²")
    # Express as K = C + p1*α + p2*α²
    # The shape g(α) = p1*α + p2*α²
    # Peak of g at α* = -p1/(2*p2) if p2 < 0
    if p2 != 0:
        alpha_peak = -p1 / (2 * p2)
        g_peak = p1 * alpha_peak + p2 * alpha_peak**2
        print(f"  g(α) peaks at α* = {alpha_peak:.3f}, g(α*) = {g_peak:.2f}")
    print(f"  Transitions: {trans}")
    adj = adjacent_agreement(w, p1, p2)
    print(f"  Adjacent agreement: {adj:.4f}")

    print("\nGenerating plots ...")
    plot_results(mean, w, p1, p2)
    print("\nDone.")


if __name__ == "__main__":
    main()
