#!/usr/bin/env python3
"""
Find individual quadratic curves K = a1*α² + a2*α + a3 along which
the full 8-method rank ordering is maximally uniform.

Procedure:
  1. For each seed point (K0, α0), find (a1, a2, a3) such that the curve
     passes near (K0, α0) and rank vectors along it are most similar.
  2. Collect all fitted curves.
  3. Analyze how (a1, a2, a3) vary across the space.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

ALL_METHODS = ["approval", "borda", "irv", "minimax", "plurality",
               "random_dictator", "star", "total_score"]
# Methods used for rank uniformity optimization (exclude approval & random dictator)
METHODS = ["borda", "irv", "minimax", "plurality", "star", "total_score"]
METHOD_LABELS = {
    "borda": "Borda", "irv": "IRV", "minimax": "Minimax",
    "plurality": "Plurality", "star": "STAR", "total_score": "Total Score",
}
METHOD_COLORS = {
    "borda": "#377eb8", "irv": "#4daf4a",
    "minimax": "#984ea3", "plurality": "#ff7f00",
    "star": "#f781bf", "total_score": "#999999",
}

K_FULL = np.arange(1, 21, dtype=float)
A_FULL = np.round(np.arange(0, 1.05, 0.05), 2)


def load_data():
    d = np.load("results/direct_results.npz")
    all_methods = list(d['methods'])
    keep = [all_methods.index(m) for m in METHODS]
    return d['mean'][:, :, keep]  # (20, 21, 6)


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


def curve_K(alpha, a1, a2, a3):
    """K = a1*α² + a2*α + a3"""
    return a1 * alpha**2 + a2 * alpha + a3


def rank_msd_along_curve(ranks, a1, a2, a3, eps_K=1.0):
    """
    Mean squared rank difference for all cell pairs that both lie
    within eps_K of the curve K = a1*α² + a2*α + a3.

    Lower = more uniform ranking along the curve.
    Also returns the number of cells captured.
    """
    nK, nA, nM = ranks.shape
    # Collect cells near the curve
    near_cells = []
    for j, alpha in enumerate(A_FULL):
        K_pred = curve_K(alpha, a1, a2, a3)
        for i, K in enumerate(K_FULL):
            if abs(K - K_pred) <= eps_K:
                near_cells.append(ranks[i, j, :])

    if len(near_cells) < 2:
        return 999.0, 0

    near_cells = np.array(near_cells)
    n = len(near_cells)

    # Mean squared difference of all pairs
    total_msd = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_msd += np.mean((near_cells[i] - near_cells[j])**2)
            n_pairs += 1

    return total_msd / n_pairs if n_pairs > 0 else 999.0, n


def fit_curve_at_seed(ranks, K0, alpha0, eps_K=1.0):
    """
    Find (a1, a2, a3) that:
      - passes through (alpha0, K0): a1*alpha0² + a2*alpha0 + a3 = K0
      - minimizes rank MSD along the curve within eps_K

    Constraint: a3 = K0 - a1*alpha0² - a2*alpha0
    So we optimize over (a1, a2) only.
    """
    def loss(params):
        a1, a2 = params
        a3 = K0 - a1 * alpha0**2 - a2 * alpha0
        msd, n = rank_msd_along_curve(ranks, a1, a2, a3, eps_K)
        # Penalize curves that capture too few cells
        if n < 5:
            return 999.0
        return msd

    best = (0, 0, 999.0)
    # Grid search — moderate curvatures only to avoid degenerate fits
    for a1 in np.linspace(-30, 30, 61):
        for a2 in np.linspace(-30, 30, 61):
            val = loss([a1, a2])
            if val < best[2]:
                best = (a1, a2, val)

    # Refine
    res = minimize(loss, [best[0], best[1]], method='Nelder-Mead',
                   options={'xatol': 0.1, 'fatol': 1e-4})
    a1_opt, a2_opt = res.x
    a3_opt = K0 - a1_opt * alpha0**2 - a2_opt * alpha0
    msd_opt, n_opt = rank_msd_along_curve(ranks, a1_opt, a2_opt, a3_opt, eps_K)

    return a1_opt, a2_opt, a3_opt, msd_opt, n_opt


def main():
    print("=== Local Curve Fitting ===\n")
    mean = load_data()
    ranks = compute_ranks(mean)
    nK, nA, nM = ranks.shape

    eps_K = 1.0  # cells within ±1 K of curve
    print(f"eps_K = {eps_K}\n")

    # Seed points: sample across the (K, α) space
    # Use a grid of seed points
    seed_Ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    seed_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = []
    print(f"{'K0':>4s} {'α0':>5s} | {'a1':>8s} {'a2':>8s} {'a3':>8s} | {'MSD':>6s} {'n':>3s}")
    print("-" * 60)

    for K0 in seed_Ks:
        for alpha0 in seed_alphas:
            a1, a2, a3, msd, n = fit_curve_at_seed(ranks, K0, alpha0, eps_K)
            results.append({
                'K0': K0, 'alpha0': alpha0,
                'a1': a1, 'a2': a2, 'a3': a3,
                'msd': msd, 'n': n
            })
            print(f"{K0:4.0f} {alpha0:5.2f} | {a1:8.2f} {a2:8.2f} {a3:8.2f} | "
                  f"{msd:6.3f} {n:3d}")

    # ── Analysis of fitted coefficients ──
    print("\n=== Coefficient Analysis ===\n")
    a1s = np.array([r['a1'] for r in results])
    a2s = np.array([r['a2'] for r in results])
    a3s = np.array([r['a3'] for r in results])
    msds = np.array([r['msd'] for r in results])
    K0s = np.array([r['K0'] for r in results])
    alpha0s = np.array([r['alpha0'] for r in results])

    # Weight by quality (inverse MSD)
    good = msds < 5.0  # filter out bad fits
    print(f"Good fits (MSD < 5): {good.sum()} / {len(results)}")
    print(f"\nAmong good fits:")
    print(f"  a1: mean={a1s[good].mean():.2f}, std={a1s[good].std():.2f}, "
          f"range=[{a1s[good].min():.2f}, {a1s[good].max():.2f}]")
    print(f"  a2: mean={a2s[good].mean():.2f}, std={a2s[good].std():.2f}, "
          f"range=[{a2s[good].min():.2f}, {a2s[good].max():.2f}]")
    print(f"  a3: mean={a3s[good].mean():.2f}, std={a3s[good].std():.2f}, "
          f"range=[{a3s[good].min():.2f}, {a3s[good].max():.2f}]")

    # Check if a1 and a2 correlate with a3 (the "complexity" parameter)
    if good.sum() > 3:
        from numpy.polynomial import polynomial as P
        # a1 vs a3
        c1 = np.corrcoef(a3s[good], a1s[good])[0, 1]
        c2 = np.corrcoef(a3s[good], a2s[good])[0, 1]
        print(f"\n  Correlation a1 vs a3: {c1:.3f}")
        print(f"  Correlation a2 vs a3: {c2:.3f}")

        # If a1 and a2 are roughly constant → single parameter (a3) suffices
        # If they vary with a3 → need to model the variation
        a1_cv = a1s[good].std() / (abs(a1s[good].mean()) + 1e-10)
        a2_cv = a2s[good].std() / (abs(a2s[good].mean()) + 1e-10)
        print(f"\n  a1 coefficient of variation: {a1_cv:.3f}")
        print(f"  a2 coefficient of variation: {a2_cv:.3f}")
        if a1_cv < 0.3 and a2_cv < 0.3:
            print("  → a1, a2 are roughly constant: single-parameter C ≈ a3 suffices!")
            print(f"  → Curve family: K ≈ {a1s[good].mean():.1f}α² + "
                  f"{a2s[good].mean():.1f}α + C")
        else:
            print("  → a1, a2 vary: shape changes with complexity level")

    # ── Plots ──
    print("\nGenerating plots ...")

    # Plot 1: All fitted curves on winner map
    winner = np.argmax(mean, axis=2)
    from matplotlib.colors import to_rgb

    fig, ax = plt.subplots(figsize=(10, 7))
    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            img[i, j] = to_rgb(METHOD_COLORS[METHODS[winner[i, j]]])
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])

    alpha_fine = np.linspace(0, 1, 200)
    cmap_curves = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for idx, r in enumerate(results):
        if r['msd'] < 5.0:
            K_curve = curve_K(alpha_fine, r['a1'], r['a2'], r['a3'])
            x_plot = alpha_fine / 0.05
            y_plot = K_curve - 1
            # Clip to plot region
            mask = (y_plot >= -0.5) & (y_plot <= nK - 0.5)
            if mask.sum() > 1:
                ax.plot(x_plot[mask], y_plot[mask], color=cmap_curves[idx],
                        linewidth=1.5, alpha=0.7)

    ax.set_xlim(-0.5, nA - 0.5)
    ax.set_ylim(-0.5, nK - 0.5)
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                       rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(int(K_FULL[i])) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title("Fitted iso-rank curves on winner map\n"
                 "(color = complexity level, light→dark)", fontsize=13)
    from matplotlib.patches import Patch as MPatch
    handles = [MPatch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in METHODS]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "local_curves_winner.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # Plot 2: Coefficient variation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (coeff, name) in zip(axes, [(a3s, 'a3 (≈ complexity)'),
                                         (a1s, 'a1 (α² coeff)'),
                                         (a2s, 'a2 (α coeff)')]):
        sc = ax.scatter(alpha0s[good], K0s[good], c=coeff[good],
                        cmap='viridis', s=80, edgecolors='black', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel("seed α₀")
        ax.set_ylabel("seed K₀")
        ax.set_title(name)
    fig.suptitle("How curve coefficients vary across (K, α) space", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "local_curves_coefficients.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # Plot 3: a1 and a2 vs a3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(a3s[good], a1s[good], c='blue', s=40, alpha=0.7)
    axes[0].set_xlabel("a3 (complexity)")
    axes[0].set_ylabel("a1 (α² coefficient)")
    axes[0].set_title(f"a1 vs a3  (corr={np.corrcoef(a3s[good], a1s[good])[0,1]:.3f})")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(a3s[good], a2s[good], c='red', s=40, alpha=0.7)
    axes[1].set_xlabel("a3 (complexity)")
    axes[1].set_ylabel("a2 (α coefficient)")
    axes[1].set_title(f"a2 vs a3  (corr={np.corrcoef(a3s[good], a2s[good])[0,1]:.3f})")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Do curve shapes change with complexity?", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "local_curves_a1a2_vs_a3.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
