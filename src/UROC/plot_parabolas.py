#!/usr/bin/env python3
"""
Overlay parabolic iso-complexity curves K = c * 6α(1-α) on the mean
fitness rank heatmaps.  For each of the 6 target methods, find the
coefficient c that maximally overlaps with its rank-1 region.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Methods to fit parabolas for, in complexity order
TARGET = ["total_score", "approval", "plurality", "irv", "borda", "star"]
TARGET_COLORS = {
    "total_score": "#999999",
    "approval":    "#e41a1c",
    "plurality":   "#ff7f00",
    "irv":         "#4daf4a",
    "borda":       "#377eb8",
    "star":        "#f781bf",
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


def compute_ranks(mean):
    """Rank methods 1-8 at each (K, alpha). Rank 1 = best."""
    nK, nA, nM = mean.shape
    ranks = np.full((nK, nA, nM), np.nan)
    for i in range(nK):
        for j in range(nA):
            vals = mean[i, j, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                for r, idx in enumerate(order):
                    ranks[i, j, idx] = r + 1
    return ranks


def parabola_K(alpha, c):
    """K = c * 6 * alpha * (1 - alpha)."""
    return c * 6.0 * alpha * (1.0 - alpha)


def fit_c_for_method(ranks, method_idx, c_range=np.linspace(0.5, 50, 500)):
    """
    Find c that maximizes overlap of the parabola K = c*6α(1-α)
    with rank-1 cells for the given method.

    For each c, walk along the parabola and count how many (K, α) cells
    within ±1 of the curve have rank 1 for this method.
    """
    nK, nA = ranks.shape[:2]
    best_c, best_score = 0, -1

    for c in c_range:
        score = 0
        total = 0
        for j, alpha in enumerate(A_FULL):
            K_pred = parabola_K(alpha, c)
            # Check cells within ±1 of the parabola
            for i, K in enumerate(K_FULL):
                if abs(K - K_pred) <= 1.0:
                    if not np.isnan(ranks[i, j, method_idx]):
                        total += 1
                        if ranks[i, j, method_idx] == 1:
                            score += 1
        if total > 0:
            frac = score / total
            if frac > best_score or (frac == best_score and c > best_c):
                best_score = frac
                best_c = c
    return best_c, best_score


def main():
    print("=== Parabolic Overlay on Rank Heatmaps ===\n")
    mean = load_data()
    ranks = compute_ranks(mean)
    nK, nA, nM = ranks.shape

    # Fit c for each target method
    print("Fitting parabola coefficients ...")
    c_values = {}
    for m in TARGET:
        midx = METHODS.index(m)
        c, score = fit_c_for_method(ranks, midx)
        c_values[m] = c
        print(f"  {METHOD_LABELS[m]:12s}: c = {c:.1f}  (rank-1 overlap = {score:.1%})")

    # ── Figure 1: 2×4 rank heatmaps with ALL parabolas overlaid ──
    print("\nPlotting rank heatmaps with parabola overlay ...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()
    cmap = plt.cm.RdYlGn_r

    alpha_fine = np.linspace(0, 1, 200)

    for ax, method in zip(axes, METHODS):
        midx = METHODS.index(method)
        rank_mat = ranks[:, :, midx]

        im = ax.imshow(rank_mat, aspect='auto', origin='lower',
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA-0.5, -0.5, nK-0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, nA, 4))
        ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 4)],
                           rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(0, nK, 2))
        ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)], fontsize=8)
        ax.set_xlabel("α", fontsize=9)
        ax.set_ylabel("K", fontsize=9)

        # Overlay all target parabolas
        for tm in TARGET:
            c = c_values[tm]
            K_curve = parabola_K(alpha_fine, c)
            # Convert to plot coordinates: α → x index, K → y index
            x_plot = alpha_fine / 0.05  # α index (0 to 20)
            y_plot = K_curve - 1        # K index (K=1 → 0)
            color = TARGET_COLORS[tm]
            lw = 3.0 if tm == method else 1.0
            ls = '-' if tm == method else '--'
            alpha_line = 1.0 if tm == method else 0.5
            ax.plot(x_plot, y_plot, color=color, linewidth=lw,
                    linestyle=ls, alpha=alpha_line)

        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        cb.ax.tick_params(labelsize=6)

    # Legend for parabolas
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color=TARGET_COLORS[m], linewidth=2,
                           label=f"{METHOD_LABELS[m]} (c={c_values[m]:.1f})")
                    for m in TARGET]
    fig.legend(handles=legend_lines, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Mean fitness rank with iso-complexity parabolas: K = c · 6α(1−α)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = os.path.join(FIGDIR, "phaseF1_parabola_overlay.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    # ── Figure 2: Single combined winner map with all parabolas ──
    print("Plotting combined winner map ...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Winner among all 8 methods
    winner = np.full((nK, nA), -1, dtype=int)
    for i in range(nK):
        for j in range(nA):
            vals = mean[i, j, :]
            if not np.all(np.isnan(vals)):
                winner[i, j] = np.nanargmax(vals)

    # Color by winner
    from matplotlib.colors import to_rgb
    all_colors = {
        "approval": "#e41a1c", "borda": "#377eb8", "irv": "#4daf4a",
        "minimax": "#984ea3", "plurality": "#ff7f00", "random_dictator": "#a65628",
        "star": "#f781bf", "total_score": "#999999",
    }
    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            if winner[i, j] >= 0:
                img[i, j] = to_rgb(all_colors[METHODS[winner[i, j]]])

    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])

    for tm in TARGET:
        c = c_values[tm]
        K_curve = parabola_K(alpha_fine, c)
        x_plot = alpha_fine / 0.05
        y_plot = K_curve - 1
        ax.plot(x_plot, y_plot, color='white', linewidth=2.5)
        ax.plot(x_plot, y_plot, color=TARGET_COLORS[tm], linewidth=1.5,
                label=f"{METHOD_LABELS[tm]} (c={c:.1f})")

    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                       rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title("Winner map with iso-complexity parabolas: K = c · 6α(1−α)", fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    path = os.path.join(FIGDIR, "winner_map_parabolas.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
