#!/usr/bin/env python3
"""
Representative democracy analysis.

Figures produced:
  RepR1 - Cost-of-representation heatmaps: Δ(best_rep − direct) for each method
  RepR2 - Method rank heatmaps within representative democracy (best config)
  RepR3 - Candidate selection strategy comparison (config winner map + mean gain)
  RepR4 - The Borda paradox: rank of Borda (direct vs rep) side-by-side
  RepR5 - Approval recovery: Approval's rank improvement under representation
  RepR6 - Random-dictator filter effect: dramatic benefit of candidate pre-selection
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

BASE_D  = "results/direct"
BASE_R  = "results/representative"
FIGDIR  = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]
METHOD_LABELS = {
    "approval":        "Approval",
    "borda":           "Borda",
    "irv":             "IRV",
    "minimax":         "Minimax",
    "plurality":       "Plurality",
    "random_dictator": "Rand. Dict.",
    "star":            "STAR",
    "total_score":     "Total Score",
}
METHOD_COLORS = {
    "approval":        "#e41a1c",
    "borda":           "#377eb8",
    "irv":             "#4daf4a",
    "minimax":         "#984ea3",
    "plurality":       "#ff7f00",
    "random_dictator": "#a65628",
    "star":            "#f781bf",
    "total_score":     "#999999",
}

# Representative democracy configurations
CONFIGS = ["c3_t0.1", "c3_t1.0", "c3_tuniform", "c5_t0.1", "c5_t1.0", "c5_tuniform"]
CONFIG_LABELS = {
    "c3_t0.1":     "3 cand, τ=0.1",
    "c3_t1.0":     "3 cand, τ=1.0",
    "c3_tuniform": "3 cand, random",
    "c5_t0.1":     "5 cand, τ=0.1",
    "c5_t1.0":     "5 cand, τ=1.0",
    "c5_tuniform": "5 cand, random",
}
CONFIG_COLORS = {
    "c3_t0.1":     "#1f77b4",
    "c3_t1.0":     "#aec7e8",
    "c3_tuniform": "#d62728",
    "c5_t0.1":     "#2ca02c",
    "c5_t1.0":     "#98df8a",
    "c5_tuniform": "#ff7f0e",
}

# Available (K, alpha) pairs for representative analysis
K_LIST = [5, 10, 15]
A_LIST = [0.00, 0.25, 0.50, 0.75, 1.00]
A_REAL = [0.25, 0.50, 0.75]   # "realistic" alpha values


# ---- Data loading ----

def _load_terminal(base, subpath, n_tail=10):
    """Load mean of last n_tail steps across all runs. Returns np.nan on missing."""
    try:
        h = np.load(os.path.join(base, subpath, "mean_history.npy"))
        return float(h[:, -n_tail:].mean())
    except Exception:
        return np.nan


def load_direct(K, alpha, method):
    return _load_terminal(BASE_D, f"K{K}_a{alpha:.2f}/{method}")


def load_rep(K, alpha, config, method):
    return _load_terminal(BASE_R, f"K{K}_a{alpha:.2f}/{config}/{method}")


def _build_terminal_matrix(loader_fn):
    """Build (len(K_LIST), len(A_LIST), len(METHODS)) array using loader_fn(K, a, m)."""
    mat = np.full((len(K_LIST), len(A_LIST), len(METHODS)), np.nan)
    for ki, K in enumerate(K_LIST):
        for ai, a in enumerate(A_LIST):
            for mi, m in enumerate(METHODS):
                mat[ki, ai, mi] = loader_fn(K, a, m)
    return mat


def _save(fig, name, dpi=150):
    path = os.path.join(FIGDIR, name + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---- Precompute full data tables ----

print("Loading data...")
direct_mat  = _build_terminal_matrix(load_direct)   # (K, A, M)
rep_mats    = {}  # config -> (K, A, M)
for cfg in CONFIGS:
    rep_mats[cfg] = _build_terminal_matrix(
        lambda K, a, m, cfg=cfg: load_rep(K, a, cfg, m)
    )

# Best rep config per (K, A, M)
rep_stack = np.stack([rep_mats[c] for c in CONFIGS], axis=0)  # (nC, K, A, M)
best_rep   = np.nanmax(rep_stack, axis=0)   # (K, A, M) - best value across configs
best_rep_idx = np.nanargmax(rep_stack, axis=0)  # index into CONFIGS

# Delta: best_rep - direct
delta_mat = best_rep - direct_mat   # (K, A, M)

print("Data loaded.")


# ======================================================================
# Fig RepR1: Cost-of-representation heatmaps (2×4 grid, one per method)
# ======================================================================
def fig_repr1_cost_heatmaps():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Cost / Benefit of Representative Democracy\n"
                 r"$\Delta$ terminal fitness (best representative − direct)",
                 fontsize=13, y=1.01)

    vmax = np.nanpercentile(np.abs(delta_mat), 95)
    vmin = -vmax

    for mi, m in enumerate(METHODS):
        ax = axes[mi // 4, mi % 4]
        data = delta_mat[:, :, mi]          # (K, A)

        # Exclude alpha=0 and alpha=1 rows from color scale but still show them
        real_vals = delta_mat[:, 1:4, mi].ravel()
        vmax_m = np.nanpercentile(np.abs(real_vals), 98) if len(real_vals) > 0 else 0.5
        vmin_m = -vmax_m

        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap="RdYlGn", vmin=vmin_m, vmax=vmax_m,
                       interpolation="nearest")

        # Annotate each cell with Δ value
        for ki in range(len(K_LIST)):
            for ai in range(len(A_LIST)):
                v = data[ki, ai]
                if not np.isnan(v):
                    ax.text(ai, ki, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(v) < 0.6*vmax_m else "white")

        ax.set_xticks(range(len(A_LIST)))
        ax.set_xticklabels([f"{a:.2f}" for a in A_LIST], fontsize=8)
        ax.set_yticks(range(len(K_LIST)))
        ax.set_yticklabels([f"K={K}" for K in K_LIST], fontsize=8)
        ax.set_title(METHOD_LABELS[m], fontsize=10, color=METHOD_COLORS[m], fontweight="bold")
        if mi % 4 == 0:
            ax.set_ylabel("K", fontsize=9)
        if mi >= 4:
            ax.set_xlabel("α", fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)

    # Shade alpha=0 and alpha=1 columns to indicate non-realistic
    for ax in axes.ravel():
        for ai_bound in [0, 4]:  # index of alpha=0.00 and alpha=1.00
            ax.axvline(ai_bound - 0.5, color="gray", lw=0.5, ls="--")
            ax.axvline(ai_bound + 0.5, color="gray", lw=0.5, ls="--")

    fig.tight_layout()
    _save(fig, "RepR1_cost_of_representation")


# ======================================================================
# Fig RepR2: Method rank heatmaps within representative democracy (best config)
# ======================================================================
def fig_repr2_rank_heatmaps():
    # Use best_rep matrix for ranking
    rank_mat = np.full_like(best_rep, np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            vals = best_rep[ki, ai, :]
            if np.all(np.isnan(vals)):
                continue
            # Rank 1 = highest fitness
            order = np.argsort(-vals)
            ranks = np.empty(len(vals))
            ranks[order] = np.arange(1, len(vals) + 1)
            rank_mat[ki, ai, :] = ranks

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Method Rank Heatmaps — Representative Democracy (best config)\n"
                 "Rank 1 = highest terminal mean fitness",
                 fontsize=13, y=1.01)

    cmap = plt.cm.RdYlGn_r   # green=1, red=8

    for mi, m in enumerate(METHODS):
        ax = axes[mi // 4, mi % 4]
        data = rank_mat[:, :, mi]

        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap, vmin=1, vmax=len(METHODS),
                       interpolation="nearest")

        for ki in range(len(K_LIST)):
            for ai in range(len(A_LIST)):
                v = data[ki, ai]
                if not np.isnan(v):
                    ax.text(ai, ki, f"{int(v)}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if (v <= 2 or v >= 7) else "black")

        ax.set_xticks(range(len(A_LIST)))
        ax.set_xticklabels([f"{a:.2f}" for a in A_LIST], fontsize=8)
        ax.set_yticks(range(len(K_LIST)))
        ax.set_yticklabels([f"K={K}" for K in K_LIST], fontsize=8)
        ax.set_title(METHOD_LABELS[m], fontsize=10, color=METHOD_COLORS[m], fontweight="bold")
        if mi % 4 == 0:
            ax.set_ylabel("K", fontsize=9)
        if mi >= 4:
            ax.set_xlabel("α", fontsize=9)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=len(METHODS)))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Rank (1=best)")
    cbar_ax.invert_yaxis()

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    _save(fig, "RepR2_rep_rank_heatmaps")


# ======================================================================
# Fig RepR3: Candidate selection strategy comparison
# ======================================================================
def fig_repr3_config_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Candidate Selection Strategy: Configuration Comparison\n"
                 "Mean terminal fitness across all 8 voting methods",
                 fontsize=13)

    for ki, K in enumerate(K_LIST):
        ax = axes[ki]
        # For each config and alpha, compute mean across non-random-dictator methods
        methods_no_rd = [m for m in METHODS if m != "random_dictator"]
        mi_no_rd = [METHODS.index(m) for m in methods_no_rd]

        cfg_means = {}
        direct_means = []
        for ai, a in enumerate(A_LIST):
            d = np.nanmean(direct_mat[ki, ai, mi_no_rd])
            direct_means.append(d)

        for cfg in CONFIGS:
            vals = []
            for ai, a in enumerate(A_LIST):
                v = np.nanmean(rep_mats[cfg][ki, ai, mi_no_rd])
                vals.append(v)
            cfg_means[cfg] = vals

        x = A_LIST
        ax.plot(x, direct_means, "k-", lw=2.5, label="Direct democracy", zorder=10)
        for cfg in CONFIGS:
            ax.plot(x, cfg_means[cfg], "o-", color=CONFIG_COLORS[cfg],
                    label=CONFIG_LABELS[cfg], lw=1.5, ms=5)

        ax.axvspan(-0.05, 0.10, alpha=0.07, color="gray", label="α=0 (independent)")
        ax.axvspan(0.90, 1.05, alpha=0.07, color="gray")
        ax.set_xlabel("α (cross-dependency fraction)", fontsize=10)
        ax.set_ylabel("Mean terminal fitness", fontsize=10)
        ax.set_title(f"K = {K}", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if ki == 2:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, "RepR3_config_comparison")


# ======================================================================
# Fig RepR4: The Borda Paradox – rank of Borda in direct vs rep democracy
# ======================================================================
def fig_repr4_borda_paradox():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("The Borda Paradox: Rank in Direct vs. Representative Democracy\n"
                 "Borda excels at policy voting but struggles at candidate ranking",
                 fontsize=13)

    bi = METHODS.index("borda")

    # Compute ranks in direct democracy
    direct_ranks = np.full((len(K_LIST), len(A_LIST)), np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            vals = direct_mat[ki, ai, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                ranks = np.empty(len(vals))
                ranks[order] = np.arange(1, len(vals)+1)
                direct_ranks[ki, ai] = ranks[bi]

    # Compute ranks in best representative democracy
    rep_ranks = np.full((len(K_LIST), len(A_LIST)), np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            vals = best_rep[ki, ai, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                ranks = np.empty(len(vals))
                ranks[order] = np.arange(1, len(vals)+1)
                rep_ranks[ki, ai] = ranks[bi]

    for ki, K in enumerate(K_LIST):
        ax = axes[ki]
        ax.plot(A_LIST, direct_ranks[ki], "o-", color="#377eb8", lw=2.5, ms=8,
                label="Direct democracy", zorder=5)
        ax.plot(A_LIST, rep_ranks[ki], "s--", color="#e41a1c", lw=2.5, ms=8,
                label="Representative (best)", zorder=5)

        # Shade gap
        ax.fill_between(A_LIST, direct_ranks[ki], rep_ranks[ki],
                        where=rep_ranks[ki] > direct_ranks[ki],
                        alpha=0.2, color="red", label="Borda hurt by representation")
        ax.fill_between(A_LIST, direct_ranks[ki], rep_ranks[ki],
                        where=rep_ranks[ki] <= direct_ranks[ki],
                        alpha=0.2, color="green", label="Borda helped by representation")

        ax.axvspan(-0.05, 0.10, alpha=0.07, color="gray")
        ax.axvspan(0.90, 1.05, alpha=0.07, color="gray")
        ax.set_ylim(0.5, len(METHODS) + 0.5)
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("α (cross-dependency fraction)", fontsize=10)
        ax.set_ylabel("Borda's rank (1 = best)", fontsize=10)
        ax.set_title(f"K = {K}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        if ki == 0:
            ax.legend(fontsize=8, loc="lower center")

    fig.tight_layout()
    _save(fig, "RepR4_borda_paradox")


# ======================================================================
# Fig RepR5: Approval recovery – rank of Approval under representation
# ======================================================================
def fig_repr5_approval_recovery():
    """Approval is weak in direct democracy but recovers under representation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Approval Voting Recovery Under Representative Democracy\n"
                 "Approval ranks poorly in direct democracy but improves with candidate filtering",
                 fontsize=13)

    ai_idx = METHODS.index("approval")

    # Direct ranks
    direct_ranks = np.full((len(K_LIST), len(A_LIST)), np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            vals = direct_mat[ki, ai, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                ranks = np.empty(len(vals))
                ranks[order] = np.arange(1, len(vals)+1)
                direct_ranks[ki, ai] = ranks[ai_idx]

    # Best rep ranks
    rep_ranks = np.full((len(K_LIST), len(A_LIST)), np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            vals = best_rep[ki, ai, :]
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                ranks = np.empty(len(vals))
                ranks[order] = np.arange(1, len(vals)+1)
                rep_ranks[ki, ai] = ranks[ai_idx]

    for ki, K in enumerate(K_LIST):
        ax = axes[ki]
        ax.plot(A_LIST, direct_ranks[ki], "o-", color="#e41a1c", lw=2.5, ms=8,
                label="Direct democracy")
        ax.plot(A_LIST, rep_ranks[ki], "s--", color="#4daf4a", lw=2.5, ms=8,
                label="Representative (best)")

        ax.fill_between(A_LIST, direct_ranks[ki], rep_ranks[ki],
                        where=rep_ranks[ki] < direct_ranks[ki],
                        alpha=0.2, color="green", label="Approval improves")

        ax.axvspan(-0.05, 0.10, alpha=0.07, color="gray")
        ax.axvspan(0.90, 1.05, alpha=0.07, color="gray")
        ax.set_ylim(0.5, len(METHODS) + 0.5)
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("α (cross-dependency fraction)", fontsize=10)
        ax.set_ylabel("Approval's rank (1 = best)", fontsize=10)
        ax.set_title(f"K = {K}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        if ki == 0:
            ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "RepR5_approval_recovery")


# ======================================================================
# Fig RepR6: Random dictator filter effect
# ======================================================================
def fig_repr6_random_dictator_boost():
    """
    Random dictator benefits most from the representative layer because
    selecting among a few good candidates before random selection is
    fundamentally better than unconstrained random dictator.
    """
    rd_idx = METHODS.index("random_dictator")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("The Candidate Filter Effect on Random Dictator\n"
                 "Pre-selecting candidates dramatically improves random selection outcomes",
                 fontsize=13)

    for ki, K in enumerate(K_LIST):
        ax = axes[ki]
        d_vals = direct_mat[ki, :, rd_idx]
        ax.plot(A_LIST, d_vals, "k-o", lw=2.5, ms=8, label="Direct (rand. dictator)", zorder=10)

        for cfg in CONFIGS:
            r_vals = rep_mats[cfg][ki, :, rd_idx]
            ax.plot(A_LIST, r_vals, "s--", color=CONFIG_COLORS[cfg],
                    label=CONFIG_LABELS[cfg], lw=1.5, ms=5)

        ax.axvspan(-0.05, 0.10, alpha=0.07, color="gray")
        ax.axvspan(0.90, 1.05, alpha=0.07, color="gray")
        ax.set_xlabel("α (cross-dependency fraction)", fontsize=10)
        ax.set_ylabel("Terminal mean fitness", fontsize=10)
        ax.set_title(f"K = {K}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        if ki == 2:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    _save(fig, "RepR6_random_dictator_filter")


# ======================================================================
# Fig RepR7: Head-to-head: best direct vs best representative
#            (winner map at each (K, alpha))
# ======================================================================
def fig_repr7_direct_vs_rep():
    """
    At each (K, alpha), compare: best direct method vs best rep (method × config).
    Show whether representative democracy can match direct democracy
    when both are optimally tuned.
    """
    # Best direct: max over methods
    best_direct = np.nanmax(direct_mat, axis=2)    # (K, A)
    best_direct_mi = np.nanargmax(direct_mat, axis=2)

    # Best rep overall: max over methods of best_rep (already max over configs)
    best_rep_overall = np.nanmax(best_rep, axis=2)  # (K, A)
    best_rep_mi = np.nanargmax(best_rep, axis=2)

    # Delta: best_rep_overall - best_direct
    delta = best_rep_overall - best_direct

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Optimal Tuning: Best Representative vs. Best Direct Democracy\n"
                 "Δ fitness (best rep – best direct); positive = rep wins",
                 fontsize=13)

    # Panel 1: delta heatmap
    ax = axes[0]
    vmax = np.nanpercentile(np.abs(delta[: , 1:4]), 98)
    im = ax.imshow(delta, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            v = delta[ki, ai]
            if not np.isnan(v):
                ax.text(ai, ki, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.6*vmax else "white")
    ax.set_xticks(range(len(A_LIST)))
    ax.set_xticklabels([f"{a:.2f}" for a in A_LIST], fontsize=9)
    ax.set_yticks(range(len(K_LIST)))
    ax.set_yticklabels([f"K={K}" for K in K_LIST], fontsize=9)
    ax.set_title("Δ = best_rep − best_direct", fontsize=10)
    ax.set_xlabel("α", fontsize=10)
    ax.set_ylabel("K", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Panel 2: which direct method wins most
    ax = axes[1]
    cmap_m = plt.cm.get_cmap("tab10", len(METHODS))
    img_direct = np.full((len(K_LIST), len(A_LIST), 4), np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            mi = best_direct_mi[ki, ai]
            c = mcolors.to_rgba(METHOD_COLORS[METHODS[mi]])
            img_direct[ki, ai] = c
    ax.imshow(img_direct, aspect="auto", origin="lower", interpolation="nearest")
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            mi = best_direct_mi[ki, ai]
            ax.text(ai, ki, METHOD_LABELS[METHODS[mi]][:3], ha="center", va="center",
                    fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(A_LIST)))
    ax.set_xticklabels([f"{a:.2f}" for a in A_LIST], fontsize=9)
    ax.set_yticks(range(len(K_LIST)))
    ax.set_yticklabels([f"K={K}" for K in K_LIST], fontsize=9)
    ax.set_title("Best direct democracy method", fontsize=10)
    ax.set_xlabel("α", fontsize=10)
    patches = [Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHODS
               if m != "random_dictator"]
    ax.legend(handles=patches, fontsize=6, loc="upper right", framealpha=0.85,
              ncol=2, borderpad=0.5)

    # Panel 3: which rep method wins most
    ax = axes[2]
    img_rep = np.full((len(K_LIST), len(A_LIST), 4), np.nan)
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            mi = best_rep_mi[ki, ai]
            c = mcolors.to_rgba(METHOD_COLORS[METHODS[mi]])
            img_rep[ki, ai] = c
    ax.imshow(img_rep, aspect="auto", origin="lower", interpolation="nearest")
    for ki in range(len(K_LIST)):
        for ai in range(len(A_LIST)):
            mi = best_rep_mi[ki, ai]
            ax.text(ai, ki, METHOD_LABELS[METHODS[mi]][:3], ha="center", va="center",
                    fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(A_LIST)))
    ax.set_xticklabels([f"{a:.2f}" for a in A_LIST], fontsize=9)
    ax.set_yticks(range(len(K_LIST)))
    ax.set_yticklabels([f"K={K}" for K in K_LIST], fontsize=9)
    ax.set_title("Best representative democracy method", fontsize=10)
    ax.set_xlabel("α", fontsize=10)
    ax.legend(handles=patches, fontsize=6, loc="upper right", framealpha=0.85,
              ncol=2, borderpad=0.5)

    fig.tight_layout()
    _save(fig, "RepR7_direct_vs_rep_optimal")


# ======================================================================
# Fig RepR8: Summary panel — all methods, direct vs rep across K
#            Focus on realistic α ∈ {0.25, 0.50, 0.75}
# ======================================================================
def fig_repr8_summary_panel():
    """
    3×3 panel: rows = realistic α (0.25, 0.50, 0.75), cols = K (5, 10, 15).
    Each cell: bar chart showing terminal fitness for each method,
    direct (solid) vs best representative (hatched).
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey="row")
    fig.suptitle("Direct vs. Representative Democracy — All Methods\n"
                 "Realistic α values, mean terminal fitness ± SE",
                 fontsize=14, y=1.01)

    ai_map = {a: i for i, a in enumerate(A_LIST)}
    x = np.arange(len(METHODS))
    w = 0.38

    for row, a in enumerate(A_REAL):
        ai = ai_map[a]
        for col, K in enumerate(K_LIST):
            ki = K_LIST.index(K)
            ax = axes[row, col]

            d_vals = direct_mat[ki, ai, :]
            r_vals = best_rep[ki, ai, :]

            bars_d = ax.bar(x - w/2, d_vals, w, color=[METHOD_COLORS[m] for m in METHODS],
                            edgecolor="white", linewidth=0.5, label="Direct")
            bars_r = ax.bar(x + w/2, r_vals, w, color=[METHOD_COLORS[m] for m in METHODS],
                            alpha=0.65, edgecolor="black", linewidth=0.8,
                            hatch="///", label="Representative (best)")

            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[m][:4] for m in METHODS],
                               rotation=40, ha="right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_title(f"K={K},  α={a:.2f}", fontsize=10)

            if col == 0:
                ax.set_ylabel("Terminal mean fitness", fontsize=9)

    # Legend
    from matplotlib.patches import Patch as MPatch
    handles = [
        MPatch(facecolor="gray", edgecolor="white", label="Direct democracy"),
        MPatch(facecolor="gray", edgecolor="black", hatch="///",
               alpha=0.65, label="Representative (best config)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               fontsize=10, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "RepR8_summary_bars")


# ======================================================================
# Fig RepR9: Selection temperature effect (fixed candidate count c5)
# ======================================================================
def fig_repr9_temperature_effect():
    """
    Show how selection temperature (random, τ=1.0, τ=0.1) affects outcomes
    for 5-candidate rep democracy, averaged over all realistic methods.
    """
    methods_core = [m for m in METHODS if m != "random_dictator"]
    mi_core = [METHODS.index(m) for m in methods_core]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Selection Temperature Effect in Representative Democracy (5 candidates)\n"
                 "Mean terminal fitness across 7 methods (excluding random dictator)",
                 fontsize=12)

    temps = ["c5_tuniform", "c5_t1.0", "c5_t0.1"]
    temp_labels = {"c5_tuniform": "Random (τ=∞)",
                   "c5_t1.0": "Mild bias (τ=1.0)",
                   "c5_t0.1": "Strong bias (τ=0.1)"}
    temp_colors = {"c5_tuniform": "#d62728", "c5_t1.0": "#2ca02c", "c5_t0.1": "#1f77b4"}

    for ki, K in enumerate(K_LIST):
        ax = axes[ki]
        d_means = [np.nanmean(direct_mat[ki, ai, mi_core]) for ai, _ in enumerate(A_LIST)]
        ax.plot(A_LIST, d_means, "k-", lw=2.5, label="Direct democracy", zorder=10)

        for cfg in temps:
            vals = [np.nanmean(rep_mats[cfg][ki, ai, mi_core]) for ai in range(len(A_LIST))]
            ax.plot(A_LIST, vals, "o--", color=temp_colors[cfg],
                    label=temp_labels[cfg], lw=1.8, ms=6)

        ax.axvspan(-0.05, 0.10, alpha=0.07, color="gray")
        ax.axvspan(0.90, 1.05, alpha=0.07, color="gray")
        ax.set_xlabel("α", fontsize=10)
        ax.set_ylabel("Mean terminal fitness", fontsize=10)
        ax.set_title(f"K = {K}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        if ki == 0:
            ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "RepR9_temperature_effect")


# ======================================================================
# Fig RepR10: Rank crossover — Borda and Approval swap in direct vs rep
#             Most concise single figure capturing the main finding
# ======================================================================
def fig_repr10_rank_crossover():
    """
    2×3 panel: rows = Borda, Approval; cols = K (5, 10, 15).
    Each cell: rank in direct (solid) vs best rep (dashed) over α.
    Compact, paper-ready version of the key finding.
    """
    bi = METHODS.index("borda")
    ai_idx = METHODS.index("approval")

    direct_ranks_borda    = np.full((len(K_LIST), len(A_LIST)), np.nan)
    direct_ranks_approval = np.full((len(K_LIST), len(A_LIST)), np.nan)
    rep_ranks_borda       = np.full((len(K_LIST), len(A_LIST)), np.nan)
    rep_ranks_approval    = np.full((len(K_LIST), len(A_LIST)), np.nan)

    for ki in range(len(K_LIST)):
        for axi in range(len(A_LIST)):
            d = direct_mat[ki, axi, :]
            r = best_rep[ki, axi, :]
            for rk_arr, mat, method_idx in [
                (direct_ranks_borda, d, bi),
                (direct_ranks_approval, d, ai_idx),
                (rep_ranks_borda, r, bi),
                (rep_ranks_approval, r, ai_idx),
            ]:
                if not np.all(np.isnan(mat)):
                    order = np.argsort(-mat)
                    ranks = np.empty(len(mat))
                    ranks[order] = np.arange(1, len(mat)+1)
                    if ki == ki and axi == axi:
                        pass
                    # Only write for matching ki/axi
            # Redo cleanly
            if not np.all(np.isnan(d)):
                order = np.argsort(-d)
                ranks_d = np.empty(len(d))
                ranks_d[order] = np.arange(1, len(d)+1)
                direct_ranks_borda[ki, axi] = ranks_d[bi]
                direct_ranks_approval[ki, axi] = ranks_d[ai_idx]
            if not np.all(np.isnan(r)):
                order = np.argsort(-r)
                ranks_r = np.empty(len(r))
                ranks_r[order] = np.arange(1, len(r)+1)
                rep_ranks_borda[ki, axi] = ranks_r[bi]
                rep_ranks_approval[ki, axi] = ranks_r[ai_idx]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    fig.suptitle("Method Rank Reversal: Direct vs. Representative Democracy\n"
                 "Borda (top) and Approval (bottom) swap positions across democracy type",
                 fontsize=13)

    row_data = [
        ("Borda",    direct_ranks_borda,    rep_ranks_borda,    "#377eb8"),
        ("Approval", direct_ranks_approval, rep_ranks_approval, "#e41a1c"),
    ]

    for row, (name, d_ranks, r_ranks, color) in enumerate(row_data):
        for col, K in enumerate(K_LIST):
            ax = axes[row, col]
            ki = K_LIST.index(K)

            ax.plot(A_LIST, d_ranks[ki], "o-", color=color, lw=2.5, ms=8,
                    label=f"{name} – Direct")
            ax.plot(A_LIST, r_ranks[ki], "s--", color=color, lw=2.5, ms=8,
                    alpha=0.6, label=f"{name} – Representative")

            # Shade where rep is worse vs better
            ax.fill_between(A_LIST, d_ranks[ki], r_ranks[ki],
                            where=(r_ranks[ki] > d_ranks[ki]),
                            alpha=0.15, color="red")
            ax.fill_between(A_LIST, d_ranks[ki], r_ranks[ki],
                            where=(r_ranks[ki] <= d_ranks[ki]),
                            alpha=0.15, color="green")

            ax.axvspan(-0.05, 0.10, alpha=0.06, color="gray")
            ax.axvspan(0.90, 1.05, alpha=0.06, color="gray")
            ax.set_ylim(0.5, len(METHODS) + 0.5)
            ax.invert_yaxis()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)

            if row == 0:
                ax.set_title(f"K = {K}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{name}\nRank (1=best)", fontsize=10, color=color)
            if row == 1:
                ax.set_xlabel("α (cross-dependency fraction)", fontsize=10)

            if row == 0 and col == 0:
                from matplotlib.lines import Line2D
                handles = [
                    Line2D([0], [0], color="gray", lw=2.5, label="Direct democracy"),
                    Line2D([0], [0], color="gray", lw=2.5, ls="--", alpha=0.6,
                           label="Representative (best)"),
                ]
                ax.legend(handles=handles, fontsize=8, loc="upper left")

    fig.tight_layout()
    _save(fig, "RepR10_rank_crossover")


# ======================================================================
# Fig RepR11: Learning curves — direct vs representative for key (K, α) cases
# ======================================================================
def fig_repr11_learning_curves():
    """
    Learning curves (mean fitness over iterations) for 3 key cases:
    K=5 α=0.25, K=10 α=0.50, K=15 α=0.75 — comparing direct vs rep
    for Borda, Plurality, and Approval.
    Show the trajectory, not just the terminal value.
    """
    CASES = [(5, 0.25), (10, 0.50), (15, 0.75)]
    FOCUS_METHODS = ["borda", "plurality", "approval"]
    FOCUS_COLORS  = {"borda": "#377eb8", "plurality": "#ff7f00", "approval": "#e41a1c"}
    REP_CFG = "c5_t1.0"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Learning Curves: Direct vs. Representative Democracy\n"
                 "Mean fitness over iterations (shaded = ±1 SE across 50 runs)",
                 fontsize=13)

    for ci, (K, a) in enumerate(CASES):
        ax = axes[ci]

        for m in FOCUS_METHODS:
            color = FOCUS_COLORS[m]
            # Direct
            try:
                h_d = np.load(
                    f"{BASE_D}/K{K}_a{a:.2f}/{m}/mean_history.npy"
                )  # (runs, iters)
                mean_d = h_d.mean(axis=0)
                se_d   = h_d.std(axis=0) / np.sqrt(h_d.shape[0])
                t = np.arange(len(mean_d))
                ax.plot(t, mean_d, "-", color=color, lw=2, label=f"{METHOD_LABELS[m]} (direct)")
                ax.fill_between(t, mean_d - se_d, mean_d + se_d, alpha=0.15, color=color)
            except Exception:
                pass

            # Representative
            try:
                h_r = np.load(
                    f"{BASE_R}/K{K}_a{a:.2f}/{REP_CFG}/{m}/mean_history.npy"
                )
                mean_r = h_r.mean(axis=0)
                se_r   = h_r.std(axis=0) / np.sqrt(h_r.shape[0])
                t = np.arange(len(mean_r))
                ax.plot(t, mean_r, "--", color=color, lw=2, alpha=0.7,
                        label=f"{METHOD_LABELS[m]} (rep)")
                ax.fill_between(t, mean_r - se_r, mean_r + se_r, alpha=0.08, color=color)
            except Exception:
                pass

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Mean population fitness", fontsize=10)
        ax.set_title(f"K={K},  α={a:.2f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if ci == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, "RepR11_learning_curves")


# ======================================================================
# Run all figures
# ======================================================================
if __name__ == "__main__":
    print("Generating representative democracy figures...")
    fig_repr1_cost_heatmaps()
    fig_repr2_rank_heatmaps()
    fig_repr3_config_comparison()
    fig_repr4_borda_paradox()
    fig_repr5_approval_recovery()
    fig_repr6_random_dictator_boost()
    fig_repr7_direct_vs_rep()
    fig_repr8_summary_panel()
    fig_repr9_temperature_effect()
    fig_repr10_rank_crossover()
    fig_repr11_learning_curves()
    print("All representative democracy figures complete.")
