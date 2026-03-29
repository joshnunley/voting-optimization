#!/usr/bin/env python3
"""
Phase diagram analysis: visualize voting method performance across (K, alpha) space.

Figures produced:
  Phase1 - Winner map: which method ranks #1 at each (K, alpha)
  Phase2 - Approval deviation: Approval relative to mean, normalized per config
  Phase3 - All-method terminal fitness heatmaps (3×3 grid)
  Phase4 - Method rank heatmaps (all 8 methods, 2×4 grid)
  Phase5 - Normalized gain: (method - random_dictator) / (best - random_dictator)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import json

BASE   = "results/direct"
FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]
METHOD_LABELS = {
    "approval": "Approval",
    "borda": "Borda",
    "irv": "IRV",
    "minimax": "Minimax",
    "plurality": "Plurality",
    "random_dictator": "Rand. Dict.",
    "star": "STAR",
    "total_score": "Total Score",
}

# Colors for method winner maps
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

# All K and alpha values in data
K_COARSE   = [1, 5, 10, 15, 20]
A_COARSE   = [0.00, 0.25, 0.50, 0.75, 1.00]
K_DENSE    = [5, 10, 15, 20]
A_DENSE    = [0.00, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 1.00]


def _save(fig, name, dpi=150):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_terminal(K, alpha, method, n_terminal=10):
    """Return mean of last n_terminal timesteps averaged over runs."""
    d = os.path.join(BASE, f"K{K}_a{alpha:.2f}", method)
    p = os.path.join(d, "mean_history.npy")
    if not os.path.exists(p):
        return np.nan
    mh = np.load(p)          # (runs, iterations)
    return float(np.mean(mh[:, -n_terminal:]))


def build_matrix(K_list, alpha_list, method):
    """Build 2D array [K_idx, alpha_idx] of terminal fitness."""
    mat = np.full((len(K_list), len(alpha_list)), np.nan)
    for i, K in enumerate(K_list):
        for j, a in enumerate(alpha_list):
            mat[i, j] = load_terminal(K, a, method)
    return mat


def build_all_methods(K_list, alpha_list):
    """Return dict method -> (nK, nA) matrix."""
    return {m: build_matrix(K_list, alpha_list, m) for m in METHODS}


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase1: Winner map — which method is #1 at each (K, alpha)
# ─────────────────────────────────────────────────────────────────────────────
def fig_winner_map():
    print("Phase1: winner map (dense grid) ...")
    mats = build_all_methods(K_DENSE, A_DENSE)

    nK, nA = len(K_DENSE), len(A_DENSE)
    winner_idx = np.full((nK, nA), -1, dtype=int)
    winner_val = np.full((nK, nA), np.nan)

    for i in range(nK):
        for j in range(nA):
            vals = np.array([mats[m][i, j] for m in METHODS])
            if not np.all(np.isnan(vals)):
                best = np.nanargmax(vals)
                winner_idx[i, j] = best
                winner_val[i, j] = vals[best]

    # Build RGB image
    color_arr = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            if winner_idx[i, j] >= 0:
                m = METHODS[winner_idx[i, j]]
                c = mcolors.to_rgb(METHOD_COLORS[m])
                color_arr[i, j] = c

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(color_arr, aspect="auto", origin="lower",
              extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])

    ax.set_xticks(range(nA))
    ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(nK))
    ax.set_yticklabels([f"K={K}" for K in K_DENSE])
    ax.set_xlabel("α (cross-dependency fraction)")
    ax.set_ylabel("K (epistatic degree)")
    ax.set_title("Best-performing voting method by (K, α)")

    # Annotate with rank-1 method abbreviation
    abbrev = {
        "approval": "App", "borda": "Bor", "irv": "IRV",
        "minimax": "Min", "plurality": "Plu", "random_dictator": "RD",
        "star": "STA", "total_score": "TS",
    }
    for i in range(nK):
        for j in range(nA):
            if winner_idx[i, j] >= 0:
                m = METHODS[winner_idx[i, j]]
                ax.text(j, i, abbrev[m], ha="center", va="center",
                        fontsize=7, color="white",
                        fontweight="bold")

    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in METHODS if m != "random_dictator"]
    handles.append(Patch(facecolor=METHOD_COLORS["random_dictator"],
                         label=METHOD_LABELS["random_dictator"]))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "phase1_winner_map.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase2: Approval deviation from mean
# ─────────────────────────────────────────────────────────────────────────────
def fig_approval_deviation():
    print("Phase2: Approval deviation ...")
    mats = build_all_methods(K_DENSE, A_DENSE)

    nK, nA = len(K_DENSE), len(A_DENSE)
    # Compute mean across all methods at each (K, alpha)
    all_stack = np.stack([mats[m] for m in METHODS], axis=0)  # (8, nK, nA)
    ensemble_mean = np.nanmean(all_stack, axis=0)
    ensemble_std  = np.nanstd(all_stack, axis=0)

    approval_mat = mats["approval"]
    # Normalized deviation: z-score of approval relative to ensemble
    deviation = (approval_mat - ensemble_mean) / (ensemble_std + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: raw deviation
    im0 = axes[0].imshow(approval_mat - ensemble_mean, aspect="auto",
                         origin="lower", cmap="RdBu_r",
                         extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
    plt.colorbar(im0, ax=axes[0], label="Approval − ensemble mean")
    axes[0].set_title("Approval raw deviation from ensemble mean")

    # Right: z-score deviation
    vmax = np.nanmax(np.abs(deviation))
    im1 = axes[1].imshow(deviation, aspect="auto", origin="lower",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
    plt.colorbar(im1, ax=axes[1], label="z-score (Approval vs ensemble)")
    axes[1].set_title("Approval z-score deviation")

    for ax in axes:
        ax.set_xticks(range(nA))
        ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"K={K}" for K in K_DENSE])
        ax.set_xlabel("α")
        ax.set_ylabel("K")

    # Contour at zero
    for ax, data in [(axes[0], approval_mat - ensemble_mean), (axes[1], deviation)]:
        if not np.all(np.isnan(data)):
            try:
                ax.contour(data, levels=[0], colors="black", linewidths=1.5,
                           extent=[-0.5, nA - 0.5, -0.5, nK - 0.5], origin="lower")
            except Exception:
                pass

    fig.tight_layout()
    _save(fig, "phase2_approval_deviation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase3: All-method terminal fitness heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def fig_all_method_heatmaps():
    print("Phase3: all-method heatmaps ...")
    mats = build_all_methods(K_DENSE, A_DENSE)
    nK, nA = len(K_DENSE), len(A_DENSE)

    # Compute global vmin/vmax (excluding random_dictator for scale)
    non_rd = [mats[m] for m in METHODS if m != "random_dictator"]
    all_vals = np.concatenate([m.ravel() for m in non_rd])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    for ax, method in zip(axes, METHODS):
        im = ax.imshow(mats[method], aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax,
                       extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xticks(range(nA))
        ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"K={K}" for K in K_DENSE], fontsize=8)
        ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel("K", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("Terminal mean fitness across (K, α) — direct democracy", fontsize=13)
    fig.tight_layout()
    _save(fig, "phase3_all_method_heatmaps.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase4: Method rank heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def fig_rank_heatmaps():
    print("Phase4: rank heatmaps ...")
    mats = build_all_methods(K_DENSE, A_DENSE)
    nK, nA = len(K_DENSE), len(A_DENSE)

    # For each (K, alpha), rank all methods (rank 1 = best)
    rank_mats = {}
    for m in METHODS:
        rank_mats[m] = np.full((nK, nA), np.nan)

    for i in range(nK):
        for j in range(nA):
            vals = np.array([mats[m][i, j] for m in METHODS])
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)  # descending
                for rank, idx in enumerate(order):
                    rank_mats[METHODS[idx]][i, j] = rank + 1

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    cmap = plt.cm.RdYlGn_r  # green=rank1, red=rank8

    for ax, method in zip(axes, METHODS):
        im = ax.imshow(rank_mats[method], aspect="auto", origin="lower",
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xticks(range(nA))
        ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"K={K}" for K in K_DENSE], fontsize=8)
        ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel("K", fontsize=8)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
        cb.set_ticklabels(["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"])
        cb.ax.tick_params(labelsize=7)

        # Annotate rank number
        for ii in range(nK):
            for jj in range(nA):
                r = rank_mats[method][ii, jj]
                if not np.isnan(r):
                    ax.text(jj, ii, str(int(r)), ha="center", va="center",
                            fontsize=6, color="black")

    fig.suptitle("Method rank (1=best, 8=worst) by (K, α) — direct democracy", fontsize=13)
    fig.tight_layout()
    _save(fig, "phase4_rank_heatmaps.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase5: Normalized gain relative to random dictator
# ─────────────────────────────────────────────────────────────────────────────
def fig_normalized_gain():
    print("Phase5: normalized gain ...")
    mats = build_all_methods(K_DENSE, A_DENSE)
    nK, nA = len(K_DENSE), len(A_DENSE)

    rd = mats["random_dictator"]
    # Best non-RD method at each cell
    non_rd_stack = np.stack([mats[m] for m in METHODS if m != "random_dictator"], axis=0)
    best_non_rd = np.nanmax(non_rd_stack, axis=0)

    # Normalized gain for each method: (method - RD) / (best - RD)
    # Value of 1.0 means this method IS the best; value < 0 means worse than RD
    norm_mats = {}
    for m in METHODS:
        if m == "random_dictator":
            norm_mats[m] = np.zeros((nK, nA))
        else:
            denom = best_non_rd - rd
            with np.errstate(invalid="ignore", divide="ignore"):
                norm_mats[m] = np.where(np.abs(denom) > 1e-10,
                                        (mats[m] - rd) / denom, np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    for ax, method in zip(axes, METHODS):
        if method == "random_dictator":
            data = np.zeros_like(norm_mats[method])
        else:
            data = norm_mats[method]
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap="RdYlGn", vmin=0, vmax=1.0,
                       extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xticks(range(nA))
        ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"K={K}" for K in K_DENSE], fontsize=8)
        ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel("K", fontsize=8)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("Normalized gain", fontsize=7)

    fig.suptitle("Normalized gain: (method − Rand.Dict.) / (best − Rand.Dict.)", fontsize=13)
    fig.tight_layout()
    _save(fig, "phase5_normalized_gain.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase6: Approval rank vs alpha for each K (line plot)
#   Shows the U-shape clearly
# ─────────────────────────────────────────────────────────────────────────────
def fig_approval_rank_lines():
    print("Phase6: Approval rank vs alpha lines ...")
    mats = build_all_methods(K_DENSE, A_DENSE)
    nK, nA = len(K_DENSE), len(A_DENSE)

    # Compute rank of approval at each (K, alpha)
    approval_rank = np.full((nK, nA), np.nan)
    for i in range(nK):
        for j in range(nA):
            vals = np.array([mats[m][i, j] for m in METHODS])
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                for rank, idx in enumerate(order):
                    if METHODS[idx] == "approval":
                        approval_rank[i, j] = rank + 1
                        break

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, nK))
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (K, c) in enumerate(zip(K_DENSE, colors)):
        ranks = approval_rank[i, :]
        # Mark available points
        valid = ~np.isnan(ranks)
        ax.plot(np.array(A_DENSE)[valid], ranks[valid], "o-",
                color=c, label=f"K={K}", linewidth=2, markersize=6)

    ax.axhline(1, color="gray", linestyle="--", alpha=0.4, label="Rank 1 (best)")
    ax.axhline(8, color="gray", linestyle=":", alpha=0.4, label="Rank 8 (worst)")
    ax.set_xlabel("α (cross-dependency fraction)", fontsize=12)
    ax.set_ylabel("Approval voting rank", fontsize=12)
    ax.set_title("Approval voting rank across α — phase boundary", fontsize=13)
    ax.set_ylim(0.5, 8.5)
    ax.invert_yaxis()
    ax.set_xticks(A_DENSE)
    ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right")
    ax.legend(fontsize=10, loc="center right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "phase6_approval_rank_lines.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase7: Terminal fitness cross-sections at fixed K and fixed alpha
# ─────────────────────────────────────────────────────────────────────────────
def fig_cross_sections():
    print("Phase7: cross-section line plots ...")
    mats = build_all_methods(K_DENSE, A_DENSE)
    nK, nA = len(K_DENSE), len(A_DENSE)

    method_colors = {m: c for m, c in zip(METHODS, plt.cm.tab10(np.linspace(0, 1, 8)))}
    method_ls     = {m: ls for m, ls in zip(METHODS,
                     ["-", "-", "-", "-", "--", ":", "--", "-"])}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: fixed K, vary alpha
    for col, (Ki, K) in enumerate([(1, 10), (3, 20)]):
        ax = axes[0, col]
        for m in METHODS:
            vals = mats[m][Ki, :]
            valid = ~np.isnan(vals)
            ax.plot(np.array(A_DENSE)[valid], vals[valid],
                    color=method_colors[m], ls=method_ls[m],
                    label=METHOD_LABELS[m], linewidth=2, marker="o", markersize=4)
        ax.set_title(f"Terminal fitness vs α   (K={K})", fontsize=11)
        ax.set_xlabel("α")
        ax.set_ylabel("Terminal mean fitness")
        ax.set_xticks(A_DENSE)
        ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Bottom row: fixed alpha, vary K
    for col, (ai, a) in enumerate([(2, 0.30), (5, 0.50)]):
        ax = axes[1, col]
        for m in METHODS:
            vals = [mats[m][ki, ai] for ki in range(nK)]
            valid_ki = [ki for ki in range(nK) if not np.isnan(vals[ki])]
            if valid_ki:
                ax.plot([K_DENSE[ki] for ki in valid_ki],
                        [vals[ki] for ki in valid_ki],
                        color=method_colors[m], ls=method_ls[m],
                        label=METHOD_LABELS[m], linewidth=2, marker="o", markersize=4)
        ax.set_title(f"Terminal fitness vs K   (α={a:.2f})", fontsize=11)
        ax.set_xlabel("K")
        ax.set_ylabel("Terminal mean fitness")
        ax.set_xticks(K_DENSE)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "phase7_cross_sections.pdf")


def _load_terminal_stat(K, alpha, method, stat="variance", n=10):
    """Load terminal value (mean of last n steps) for a given statistic."""
    fname = {"mean": "mean_history", "variance": "variance_history",
             "median": None}[stat]
    if fname is not None:
        p = os.path.join(BASE, f"K{K}_a{alpha:.2f}", method, f"{fname}.npy")
        if not os.path.exists(p):
            return np.nan
        return float(np.mean(np.load(p)[:, -n:]))
    else:
        # Median: load fitness distributions, take median per run at last step
        p = os.path.join(BASE, f"K{K}_a{alpha:.2f}", method, "fitness_distributions.npy")
        if not os.path.exists(p):
            return np.nan
        fd = np.load(p, allow_pickle=True)   # (runs, iters) object array
        medians = []
        for run in range(len(fd)):
            x = np.array(fd[run][-1], dtype=float)
            medians.append(np.median(x))
        return float(np.mean(medians))


def _build_stat_matrix(K_list, alpha_list, stat):
    """Build (nK, nA) matrix of terminal stat for each method."""
    return {m: np.array([[_load_terminal_stat(K, a, m, stat)
                          for a in alpha_list]
                         for K in K_list])
            for m in METHODS}


def _rank_heatmap_grid(mats, K_list, A_list, title, fname, higher_is_better=True):
    """Generic 2×4 rank heatmap grid."""
    nK, nA = len(K_list), len(A_list)
    rank_mats = {m: np.full((nK, nA), np.nan) for m in METHODS}

    for i in range(nK):
        for j in range(nA):
            vals = np.array([mats[m][i, j] for m in METHODS])
            if not np.all(np.isnan(vals)):
                if higher_is_better:
                    order = np.argsort(-vals)
                else:
                    order = np.argsort(vals)   # lower stat = rank 1
                for rank, idx in enumerate(order):
                    rank_mats[METHODS[idx]][i, j] = rank + 1

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    cmap = plt.cm.RdYlGn_r

    for ax, method in zip(axes, METHODS):
        im = ax.imshow(rank_mats[method], aspect="auto", origin="lower",
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xticks(range(nA))
        ax.set_xticklabels([f"{a:.2f}" for a in A_list],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"K={K}" for K in K_list], fontsize=8)
        ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel("K", fontsize=8)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        cb.ax.tick_params(labelsize=7)
        for ii in range(nK):
            for jj in range(nA):
                r = rank_mats[method][ii, jj]
                if not np.isnan(r):
                    ax.text(jj, ii, str(int(r)), ha="center", va="center",
                            fontsize=6, color="black")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase8: Rank heatmaps by terminal VARIANCE (rank 1 = most equitable)
# ─────────────────────────────────────────────────────────────────────────────
def fig_rank_heatmaps_variance():
    print("Phase8: rank heatmaps (variance) ...")
    mats = _build_stat_matrix(K_DENSE, A_DENSE, "variance")
    _rank_heatmap_grid(
        mats, K_DENSE, A_DENSE,
        title="Method rank by terminal variance (1=lowest variance/most equitable) — direct democracy",
        fname="phase8_rank_heatmaps_variance.pdf",
        higher_is_better=False,  # lower variance = rank 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase9: Rank heatmaps by terminal MEDIAN fitness (rank 1 = best median)
# ─────────────────────────────────────────────────────────────────────────────
def fig_rank_heatmaps_median():
    print("Phase9: rank heatmaps (median) — loading fitness distributions ...")
    mats = _build_stat_matrix(K_DENSE, A_DENSE, "median")
    _rank_heatmap_grid(
        mats, K_DENSE, A_DENSE,
        title="Method rank by terminal median fitness (1=best median) — direct democracy",
        fname="phase9_rank_heatmaps_median.pdf",
        higher_is_better=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure Phase10: Side-by-side mean rank vs variance rank for Borda & Plurality
# Directly shows the K transition story
# ─────────────────────────────────────────────────────────────────────────────
def fig_borda_plurality_story():
    print("Phase10: Borda vs Plurality rank story ...")
    mean_mats = build_all_methods(K_DENSE, A_DENSE)
    nK, nA = len(K_DENSE), len(A_DENSE)

    # Compute rank for each method at each cell (by mean fitness)
    focus = ["plurality", "borda"]
    rank_by_method = {m: np.full((nK, nA), np.nan) for m in focus}

    for i in range(nK):
        for j in range(nA):
            vals = np.array([mean_mats[m][i, j] for m in METHODS])
            if not np.all(np.isnan(vals)):
                order = np.argsort(-vals)
                for rank, idx in enumerate(order):
                    if METHODS[idx] in focus:
                        rank_by_method[METHODS[idx]][i, j] = rank + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.RdYlGn_r

    for ax, method in zip(axes, focus):
        im = ax.imshow(rank_by_method[method], aspect="auto", origin="lower",
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
        ax.set_title(f"{METHOD_LABELS[method]} rank", fontsize=13)
        ax.set_xticks(range(nA))
        ax.set_xticklabels([f"{a:.2f}" for a in A_DENSE],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"K={K}" for K in K_DENSE], fontsize=9)
        ax.set_xlabel("α (cross-dependency)", fontsize=10)
        ax.set_ylabel("K (landscape ruggedness)", fontsize=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.9)
        cb.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        for ii in range(nK):
            for jj in range(nA):
                r = rank_by_method[method][ii, jj]
                if not np.isnan(r):
                    ax.text(jj, ii, str(int(r)), ha="center", va="center",
                            fontsize=8, color="black", fontweight="bold")

    fig.suptitle("Plurality leads at low K, Borda dominates at high K\n"
                 "(Green=rank 1, Red=rank 8; across all realistic α values)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "phase10_borda_plurality_story.pdf")


if __name__ == "__main__":
    fig_winner_map()
    fig_approval_deviation()
    fig_all_method_heatmaps()
    fig_rank_heatmaps()
    fig_normalized_gain()
    fig_approval_rank_lines()
    fig_cross_sections()
    fig_rank_heatmaps_variance()
    fig_rank_heatmaps_median()
    fig_borda_plurality_story()
    print("All phase figures done.")
