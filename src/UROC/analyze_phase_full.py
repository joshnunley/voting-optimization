#!/usr/bin/env python3
"""
Phase heatmaps over the full K=1-20, alpha=0.00-1.00 direct democracy sweep.

Figures:
  PhaseF1 — Mean fitness RANK heatmap (2x4 grid, rank 1=best)
  PhaseF2 — Variance RANK heatmap (2x4 grid, rank 1=lowest variance)
  PhaseF3 — Mean fitness VALUE heatmap (2x4 grid, shared colorscale)
  PhaseF4 — Variance VALUE heatmap (2x4 grid, shared colorscale)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = "results/direct"
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

K_FULL = list(range(1, 21))                             # 1-20
A_FULL = [round(a * 0.05, 2) for a in range(21)]        # 0.00-1.00
N_TERMINAL = 10   # average last 10 timesteps


def _save(fig, name, dpi=200):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_terminal(K, alpha, method, stat="mean"):
    """Terminal value: mean of last N_TERMINAL timesteps, averaged across runs."""
    fname = {"mean": "mean_history", "variance": "variance_history"}[stat]
    p = os.path.join(BASE, f"K{K}_a{alpha:.2f}", method, f"{fname}.npy")
    if not os.path.exists(p):
        return np.nan
    arr = np.load(p)  # (runs, iterations)
    return float(np.mean(arr[:, -N_TERMINAL:]))


def build_matrix(stat="mean"):
    """Return dict method -> (nK, nA) array of terminal stat. Uses npz if available."""
    npz_path = "results/direct_results.npz"
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        method_list = list(d['methods'])
        data = d[stat]  # (nK, nA, nM)
        return {m: data[:, :, method_list.index(m)] for m in METHODS}

    mats = {}
    for m in METHODS:
        mat = np.full((len(K_FULL), len(A_FULL)), np.nan)
        for i, K in enumerate(K_FULL):
            for j, a in enumerate(A_FULL):
                mat[i, j] = load_terminal(K, a, m, stat=stat)
        mats[m] = mat
    return mats


def compute_ranks(mats, higher_is_better=True):
    """Rank methods 1-8 at each (K, alpha) cell."""
    nK, nA = len(K_FULL), len(A_FULL)
    rank_mats = {m: np.full((nK, nA), np.nan) for m in METHODS}
    for i in range(nK):
        for j in range(nA):
            vals = np.array([mats[m][i, j] for m in METHODS])
            if np.all(np.isnan(vals)):
                continue
            order = np.argsort(-vals) if higher_is_better else np.argsort(vals)
            for rank, idx in enumerate(order):
                rank_mats[METHODS[idx]][i, j] = rank + 1
    return rank_mats


def _heatmap_grid(data_mats, title, fname, cmap, vmin, vmax, cbar_label,
                  annotate_int=False):
    """Generic 2x4 heatmap grid over full K x alpha space."""
    nK, nA = len(K_FULL), len(A_FULL)
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()

    for ax, method in zip(axes, METHODS):
        im = ax.imshow(data_mats[method], aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[-0.5, nA - 0.5, -0.5, nK - 0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold")

        # Tick labels — show every other K to avoid crowding
        ax.set_xticks(range(0, nA, 2))
        ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(0, nK, 2))
        ax.set_yticklabels([f"{K_FULL[i]}" for i in range(0, nK, 2)], fontsize=8)
        ax.set_xlabel("α", fontsize=9)
        ax.set_ylabel("K", fontsize=9)

        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=7)

        if annotate_int:
            for ii in range(nK):
                for jj in range(nA):
                    r = data_mats[method][ii, jj]
                    if not np.isnan(r):
                        ax.text(jj, ii, str(int(r)), ha="center", va="center",
                                fontsize=5, color="black")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)


# ──────────────────────────────────────────────────────────────────────────
# PhaseF1: Mean fitness RANK
# ──────────────────────────────────────────────────────────────────────────
def fig_mean_rank():
    print("PhaseF1: mean fitness rank heatmap ...")
    mats = build_matrix("mean")
    ranks = compute_ranks(mats, higher_is_better=True)
    _heatmap_grid(
        ranks,
        title="Mean fitness rank (1=best) — direct democracy, K=1–20, α=0–1",
        fname="phaseF1_mean_rank.pdf",
        cmap=plt.cm.RdYlGn_r,
        vmin=1, vmax=8,
        cbar_label="Rank",
        annotate_int=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# PhaseF2: Variance RANK (rank 1 = lowest variance = most equitable)
# ──────────────────────────────────────────────────────────────────────────
def fig_variance_rank():
    print("PhaseF2: variance rank heatmap ...")
    mats = build_matrix("variance")
    ranks = compute_ranks(mats, higher_is_better=False)
    _heatmap_grid(
        ranks,
        title="Variance rank (1=lowest variance, most equitable) — direct democracy",
        fname="phaseF2_variance_rank.pdf",
        cmap=plt.cm.RdYlGn_r,
        vmin=1, vmax=8,
        cbar_label="Rank",
        annotate_int=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# PhaseF3: Mean fitness — normalized by per-cell max across methods
# ──────────────────────────────────────────────────────────────────────────
def fig_mean_value():
    print("PhaseF3: mean fitness (normalized by cell max) ...")
    mats = build_matrix("mean")
    nK, nA = len(K_FULL), len(A_FULL)

    # Per-cell max across all methods
    stack = np.stack([mats[m] for m in METHODS], axis=0)  # (8, nK, nA)
    cell_max = np.nanmax(stack, axis=0)                    # (nK, nA)

    # Divide each method's value by the cell max
    norm_mats = {}
    for m in METHODS:
        with np.errstate(invalid='ignore', divide='ignore'):
            norm_mats[m] = np.where(cell_max > 0, mats[m] / cell_max, np.nan)

    # Shared scale: max is 1, min is the global min ratio
    all_ratios = np.concatenate([norm_mats[m].ravel() for m in METHODS])
    vmin = np.nanmin(all_ratios)
    _heatmap_grid(
        norm_mats,
        title=f"Terminal mean fitness / cell max — direct democracy (scale [{vmin:.3f}, 1.000])",
        fname="phaseF3_mean_value.pdf",
        cmap="viridis",
        vmin=vmin, vmax=1.0,
        cbar_label="Fraction of best",
    )


# ──────────────────────────────────────────────────────────────────────────
# PhaseF4: Variance — normalized by per-cell max across methods
# ──────────────────────────────────────────────────────────────────────────
def fig_variance_value():
    print("PhaseF4: variance (normalized by cell max) ...")
    mats = build_matrix("variance")
    nK, nA = len(K_FULL), len(A_FULL)

    stack = np.stack([mats[m] for m in METHODS], axis=0)
    cell_max = np.nanmax(stack, axis=0)

    norm_mats = {}
    for m in METHODS:
        with np.errstate(invalid='ignore', divide='ignore'):
            norm_mats[m] = np.where(cell_max > 0, mats[m] / cell_max, np.nan)

    all_ratios = np.concatenate([norm_mats[m].ravel() for m in METHODS])
    vmin = np.nanmin(all_ratios)
    _heatmap_grid(
        norm_mats,
        title=f"Terminal variance / cell max — direct democracy (scale [{vmin:.3f}, 1.000])",
        fname="phaseF4_variance_value.pdf",
        cmap="inferno",
        vmin=vmin, vmax=1.0,
        cbar_label="Fraction of worst",
    )


if __name__ == "__main__":
    fig_mean_rank()
    fig_variance_rank()
    fig_mean_value()
    fig_variance_value()
    print("\nAll full-grid phase figures done.")
