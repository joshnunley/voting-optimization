#!/usr/bin/env python3
"""
Distributional analysis of fitness evolution within individual runs.

Key questions:
1. Do fitness distributions evolve beyond a simple mean shift?
2. Who benefits from voting — redistributive or Matthew effect?
3. Is the equity outcome stochastic (run-to-run variation)?
4. Does inequality peak early before settling?

Run from src/UROC/ with:
    python3 analyze_distributions.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import time

RESULTS = "results"
FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

VOTE_TYPES = [
    "plurality", "approval", "total_score", "borda",
    "irv", "star", "minimax", "random_dictator",
]
VOTE_LABELS = {
    "plurality":       "Plurality",
    "approval":        "Approval",
    "total_score":     "Total Score",
    "borda":           "Borda",
    "irv":             "IRV",
    "star":            "STAR",
    "minimax":         "Minimax",
    "random_dictator": "Random Dictator",
}
COLORS = {
    "plurality":       "#1f77b4",
    "approval":        "#ff7f0e",
    "total_score":     "#2ca02c",
    "borda":           "#d62728",
    "irv":             "#9467bd",
    "star":            "#8c564b",
    "minimax":         "#e377c2",
    "random_dictator": "#7f7f7f",
}
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
ITERATIONS = 300
RUNS = 50


def load_fd(K, alpha, vote_type):
    return np.load(f"{RESULTS}/direct/K{K}_a{alpha:.2f}/{vote_type}/fitness_distributions.npy",
                   allow_pickle=True)


def rank_gain_corr(fd, run):
    """Pearson r between initial rank and terminal gain for a single run."""
    t0   = np.sort(np.array(fd[run][0],   dtype=float))
    t299 = np.sort(np.array(fd[run][299], dtype=float))
    gains = t299 - t0
    if gains.std() < 1e-9 or len(gains) < 3:
        return np.nan
    rho, _ = stats.pearsonr(np.arange(len(gains)), gains)
    return rho


def iqr_trajectory(fd, run, T_list):
    """P90-P10 gap at each time in T_list for a single run."""
    gaps = []
    for t in T_list:
        x = np.array(fd[run][t], dtype=float)
        gaps.append(np.percentile(x, 90) - np.percentile(x, 10))
    return np.array(gaps)


def _save(fig, name):
    pdf = f"{FIG_DIR}/{name}.pdf"
    png = f"{FIG_DIR}/{name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=150)
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"  Saved {pdf}")
    plt.close(fig)


# ============================================================
# Fig 7: Rank-gain correlation heatmap
# Rows = methods, Columns = (K, alpha) configurations
# Color = mean r(rank, gain) over 50 runs
# Reveals: redistributive vs Matthew effect
# ============================================================

def fig_rank_gain_heatmap():
    K_list = [1, 5, 10, 15, 20]
    alpha_list = [0.25, 0.5, 0.75]   # realistic α; α=0 and α=1 are special cases
    configs = [(K, a) for K in K_list for a in alpha_list]
    n_configs = len(configs)

    # Build matrix: (n_methods, n_configs)
    mean_rho = np.full((len(VOTE_TYPES), n_configs), np.nan)
    std_rho  = np.full((len(VOTE_TYPES), n_configs), np.nan)

    for vi, vt in enumerate(VOTE_TYPES):
        for ci, (K, alpha) in enumerate(configs):
            fd = load_fd(K, alpha, vt)
            rhos = [rank_gain_corr(fd, r) for r in range(RUNS)]
            rhos = np.array([r for r in rhos if not np.isnan(r)])
            if len(rhos) > 0:
                mean_rho[vi, ci] = rhos.mean()
                std_rho[vi, ci]  = rhos.std()

    config_labels = [f"K={K}\nα={a}" for K, a in configs]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Panel A: mean rho heatmap
    ax = axes[0]
    im = ax.imshow(mean_rho, aspect='auto', origin='upper',
                   vmin=-0.8, vmax=0.8, cmap='RdBu')
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(config_labels, fontsize=6, rotation=0)
    ax.set_yticks(range(len(VOTE_TYPES)))
    ax.set_yticklabels([VOTE_LABELS[vt] for vt in VOTE_TYPES], fontsize=9)
    ax.set_title("Mean r(initial rank, gain)\nBlue=redistributive, Red=Matthew effect",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=9)

    # Draw vertical separators between K groups
    for sep in [2.5, 5.5, 8.5, 11.5]:
        ax.axvline(sep, color='white', lw=1.5, alpha=0.7)

    # Panel B: std of rho (how stochastic is the equity outcome?)
    ax = axes[1]
    im2 = ax.imshow(std_rho, aspect='auto', origin='upper',
                    vmin=0, vmax=0.6, cmap='YlOrRd')
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(config_labels, fontsize=6, rotation=0)
    ax.set_yticks(range(len(VOTE_TYPES)))
    ax.set_yticklabels([VOTE_LABELS[vt] for vt in VOTE_TYPES], fontsize=9)
    ax.set_title("Std of r (across 50 runs)\nHigh = equity outcome is stochastic",
                 fontsize=10)
    cbar2 = fig.colorbar(im2, ax=ax, shrink=0.8)
    cbar2.set_label("Std of Pearson r", fontsize=9)

    for sep in [2.5, 5.5, 8.5, 11.5]:
        ax.axvline(sep, color='white', lw=1.5, alpha=0.7)

    fig.suptitle("Equity Direction: Who Benefits from Voting?\n"
                 "(Rank-gain correlation: negative = poor benefit more, "
                 "positive = rich benefit more)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig7_rank_gain_heatmap")

    return mean_rho, std_rho, K_list, alpha_list, configs


# ============================================================
# Fig 8: Distribution snapshots for individual runs
# Violin plots at 5 time points, for contrasting configurations
# ============================================================

def fig_distribution_evolution():
    # Show contrasting cases across realistic α:
    # K=5 α=0.25 (low coupling) vs K=20 α=0.75 (high coupling)
    # For a representative single run (run 0) and a random-dictator vs borda pair

    snap_times = [0, 30, 100, 200, 299]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    configs = [
        (5,  0.25, "borda"),
        (5,  0.25, "random_dictator"),
        (20, 0.75, "borda"),
        (20, 0.75, "random_dictator"),
    ]

    for col, (K, alpha, vt) in enumerate(configs):
        fd = load_fd(K, alpha, vt)

        # Top row: single representative run (run 0)
        ax_top = axes[0, col]
        # Bottom row: mean over ALL 50 runs (violin per time)

        # Top: single run distribution as overlaid histograms
        colors_t = plt.cm.viridis(np.linspace(0.1, 0.9, len(snap_times)))
        all_vals = np.concatenate([np.array(fd[0][t], dtype=float) for t in snap_times])
        xlim = (all_vals.min() - 0.5, all_vals.max() + 0.5)
        for i, t in enumerate(snap_times):
            x = np.array(fd[0][t], dtype=float)
            ax_top.hist(x, bins=25, density=True, alpha=0.5,
                        color=colors_t[i], label=f"t={t}")
        ax_top.set_xlim(xlim)
        ax_top.set_title(f"K={K}, α={alpha}\n{VOTE_LABELS[vt]} (run 0)", fontsize=8)
        ax_top.set_xlabel("Fitness", fontsize=7)
        ax_top.set_ylabel("Density", fontsize=7)
        ax_top.tick_params(labelsize=6)
        if col == 0:
            ax_top.legend(fontsize=6, loc='upper left')

        # Bottom: quantile band plot (median ± IQR band) across 50 runs
        ax_bot = axes[1, col]
        T_dense = np.arange(0, 300, 5)
        p10 = np.zeros((RUNS, len(T_dense)))
        p50 = np.zeros((RUNS, len(T_dense)))
        p90 = np.zeros((RUNS, len(T_dense)))
        for r in range(RUNS):
            for ti, t in enumerate(T_dense):
                x = np.array(fd[r][t], dtype=float)
                p10[r, ti] = np.percentile(x, 10)
                p50[r, ti] = np.percentile(x, 50)
                p90[r, ti] = np.percentile(x, 90)

        m10 = p10.mean(axis=0)
        m50 = p50.mean(axis=0)
        m90 = p90.mean(axis=0)

        ax_bot.fill_between(T_dense, m10, m90, alpha=0.25,
                            color=COLORS[vt], label="P10–P90 band")
        ax_bot.plot(T_dense, m50, color=COLORS[vt], lw=2, label="Median (P50)")
        ax_bot.plot(T_dense, m10, color=COLORS[vt], lw=1, ls='--', alpha=0.7)
        ax_bot.plot(T_dense, m90, color=COLORS[vt], lw=1, ls='--', alpha=0.7)
        ax_bot.set_xlabel("Iteration", fontsize=7)
        ax_bot.set_ylabel("Fitness", fontsize=7)
        ax_bot.set_title(f"P10/P50/P90 over time (avg 50 runs)", fontsize=8)
        ax_bot.tick_params(labelsize=6)
        ax_bot.grid(True, lw=0.3, alpha=0.4)
        if col == 0:
            ax_bot.legend(fontsize=6)

    fig.suptitle("Fitness Distribution Evolution — K=5 (α=0.25) vs K=20 (α=0.75)\n"
                 "Top: histograms for single run.  "
                 "Bottom: quantile bands (mean over 50 runs)", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig8_distribution_evolution")


# ============================================================
# Fig 9: Inequality spike — P90-P10 gap over time
# Shows early peak in inequality that occurs at t≈10
# ============================================================

def fig_inequality_trajectory():
    T_list = list(range(0, 50, 2)) + list(range(50, 300, 5))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)

    plot_configs = [
        (1,  0.5,  "Low K, moderate α"),
        (5,  0.25, "Moderate K, low α"),
        (5,  0.75, "Moderate K, high α"),
        (10, 0.5,  "High K, moderate α"),
        (20, 0.25, "Max ruggedness, low α"),
        (20, 0.75, "Max ruggedness, high α"),
    ]

    for ax, (K, alpha, title) in zip(axes.ravel(), plot_configs):
        for vt in VOTE_TYPES:
            fd = load_fd(K, alpha, vt)
            # Mean P90-P10 gap over 50 runs
            gaps = np.zeros((RUNS, len(T_list)))
            for r in range(RUNS):
                for ti, t in enumerate(T_list):
                    x = np.array(fd[r][t], dtype=float)
                    gaps[r, ti] = np.percentile(x, 90) - np.percentile(x, 10)
            mean_gap = gaps.mean(axis=0)
            ax.plot(T_list, mean_gap, color=COLORS[vt], lw=1.2, alpha=0.85,
                    label=VOTE_LABELS[vt])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("P90 – P10 fitness gap", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.axvline(10, color='gray', lw=0.8, ls=':', alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("P90–P10 Fitness Gap Over Time  (mean over 50 runs)\n"
                 "Shows early inequality spike and method divergence",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig9_inequality_trajectory")


# ============================================================
# Fig 10: Rank-gain violin plots
# For each method, violin of per-run rank-gain correlation,
# split by K at a fixed alpha
# ============================================================

def fig_rank_gain_violins():
    alpha = 0.5   # representative realistic α (α=0 and α=1 are special cases)
    K_list = [1, 5, 10, 15, 20]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: violin per method, grouped by K=5 and K=20
    ax = axes[0]
    positions = []
    data = []
    labels = []
    colors_v = []

    for ki, K in enumerate([5, 20]):
        for vi, vt in enumerate(VOTE_TYPES):
            fd = load_fd(K, alpha, vt)
            rhos = [rank_gain_corr(fd, r) for r in range(RUNS)]
            rhos = [r for r in rhos if not np.isnan(r)]
            pos = ki * (len(VOTE_TYPES) + 1) + vi
            positions.append(pos)
            data.append(rhos)
            labels.append(VOTE_LABELS[vt])
            colors_v.append(COLORS[vt])

    parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
    for i, (pc, col) in enumerate(zip(parts['bodies'], colors_v)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel("r(initial rank, gain)", fontsize=9)
    ax.set_title("Gain-Rank Correlation Distribution\n(K=5 left group, K=20 right group, α=0.5)",
                 fontsize=9)

    # Add K group labels
    k5_center  = np.mean([ki * (len(VOTE_TYPES) + 1) + vi for ki, K in [(0, 5)]
                          for vi in range(len(VOTE_TYPES))])
    k20_center = np.mean([ki * (len(VOTE_TYPES) + 1) + vi for ki, K in [(1, 20)]
                          for vi in range(len(VOTE_TYPES))])
    ax.annotate("K=5", xy=(k5_center, -0.95), ha='center', fontsize=8, color='navy')
    ax.annotate("K=20", xy=(k20_center, -0.95), ha='center', fontsize=8, color='darkred')
    ax.set_xticks(positions[::2])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)],
                       rotation=30, ha='right', fontsize=6)
    ax.grid(True, axis='y', lw=0.3)

    # Panel B: mean rank-gain r vs K, one line per method
    ax = axes[1]
    for vt in VOTE_TYPES:
        mean_rhos = []
        for K in K_list:
            fd = load_fd(K, alpha, vt)
            rhos = [rank_gain_corr(fd, r) for r in range(RUNS)]
            rhos = [r for r in rhos if not np.isnan(r)]
            mean_rhos.append(np.mean(rhos))
        ax.plot(K_list, mean_rhos, color=COLORS[vt], marker='o', ms=5,
                lw=2, label=VOTE_LABELS[vt])
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_xlabel("K (landscape ruggedness)", fontsize=10)
    ax.set_ylabel("Mean r(initial rank, gain)", fontsize=10)
    ax.set_title("Equity Direction vs. Ruggedness  (α=0.5)\n"
                 "Negative = poor benefit more;  Positive = rich benefit more",
                 fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, lw=0.3)

    fig.tight_layout()
    _save(fig, "fig10_rank_gain_violins")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()

    print("[Fig 7] Rank-gain correlation heatmap...")
    t0 = time.time()
    fig_rank_gain_heatmap()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig 8] Distribution evolution snapshots...")
    t0 = time.time()
    fig_distribution_evolution()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig 9] Inequality trajectory (P90-P10 over time)...")
    t0 = time.time()
    fig_inequality_trajectory()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig 10] Rank-gain violin plots...")
    t0 = time.time()
    fig_rank_gain_violins()
    print(f"  {time.time()-t0:.1f}s")

    print(f"\nAll done in {time.time()-t_start:.1f}s")
    print(f"Figures saved to {FIG_DIR}/")
