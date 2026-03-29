#!/usr/bin/env python3
"""
Analysis script for voting optimization experiments.
Produces all paper figures.

Run from src/UROC/ with:
    python3 analyze.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

# ============================================================
# Configuration
# ============================================================

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

K_VALUES    = [0, 1, 5, 10, 15, 20]
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
REP_K       = [5, 10, 15]
CANDS       = [3, 5]
TEMPS_RAW   = [None, 1.0, 0.1]
TEMP_LABELS = {None: "Uniform", 1.0: "τ=1.0", 0.1: "τ=0.1"}

ITERATIONS = 300
RUNS = 50
TERMINAL_WINDOW = 10   # average over last N iterations as "terminal" value

# Consistent colors and line styles across all figures
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
LINESTYLES = {
    "plurality":       "-",
    "approval":        "-",
    "total_score":     "-",
    "borda":           "-",
    "irv":             "--",
    "star":            "--",
    "minimax":         "--",
    "random_dictator": ":",
}


# ============================================================
# Data loading helpers
# ============================================================

def load_direct(K, alpha, vote_type, what="mean"):
    """Load (runs, iterations) array for direct democracy experiment."""
    path = f"{RESULTS}/direct/K{K}_a{alpha:.2f}/{vote_type}/{what}_history.npy"
    return np.load(path)


def load_rep(K, alpha, vote_type, n_cand, temp, what="mean"):
    """Load (runs, iterations) array for representative democracy experiment."""
    ts = f"_t{temp}" if temp is not None else "_tuniform"
    path = f"{RESULTS}/representative/K{K}_a{alpha:.2f}/c{n_cand}{ts}/{vote_type}/{what}_history.npy"
    return np.load(path)


def terminal(arr):
    """Scalar: mean over runs and last TERMINAL_WINDOW iterations."""
    return float(arr[:, -TERMINAL_WINDOW:].mean())


# ============================================================
# Sanity checks (printed to stdout)
# ============================================================

def sanity_check():
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    print("\n--- K=0, α=0: all methods should score similarly ---")
    for vt in VOTE_TYPES:
        mh = load_direct(0, 0.0, vt, "mean")
        print(f"  {VOTE_LABELS[vt]:20s}: {terminal(mh):.4f}")

    print("\n--- K=20, α=1.0: highest ruggedness + full coupling ---")
    for vt in VOTE_TYPES:
        mh = load_direct(20, 1.0, vt, "mean")
        print(f"  {VOTE_LABELS[vt]:20s}: {terminal(mh):.4f}")

    print("\n--- Variance change (α=1 − α=0) at K=5 ---")
    print("  (positive = more unequal outcomes when coupling is high)")
    for vt in VOTE_TYPES:
        vh0 = load_direct(5, 0.0, vt, "variance")
        vh1 = load_direct(5, 1.0, vt, "variance")
        dv = terminal(vh1) - terminal(vh0)
        print(f"  {VOTE_LABELS[vt]:20s}: Δvar = {dv:+.3f}")

    print("\n--- K monotonicity check: terminal fitness should decrease with K ---")
    print("  (for α=0.5, plurality)")
    for K in K_VALUES:
        alpha = 0.0 if K == 0 else 0.5
        mh = load_direct(K, alpha, "plurality", "mean")
        print(f"  K={K:2d}: {terminal(mh):.4f}")


# ============================================================
# Fig 1: Learning curves — mean fitness over time
# 3×3 grid: rows = α ∈ {0.25, 0.50, 0.75}, cols = K ∈ {1, 10, 20}
# (α=0 and α=1 are mathematical special cases; main paper focus is realistic α)
# ============================================================

def fig_learning_curves():
    plot_K     = [1, 10, 20]
    plot_alpha = [0.25, 0.50, 0.75]
    T = np.arange(ITERATIONS)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)

    for row, alpha in enumerate(plot_alpha):
        for col, K in enumerate(plot_K):
            ax = axes[row, col]
            for vt in VOTE_TYPES:
                mh = load_direct(K, alpha, vt, "mean")
                curve = mh.mean(axis=0)
                sem   = mh.std(axis=0) / RUNS**0.5
                ax.plot(T, curve, color=COLORS[vt], ls=LINESTYLES[vt],
                        lw=1.5, label=VOTE_LABELS[vt])
                ax.fill_between(T, curve - sem, curve + sem,
                                color=COLORS[vt], alpha=0.08)
            ax.set_title(f"K={K}, α={alpha:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel("Mean population fitness", fontsize=8)
            if row == 2:
                ax.set_xlabel("Iteration", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Mean Fitness Over Time — Direct Democracy\n"
                 "Rows: realistic cross-dependency values (α=0 and α=1 are special cases)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig1_learning_curves")


# ============================================================
# Fig 2: Terminal fitness heatmap — K × α per method
# ============================================================

def fig_terminal_heatmap():
    K_list = [1, 5, 10, 15, 20]   # exclude K=0 (only one alpha)
    A_list = ALPHA_VALUES

    n_K = len(K_list)
    n_A = len(A_list)
    term = np.zeros((len(VOTE_TYPES), n_K, n_A))

    for mi, vt in enumerate(VOTE_TYPES):
        for ki, K in enumerate(K_list):
            for ai, alpha in enumerate(A_list):
                mh = load_direct(K, alpha, vt, "mean")
                term[mi, ki, ai] = terminal(mh)

    vmin, vmax = term.min(), term.max()

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()

    for mi, vt in enumerate(VOTE_TYPES):
        ax = axes[mi]
        im = ax.imshow(term[mi], aspect='auto', origin='lower',
                       vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(VOTE_LABELS[vt], fontsize=9, color=COLORS[vt])
        ax.set_xticks(range(n_A))
        ax.set_xticklabels([f"{a}" for a in A_list], fontsize=7)
        ax.set_yticks(range(n_K))
        ax.set_yticklabels([f"K={k}" for k in K_list], fontsize=7)
        if mi % 4 == 0:
            ax.set_ylabel("K (ruggedness)", fontsize=8)
        if mi >= 4:
            ax.set_xlabel("α (coupling)", fontsize=8)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6,
                 label="Terminal mean fitness", pad=0.02)
    fig.suptitle("Terminal Mean Fitness — K × α per Voting Method", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig2_terminal_heatmap")

    return term, K_list, A_list


# ============================================================
# Fig 3: Method rankings
# Panel A: rank heatmap (method × α, averaged over K)
# Panel B: rank trajectory across K at α=0.5
# ============================================================

def fig_rankings(term, K_list, A_list):
    n_methods, n_K, n_A = term.shape

    # ranks[mi, ki, ai]: 0=best, 7=worst
    ranks = np.zeros_like(term, dtype=int)
    for ki in range(n_K):
        for ai in range(n_A):
            order = np.argsort(-term[:, ki, ai])
            for pos, mi in enumerate(order):
                ranks[mi, ki, ai] = pos

    # Print average rankings
    avg_rank = ranks.mean(axis=(1, 2))
    print("\n" + "=" * 60)
    print("METHOD RANKINGS (averaged over all K×α, 0=best)")
    print("=" * 60)
    sorted_idx = np.argsort(avg_rank)
    for pos, mi in enumerate(sorted_idx):
        print(f"  #{pos+1:d}  {VOTE_LABELS[VOTE_TYPES[mi]]:20s}: avg rank {avg_rank[mi]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: heatmap of avg rank by α (rows=methods, cols=α)
    ax = axes[0]
    mean_rank_by_alpha = ranks.mean(axis=1)   # (n_methods, n_A)
    # Sort methods by overall avg rank for readability
    sorted_methods = [VOTE_TYPES[i] for i in sorted_idx]
    sorted_data    = mean_rank_by_alpha[sorted_idx, :]

    im = ax.imshow(sorted_data, aspect='auto', origin='upper',
                   vmin=0, vmax=7, cmap='RdYlGn_r')
    ax.set_xticks(range(n_A))
    ax.set_xticklabels([f"α={a}" for a in A_list], fontsize=9)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([VOTE_LABELS[vt] for vt in sorted_methods], fontsize=9)
    ax.set_title("Mean Rank by α  (averaged over K)\n0=best, 7=worst", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel B: rank vs K for α=0.5
    ax = axes[1]
    ai_mid = list(A_list).index(0.5)
    for mi, vt in enumerate(VOTE_TYPES):
        ax.plot(K_list, ranks[mi, :, ai_mid],
                color=COLORS[vt], ls=LINESTYLES[vt], lw=2,
                marker='o', ms=5, label=VOTE_LABELS[vt])
    ax.set_xlabel("K (landscape ruggedness)", fontsize=10)
    ax.set_ylabel("Rank  (1=best)", fontsize=10)
    ax.set_title("Method Rankings vs. K  (α=0.5)", fontsize=10)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"#{i+1}" for i in range(8)], fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, lw=0.3)

    fig.tight_layout()
    _save(fig, "fig3_rankings")


# ============================================================
# Fig 4: Variance / equity analysis
# Top: variance over time (2×2 grid)
# Bottom: mean–variance frontier (all configs, colored by method)
# ============================================================

def fig_variance():
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    plot_K     = [5, 20]
    plot_alpha = [0.25, 0.50, 0.75]
    T = np.arange(ITERATIONS)

    # Variance trajectories
    for row, alpha in enumerate(plot_alpha):
        for col, K in enumerate(plot_K):
            ax = fig.add_subplot(gs[row, col])
            for vt in VOTE_TYPES:
                vh = load_direct(K, alpha, vt, "variance")
                curve = vh.mean(axis=0)
                ax.plot(T, curve, color=COLORS[vt], ls=LINESTYLES[vt],
                        lw=1.5, label=VOTE_LABELS[vt])
            ax.set_title(f"Variance — K={K}, α={alpha:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel("Population fitness variance", fontsize=8)
            if row == 2:
                ax.set_xlabel("Iteration", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)

    # Mean–variance frontier
    ax_f = fig.add_subplot(gs[3, :])
    K_list = [1, 5, 10, 15, 20]

    for mi, vt in enumerate(VOTE_TYPES):
        xs, ys = [], []
        for K in K_list:
            for alpha in ALPHA_VALUES:
                mh = load_direct(K, alpha, vt, "mean")
                vh = load_direct(K, alpha, vt, "variance")
                xs.append(terminal(mh))
                ys.append(terminal(vh))
        ax_f.scatter(xs, ys, color=COLORS[vt], alpha=0.55, s=22,
                     label=VOTE_LABELS[vt], zorder=3)

    ax_f.set_xlabel("Terminal mean fitness  (efficiency →)", fontsize=10)
    ax_f.set_ylabel("Terminal population variance  (← equality)", fontsize=10)
    ax_f.set_title("Mean–Variance Frontier: All Direct Democracy Configurations\n"
                   "(upper-left = good efficiency + high inequality; lower-right = good efficiency + low inequality)",
                   fontsize=9)
    ax_f.legend(fontsize=7, loc='upper left', ncol=2)
    ax_f.grid(True, lw=0.3)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7,
               bbox_to_anchor=(1.02, 0.68), title="Method")

    _save(fig, "fig4_variance")


# ============================================================
# Fig 5: Representative vs. direct democracy
# Bars: Δ mean and Δ variance by (K, n_cand, temperature)
# ============================================================

def fig_representative():
    n_K  = len(REP_K)
    n_A  = len(ALPHA_VALUES)
    n_C  = len(CANDS)
    n_T  = len(TEMPS_RAW)
    n_VT = len(VOTE_TYPES)

    delta_mean = np.zeros((n_K, n_A, n_C, n_T, n_VT))
    delta_var  = np.zeros((n_K, n_A, n_C, n_T, n_VT))

    for ki, K in enumerate(REP_K):
        for ai, alpha in enumerate(ALPHA_VALUES):
            for ci, n_cand in enumerate(CANDS):
                for ti, temp in enumerate(TEMPS_RAW):
                    for vi, vt in enumerate(VOTE_TYPES):
                        dir_mh = load_direct(K, alpha, vt, "mean")
                        dir_vh = load_direct(K, alpha, vt, "variance")
                        rep_mh = load_rep(K, alpha, vt, n_cand, temp, "mean")
                        rep_vh = load_rep(K, alpha, vt, n_cand, temp, "variance")
                        delta_mean[ki, ai, ci, ti, vi] = terminal(rep_mh) - terminal(dir_mh)
                        delta_var [ki, ai, ci, ti, vi] = terminal(rep_vh) - terminal(dir_vh)

    # Print summary
    print("\n" + "=" * 60)
    print("REPRESENTATIVE vs. DIRECT: summary (mean over K, α, methods)")
    print("=" * 60)
    print("\nBy temperature:")
    for ti, temp in enumerate(TEMPS_RAW):
        dm = delta_mean[:, :, :, ti, :].mean()
        dv = delta_var [:, :, :, ti, :].mean()
        print(f"  {TEMP_LABELS[temp]:12s}: Δmean={dm:+.4f}  Δvar={dv:+.4f}")
    print("\nBy candidate count:")
    for ci, n_cand in enumerate(CANDS):
        dm = delta_mean[:, :, ci, :, :].mean()
        dv = delta_var [:, :, ci, :, :].mean()
        print(f"  C={n_cand}:          Δmean={dm:+.4f}  Δvar={dv:+.4f}")

    # Plot
    fig, axes = plt.subplots(2, n_K, figsize=(13, 8), sharey='row')
    x = np.arange(n_T)
    w = 0.35
    cand_colors = ["#4e79a7", "#f28e2b"]

    for ki, K in enumerate(REP_K):
        for row, (data, ylabel) in enumerate(
                [(delta_mean, "Δ Mean Fitness (rep − direct)"),
                 (delta_var,  "Δ Variance (rep − direct)")]):
            ax = axes[row, ki]
            for ci, n_cand in enumerate(CANDS):
                # Average over vote_types and alpha
                vals = data[ki, :, ci, :, :].mean(axis=(0, 2))   # shape (n_T,)
                ax.bar(x + (ci - 0.5) * w, vals, w,
                       label=f"C={n_cand}", color=cand_colors[ci], alpha=0.85)
            ax.axhline(0, color='black', lw=0.8, ls='--')
            ax.set_xticks(x)
            ax.set_xticklabels([TEMP_LABELS[t] for t in TEMPS_RAW], fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, axis='y', lw=0.3, alpha=0.6)
            if ki == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.set_title(f"K={K}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Candidate selection", fontsize=8)

    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle("Representative vs. Direct Democracy\n"
                 "(Δ averaged over all α and voting methods)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig5_representative")

    return delta_mean, delta_var


# ============================================================
# Fig 6 (bonus): Per-method Δ under representation
# Reveals which voting methods are most/least robust to representation
# ============================================================

def fig_rep_per_method(delta_mean, delta_var):
    """
    For each voting method, show Δmean (rep−direct) averaged over K and α,
    broken out by (n_cand, temperature).
    """
    n_T = len(TEMPS_RAW)
    n_C = len(CANDS)

    # avg over K and alpha: shape (n_C, n_T, n_VT)
    dm_avg = delta_mean.mean(axis=(0, 1))   # (n_C, n_T, n_VT)
    dv_avg = delta_var .mean(axis=(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(VOTE_TYPES))
    w = 0.12
    offsets = np.linspace(-(n_C * n_T - 1) / 2, (n_C * n_T - 1) / 2, n_C * n_T) * w

    palette = plt.cm.tab10(np.linspace(0, 0.6, n_C * n_T))

    for ax, (dm, title) in zip(axes, [(dm_avg, "Δ Mean Fitness"),
                                       (dv_avg, "Δ Variance")]):
        for ci, n_cand in enumerate(CANDS):
            for ti, temp in enumerate(TEMPS_RAW):
                idx = ci * n_T + ti
                vals = dm[ci, ti, :]
                bars = ax.bar(x + offsets[idx], vals, w,
                              label=f"C={n_cand}, {TEMP_LABELS[temp]}",
                              color=palette[idx], alpha=0.85)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels([VOTE_LABELS[vt] for vt in VOTE_TYPES],
                           rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title + " per Method\n(mean over K and α)", fontsize=10)
        ax.grid(True, axis='y', lw=0.3)

    axes[0].legend(fontsize=7, loc='lower left', ncol=2)
    fig.tight_layout()
    _save(fig, "fig6_rep_per_method")


# ============================================================
# Utility
# ============================================================

def _save(fig, name):
    pdf = f"{FIG_DIR}/{name}.pdf"
    png = f"{FIG_DIR}/{name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=150)
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"  Saved {pdf}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()

    sanity_check()

    print("\n[Fig 1] Learning curves...")
    t0 = time.time()
    fig_learning_curves()
    print(f"  {time.time()-t0:.1f}s")

    print("\n[Fig 2] Terminal fitness heatmap...")
    t0 = time.time()
    term, K_list, A_list = fig_terminal_heatmap()
    print(f"  {time.time()-t0:.1f}s")

    print("\n[Fig 3] Method rankings...")
    t0 = time.time()
    fig_rankings(term, K_list, A_list)
    print(f"  {time.time()-t0:.1f}s")

    print("\n[Fig 4] Variance analysis...")
    t0 = time.time()
    fig_variance()
    print(f"  {time.time()-t0:.1f}s")

    print("\n[Fig 5] Representative vs direct...")
    t0 = time.time()
    dm, dv = fig_representative()
    print(f"  {time.time()-t0:.1f}s")

    print("\n[Fig 6] Per-method rep breakdown...")
    t0 = time.time()
    fig_rep_per_method(dm, dv)
    print(f"  {time.time()-t0:.1f}s")

    print(f"\nAll figures saved to {FIG_DIR}/")
    print(f"Total time: {time.time()-t_start:.1f}s")
