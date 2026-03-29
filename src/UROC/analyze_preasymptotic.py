#!/usr/bin/env python3
"""
Pre-asymptotic analysis of voting optimization dynamics.

Core argument: real democratic systems operate with limited time horizons.
Rankings and conclusions change substantially depending on whether you evaluate
at t=30, t=100, or t=299 (terminal).

Run from src/UROC/ with:
    python3 analyze_preasymptotic.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
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
ITERATIONS = 300
RUNS = 50

# Time horizons for pre-asymptotic analysis
HORIZONS = [10, 30, 50, 100, 150, 299]


def load_mh(K, alpha, vt):
    return np.load(f"{RESULTS}/direct/K{K}_a{alpha:.2f}/{vt}/mean_history.npy")

def load_vh(K, alpha, vt):
    return np.load(f"{RESULTS}/direct/K{K}_a{alpha:.2f}/{vt}/variance_history.npy")

def mean_curve(K, alpha, vt):
    return load_mh(K, alpha, vt).mean(axis=0)

def sem_curve(K, alpha, vt):
    mh = load_mh(K, alpha, vt)
    return mh.std(axis=0) / RUNS**0.5

def auc(curve, t_end):
    """Mean fitness over [0, t_end] — integrated performance."""
    return curve[:t_end+1].mean()

def rank_at(curves_dict, t):
    """Return dict: vt -> rank (1=best) at time t."""
    vals = sorted(VOTE_TYPES, key=lambda v: -curves_dict[v][t])
    return {v: i+1 for i, v in enumerate(vals)}

def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(f"{FIG_DIR}/{name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"  Saved {FIG_DIR}/{name}.pdf")
    plt.close(fig)


# ============================================================
# Fig P1: Rank trajectory heatmap
# For each (K, alpha), show how rankings evolve over 300 iterations.
# x=time (dense), y=method, color=rank.
# ============================================================

def fig_rank_trajectory():
    configs = [
        (5,  0.25, "K=5, α=0.25"),
        (5,  0.75, "K=5, α=0.75"),
        (10, 0.50, "K=10, α=0.50"),
        (20, 0.50, "K=20, α=0.50"),
    ]
    # Dense time axis up to t=150 then coarser (pre-asymptotic focus)
    T_dense = list(range(0, 50)) + list(range(50, 150, 2)) + list(range(150, 300, 5))
    T_arr   = np.array(T_dense)

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 10), sharex=True)

    for ax, (K, alpha, title) in zip(axes, configs):
        curves = {vt: mean_curve(K, alpha, vt) for vt in VOTE_TYPES}
        # rank matrix: shape (n_methods, n_timepoints)
        rank_matrix = np.zeros((len(VOTE_TYPES), len(T_dense)))
        for ti, t in enumerate(T_dense):
            ranks = rank_at(curves, t)
            for vi, vt in enumerate(VOTE_TYPES):
                rank_matrix[vi, ti] = ranks[vt]

        im = ax.imshow(rank_matrix, aspect='auto', origin='upper',
                       vmin=1, vmax=8, cmap='RdYlGn_r',
                       extent=[0, T_arr[-1], 7.5, -0.5])
        ax.set_yticks(range(8))
        ax.set_yticklabels([VOTE_LABELS[vt] for vt in VOTE_TYPES], fontsize=8)
        ax.set_title(title, fontsize=10, pad=3)
        ax.axvline(30,  color='white', lw=1.5, ls='--', alpha=0.7)
        ax.axvline(100, color='white', lw=1.5, ls='--', alpha=0.7)
        ax.text(30,  -0.4, "t=30",  ha='center', fontsize=7, color='white', fontweight='bold')
        ax.text(100, -0.4, "t=100", ha='center', fontsize=7, color='white', fontweight='bold')

    axes[-1].set_xlabel("Iteration", fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Rank (1=best, 8=worst)",
                 ticks=[1,2,3,4,5,6,7,8])
    fig.suptitle("Method Rank Over Time  (Green=best, Red=worst)\n"
                 "Dashed lines at t=30 and t=100 mark pre-asymptotic boundaries",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "figP1_rank_trajectory")


# ============================================================
# Fig P2: Crossover curves
# Zoomed learning curves (t=0-150) for configurations with
# notable rank crossovers. Highlight the crossover points.
# ============================================================

def fig_crossover_curves():
    configs = [
        (5,  0.25, "K=5, α=0.25  [Approval leads; IRV rises late]"),
        (5,  0.75, "K=5, α=0.75  [Plurality rises; Minimax/STAR fall]"),
        (10, 0.50, "K=10, α=0.50  [Total Score leads early; IRV wins late]"),
        (20, 0.50, "K=20, α=0.50  [High epistasis: rapid convergence]"),
    ]
    T = np.arange(ITERATIONS)
    T_zoom = T[:151]   # focus on first 150 iterations

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, (K, alpha, title) in zip(axes.ravel(), configs):
        for vt in VOTE_TYPES:
            if vt == "random_dictator":
                continue   # exclude for clarity
            c = mean_curve(K, alpha, vt)
            s = sem_curve(K, alpha, vt)
            ax.plot(T_zoom, c[:151], color=COLORS[vt], ls=LINESTYLES[vt],
                    lw=2, label=VOTE_LABELS[vt], zorder=3)
            ax.fill_between(T_zoom, c[:151]-s[:151], c[:151]+s[:151],
                            color=COLORS[vt], alpha=0.08)

        ax.axvline(30, color='gray', lw=1, ls=':', alpha=0.6)
        ax.axvline(100, color='gray', lw=1, ls=':', alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Mean population fitness", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_xlim(0, 150)
        ax.text(30, ax.get_ylim()[0], "30", ha='center', va='bottom', fontsize=7,
                color='gray')
        ax.text(100, ax.get_ylim()[0], "100", ha='center', va='bottom', fontsize=7,
                color='gray')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning Curves (t=0–150): Crossover Dynamics\n"
                 "Vertical lines at t=30 and t=100", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "figP2_crossover_curves")


# ============================================================
# Fig P3: AUC (integrated performance) at multiple time horizons
# Side-by-side bars for each method, one cluster per horizon.
# ============================================================

def fig_auc_horizons():
    configs = [
        (5,  0.25),
        (5,  0.75),
        (10, 0.50),
        (20, 0.50),
    ]
    config_labels = ["K=5, α=0.25", "K=5, α=0.75", "K=10, α=0.50", "K=20, α=0.50"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, (K, alpha), cfg_label in zip(axes.ravel(), configs, config_labels):
        curves = {vt: mean_curve(K, alpha, vt) for vt in VOTE_TYPES}
        n_methods = len(VOTE_TYPES) - 1  # exclude random_dictator for scale
        plot_vts = [vt for vt in VOTE_TYPES if vt != "random_dictator"]

        x = np.arange(len(HORIZONS))
        w = 0.85 / len(plot_vts)

        for vi, vt in enumerate(plot_vts):
            aucs = [auc(curves[vt], t) for t in HORIZONS]
            offset = (vi - len(plot_vts)/2 + 0.5) * w
            ax.bar(x + offset, aucs, w,
                   color=COLORS[vt], alpha=0.85, label=VOTE_LABELS[vt])

        ax.set_title(cfg_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T={t}" for t in HORIZONS], fontsize=8)
        ax.set_xlabel("Time horizon (AUC over t=0 to T)", fontsize=8)
        ax.set_ylabel("Mean fitness (integrated)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis='y', lw=0.3, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Integrated Performance (AUC) at Different Time Horizons\n"
                 "(Random Dictator excluded for scale clarity)", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "figP3_auc_horizons")


# ============================================================
# Fig P4: Speed vs quality scatter
# x = time to reach 75% of total gain (convergence speed proxy)
# y = terminal mean fitness (quality)
# Each point = one (method, K, alpha) configuration
# ============================================================

def fig_speed_quality():
    K_list    = [1, 5, 10, 15, 20]
    alpha_list = [0.25, 0.5, 0.75]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, t_window, title in zip(
            axes,
            [30, 100],
            ["Speed: t to 75% gain  vs  Terminal fitness",
             "AUC(t=0–100)  vs  Terminal fitness"]):
        for vt in VOTE_TYPES:
            xs, ys = [], []
            for K in K_list:
                for alpha in alpha_list:
                    curve = mean_curve(K, alpha, vt)
                    total_g = curve[-1] - curve[0]
                    if abs(total_g) < 0.1:
                        continue
                    if t_window == 30:
                        # x = time to 75% of gain
                        target = curve[0] + 0.75 * total_g
                        t75 = next((t for t in range(300) if curve[t] >= target), 299)
                        xs.append(t75)
                    else:
                        # x = AUC over t=0..100
                        xs.append(auc(curve, 100))
                    ys.append(curve[-1])   # terminal fitness

            ax.scatter(xs, ys, color=COLORS[vt], alpha=0.6, s=30,
                       label=VOTE_LABELS[vt], zorder=3)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time to 75% gain (iterations)" if t_window == 30
                      else "AUC(t=0–100)", fontsize=9)
        ax.set_ylabel("Terminal mean fitness (t=299)", fontsize=9)
        ax.legend(fontsize=7, loc='lower right' if t_window == 30 else 'upper left')
        ax.grid(True, lw=0.3, alpha=0.4)

    fig.suptitle("Speed vs Quality Trade-off Across All Configurations\n"
                 "(Each point = one method × K × α combination)", fontsize=11)
    fig.tight_layout()
    _save(fig, "figP4_speed_quality")


# ============================================================
# Fig P5: Pre-asymptotic equity
# Variance trajectory for first 50 iterations (zoomed)
# plus comparison of early vs late variance delta by K and alpha
# ============================================================

def fig_preasymptotic_equity():
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    T_early = np.arange(50)
    configs_top = [(5, 0.25), (10, 0.50), (20, 0.75)]

    # Top row: variance in first 50 iterations (all methods)
    for col, (K, alpha) in enumerate(configs_top):
        ax = fig.add_subplot(gs[0, col])
        for vt in VOTE_TYPES:
            vh = load_vh(K, alpha, vt)
            curve = vh.mean(axis=0)[:50]
            ax.plot(T_early, curve, color=COLORS[vt], ls=LINESTYLES[vt],
                    lw=1.5, label=VOTE_LABELS[vt])
        ax.set_title(f"Variance — K={K}, α={alpha} (first 50 iter)", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Population variance", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)

    # Bottom row: Δvariance (late - early) as a bar chart, for all configs
    ax_bar = fig.add_subplot(gs[1, :])
    K_list    = [1, 5, 10, 15, 20]
    alpha_list = [0.25, 0.5, 0.75]
    configs_bar = [(K, a) for K in K_list for a in alpha_list]

    x = np.arange(len(configs_bar))
    w = 0.8 / len(VOTE_TYPES)

    for vi, vt in enumerate(VOTE_TYPES):
        deltas = []
        for K, alpha in configs_bar:
            vh = load_vh(K, alpha, vt)
            v_early = vh[:, :30].mean()
            v_late  = vh[:, 250:].mean()
            deltas.append(v_late - v_early)
        offset = (vi - len(VOTE_TYPES)/2 + 0.5) * w
        ax_bar.bar(x + offset, deltas, w,
                   color=COLORS[vt], alpha=0.8, label=VOTE_LABELS[vt])

    ax_bar.axhline(0, color='black', lw=0.8, ls='--')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"K={K}\nα={a}" for K, a in configs_bar],
                           fontsize=6, rotation=0)
    ax_bar.set_ylabel("Δ Variance  (late − early)", fontsize=9)
    ax_bar.set_title("Change in Population Fitness Variance  (late t=250–299 minus early t=0–30)\n"
                     "Positive = inequality grew;  Negative = inequality shrank", fontsize=9)
    ax_bar.legend(fontsize=6, loc='upper right', ncol=2)
    ax_bar.grid(True, axis='y', lw=0.3, alpha=0.5)

    # Add K group separators
    for sep in [2.5, 5.5, 8.5, 11.5]:
        ax_bar.axvline(sep, color='gray', lw=0.8, ls=':', alpha=0.5)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7,
               bbox_to_anchor=(1.01, 0.72), title="Method")

    fig.suptitle("Pre-asymptotic Equity Dynamics\n"
                 "Top: variance in first 50 iterations.  "
                 "Bottom: total variance change from start to end of run.",
                 fontsize=11)
    _save(fig, "figP5_preasymptotic_equity")


# ============================================================
# Fig P6: Early rank vs terminal rank comparison
# Scatter: rank at t=30 vs rank at t=299, all (K, alpha) configs
# Shows which methods consistently improve or decline in rank
# ============================================================

def fig_early_vs_late_rank():
    K_list    = [1, 5, 10, 15, 20]
    alpha_list = [0.25, 0.5, 0.75]

    early_T  = 30
    late_T   = 299

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: scatter of early rank vs late rank per method
    ax = axes[0]
    jitter = 0.15
    for vt in VOTE_TYPES:
        early_ranks, late_ranks = [], []
        for K in K_list:
            for alpha in alpha_list:
                curves = {v: mean_curve(K, alpha, v) for v in VOTE_TYPES}
                er = rank_at(curves, early_T)[vt]
                lr = rank_at(curves, late_T)[vt]
                early_ranks.append(er + np.random.uniform(-jitter, jitter))
                late_ranks.append(lr + np.random.uniform(-jitter, jitter))
        ax.scatter(early_ranks, late_ranks, color=COLORS[vt], alpha=0.55,
                   s=25, label=VOTE_LABELS[vt], zorder=3)
    ax.plot([1, 8], [1, 8], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel(f"Rank at t={early_T}  (early)", fontsize=10)
    ax.set_ylabel(f"Rank at t={late_T}  (terminal)", fontsize=10)
    ax.set_title(f"Early vs Terminal Rank\n(above diagonal = improved; below = declined)",
                 fontsize=9)
    ax.set_xlim(0.5, 8.5); ax.set_ylim(0.5, 8.5)
    ax.set_xticks(range(1, 9)); ax.set_yticks(range(1, 9))
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, lw=0.3)

    # Panel B: mean rank change (early → late) per method, as bar
    ax = axes[1]
    rank_deltas = {}
    rank_delta_stds = {}
    for vt in VOTE_TYPES:
        deltas = []
        for K in K_list:
            for alpha in alpha_list:
                curves = {v: mean_curve(K, alpha, v) for v in VOTE_TYPES}
                er = rank_at(curves, early_T)[vt]
                lr = rank_at(curves, late_T)[vt]
                deltas.append(lr - er)   # negative = improved rank
        rank_deltas[vt] = np.mean(deltas)
        rank_delta_stds[vt] = np.std(deltas)

    sorted_vts = sorted(VOTE_TYPES, key=lambda v: rank_deltas[v])
    ys = [rank_deltas[vt] for vt in sorted_vts]
    es = [rank_delta_stds[vt] for vt in sorted_vts]
    x = np.arange(len(sorted_vts))
    bars = ax.bar(x, ys, color=[COLORS[vt] for vt in sorted_vts],
                  alpha=0.85, width=0.6)
    ax.errorbar(x, ys, yerr=es, fmt='none', color='black', capsize=4, lw=1.2)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([VOTE_LABELS[vt] for vt in sorted_vts],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Mean rank change  (terminal − early rank)\n"
                  "Negative = improved (late riser);  Positive = declined", fontsize=8)
    ax.set_title(f"Mean Rank Δ from t={early_T} to t={late_T}\n"
                 f"(averaged over all K × α configurations)", fontsize=9)
    ax.grid(True, axis='y', lw=0.3)

    print("\n=== Rank change early→terminal (sorted) ===")
    for vt in sorted_vts:
        print(f"  {VOTE_LABELS[vt]:20s}: Δrank = {rank_deltas[vt]:+.2f} ± {rank_delta_stds[vt]:.2f}")

    fig.tight_layout()
    _save(fig, "figP6_early_vs_late_rank")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()

    print("[Fig P1] Rank trajectory heatmap...")
    t0 = time.time()
    fig_rank_trajectory()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig P2] Crossover curves...")
    t0 = time.time()
    fig_crossover_curves()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig P3] AUC at multiple horizons...")
    t0 = time.time()
    fig_auc_horizons()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig P4] Speed vs quality scatter...")
    t0 = time.time()
    fig_speed_quality()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig P5] Pre-asymptotic equity...")
    t0 = time.time()
    fig_preasymptotic_equity()
    print(f"  {time.time()-t0:.1f}s")

    print("[Fig P6] Early vs late rank comparison...")
    t0 = time.time()
    fig_early_vs_late_rank()
    print(f"  {time.time()-t0:.1f}s")

    print(f"\nAll done in {time.time()-t_start:.1f}s — figures in {FIG_DIR}/")
