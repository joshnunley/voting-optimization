#!/usr/bin/env python3
"""
Representative democracy analysis.

Figures:
  Rep1 — β × α interaction: mean fitness for each β at each α (collapsed over K, methods)
  Rep2 — p_self effect: delegate vs trustee gap across (K, α)
  Rep3 — Method rank heatmaps under representation (best (β, p_self) config)
  Rep4 — β effect on optimal method: does representation change the regime transitions?
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]
METHOD_LABELS = {"approval": "Approval", "borda": "Borda", "irv": "IRV",
                 "minimax": "Minimax", "plurality": "Plurality",
                 "random_dictator": "Rand. Dict.", "star": "STAR",
                 "total_score": "Total Score"}
METHOD_COLORS = {"approval": "#e41a1c", "borda": "#377eb8", "irv": "#4daf4a",
                 "minimax": "#984ea3", "plurality": "#ff7f00",
                 "random_dictator": "#a65628", "star": "#f781bf", "total_score": "#999999"}


def load_rep():
    d = np.load("results/rep_results.npz")
    # mean shape: (nK, nA, nB, nP, nM)
    return (d['mean'], d['variance'],
            d['K'], d['alpha'], d['beta'], d['p_self'], list(d['methods']))


def _save(fig, name, dpi=200):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ──────────────────────────────────────────────────────────────────
# Rep1: β × α interaction
# For each β, plot mean fitness vs α (averaged over K, best method, best p_self)
# ──────────────────────────────────────────────────────────────────
def fig_beta_alpha():
    print("Rep1: β × α interaction ...")
    mean, var, K_arr, A_arr, B_arr, P_arr, methods = load_rep()
    nK, nA, nB, nP, nM = mean.shape

    # For each (K, α, β): take best p_self and best method → max over p_self and methods
    # Shape: (nK, nA, nB)
    best_mean = mean.max(axis=(3, 4))  # max over p_self and methods

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: line plot, one line per β, x=α, y=mean fitness averaged over K
    ax = axes[0]
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    for bi, beta in enumerate(B_arr):
        # Average over K
        y = best_mean[:, :, bi].mean(axis=0)  # (nA,)
        ax.plot(A_arr, y, 'o-', color=colors[bi], linewidth=2.5, markersize=6,
                label=f'β = {beta:.1f}')
    ax.set_xlabel("α (cross-dependency)", fontsize=12)
    ax.set_ylabel("Mean fitness (best method & p_self, avg over K)", fontsize=12)
    ax.set_title("β × α interaction", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Right: heatmap β × α, averaged over K
    ax = axes[1]
    # For each (α, β), average over K
    heatdata = best_mean.mean(axis=0)  # (nA, nB)
    im = ax.imshow(heatdata.T, aspect='auto', origin='lower', cmap='viridis',
                   extent=[-0.5, nA-0.5, -0.5, nB-0.5])
    ax.set_xticks(range(nA))
    ax.set_xticklabels([f"{a:.1f}" for a in A_arr], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(nB))
    ax.set_yticklabels([f"{b:.1f}" for b in B_arr])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("β", fontsize=12)
    ax.set_title("Mean fitness (best method & p_self, avg over K)", fontsize=13)
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    _save(fig, "rep1_beta_alpha.pdf")


# ──────────────────────────────────────────────────────────────────
# Rep2: p_self effect — delegate vs trustee
# ──────────────────────────────────────────────────────────────────
def fig_pself_effect():
    print("Rep2: p_self effect ...")
    mean, var, K_arr, A_arr, B_arr, P_arr, methods = load_rep()
    nK, nA, nB, nP, nM = mean.shape

    # For each (K, α): best method, best β — compare across p_self
    # Take max over methods and β → shape (nK, nA, nP)
    best_by_pself = mean.max(axis=(2, 4))  # max over β and methods
    # But actually, let's fix β=0 (policy voting) to isolate p_self effect
    beta0_idx = 0  # β=0
    by_pself_b0 = mean[:, :, beta0_idx, :, :].max(axis=3)  # max over methods → (nK, nA, nP)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: mean fitness for each p_self at fixed K
    for col, Ki in enumerate([1, 3]):  # K=5, K=20
        ax = axes[0, col]
        K = K_arr[Ki]
        for pi, ps in enumerate(P_arr):
            ax.plot(A_arr, by_pself_b0[Ki, :, pi], 'o-', linewidth=2,
                    label=f'p_self = {ps:.1f}', markersize=5)
        ax.set_xlabel("α", fontsize=11)
        ax.set_ylabel("Mean fitness (best method, β=0)", fontsize=11)
        ax.set_title(f"K = {int(K)}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    # Bottom row: delegate advantage = p_self=0 minus p_self=1
    delegate_adv = by_pself_b0[:, :, 0] - by_pself_b0[:, :, 2]  # p_self=0 minus p_self=1

    ax = axes[1, 0]
    for Ki, K in enumerate(K_arr):
        ax.plot(A_arr, delegate_adv[Ki], linewidth=1.5, label=f'K={int(K)}',
                alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("α", fontsize=11)
    ax.set_ylabel("Delegate advantage (p_self=0 − p_self=1)", fontsize=11)
    ax.set_title("Delegate vs Trustee gap (β=0, best method)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Bottom right: heatmap of delegate advantage
    ax = axes[1, 1]
    im = ax.imshow(delegate_adv, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-np.abs(delegate_adv).max(), vmax=np.abs(delegate_adv).max(),
                   extent=[-0.5, nA-0.5, -0.5, nK-0.5])
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_arr[i]:.1f}" for i in range(0, nA, 2)], fontsize=8)
    ax.set_yticks(range(nK))
    ax.set_yticklabels([f"{int(K)}" for K in K_arr])
    ax.set_xlabel("α", fontsize=11)
    ax.set_ylabel("K", fontsize=11)
    ax.set_title("Delegate advantage (red=delegate better)", fontsize=12)
    plt.colorbar(im, ax=ax)

    fig.suptitle("Effect of p_self (trustee vs delegate)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "rep2_pself_effect.pdf")


# ──────────────────────────────────────────────────────────────────
# Rep3: Method rank heatmaps under representation
# Use β=0, p_self=0 (policy voting, delegate — the "best" rep config)
# ──────────────────────────────────────────────────────────────────
def fig_rep_method_ranks():
    print("Rep3: method ranks under representation ...")
    mean, var, K_arr, A_arr, B_arr, P_arr, methods = load_rep()
    nK, nA, nB, nP, nM = mean.shape

    # β=0, p_self=0
    bi, pi = 0, 0
    mean_bp = mean[:, :, bi, pi, :]  # (nK, nA, nM)

    # Compute ranks
    ranks = np.zeros_like(mean_bp)
    for i in range(nK):
        for j in range(nA):
            order = np.argsort(-mean_bp[i, j, :])
            for r, idx in enumerate(order):
                ranks[i, j, idx] = r + 1

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    cmap = plt.cm.RdYlGn_r

    for ax, (midx, method) in zip(axes.ravel(), enumerate(methods)):
        im = ax.imshow(ranks[:, :, midx], aspect='auto', origin='lower',
                       cmap=cmap, vmin=1, vmax=8,
                       extent=[-0.5, nA-0.5, -0.5, nK-0.5])
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, nA, 2))
        ax.set_xticklabels([f"{A_arr[i]:.1f}" for i in range(0, nA, 2)],
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(nK))
        ax.set_yticklabels([f"{int(K)}" for K in K_arr])
        ax.set_xlabel("α", fontsize=9)
        ax.set_ylabel("K", fontsize=9)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1,2,3,4,5,6,7,8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"Method rank under representation (β={B_arr[bi]}, p_self={P_arr[pi]})",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "rep3_method_ranks.pdf")


# ──────────────────────────────────────────────────────────────────
# Rep4: Does representation change the regime transitions?
# Compare winner map: direct vs rep for each (β, p_self)
# ──────────────────────────────────────────────────────────────────
def fig_regime_comparison():
    print("Rep4: regime comparison ...")
    mean, var, K_arr, A_arr, B_arr, P_arr, methods = load_rep()
    nK, nA, nB, nP, nM = mean.shape

    # Winner at each (K, α) for each (β, p_self) config
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb

    for bi, beta in enumerate(B_arr):
        for pi, ps in enumerate(P_arr):
            ax = axes[bi, pi]
            mean_bp = mean[:, :, bi, pi, :]  # (nK, nA, nM)
            winner = np.argmax(mean_bp, axis=2)

            img = np.ones((nK, nA, 3))
            for i in range(nK):
                for j in range(nA):
                    img[i, j] = to_rgb(METHOD_COLORS[methods[winner[i, j]]])

            ax.imshow(img, aspect='auto', origin='lower',
                      extent=[-0.5, nA-0.5, -0.5, nK-0.5])
            ax.set_title(f"β={beta:.1f}, p_self={ps:.1f}", fontsize=11)
            ax.set_xticks(range(0, nA, 2))
            ax.set_xticklabels([f"{A_arr[i]:.1f}" for i in range(0, nA, 2)],
                               rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(nK))
            ax.set_yticklabels([f"{int(K)}" for K in K_arr], fontsize=8)
            ax.set_xlabel("α", fontsize=8)
            ax.set_ylabel("K", fontsize=8)

    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
               for m in methods]
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Winner maps across (β, p_self) configurations\n"
                 "(rows: β=0, 0.5, 1  |  cols: p_self=0, 0.5, 1)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, "rep4_regime_comparison.pdf")


if __name__ == "__main__":
    fig_beta_alpha()
    fig_pself_effect()
    fig_rep_method_ranks()
    fig_regime_comparison()
    print("\nAll rep figures done.")
