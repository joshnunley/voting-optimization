#!/usr/bin/env python3
"""
Fit a complexity parameter C(K, alpha) such that voting method regime
transitions are maximally predicted by C alone.

The empirical progression (increasing complexity) is:
  Total Score → Plurality → (IRV) → Borda → STAR

Steps:
  1. Load terminal mean fitness, compute rank-1 method at each (K, alpha).
  2. Assign ordinal labels to the regime progression.
  3. Fit general quadratic C(K, alpha) via ordinal regression.
  4. Simplify to single-parameter form.
  5. Diagnostic plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

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

K_FULL = list(range(1, 21))
A_FULL = [round(a * 0.05, 2) for a in range(21)]

# Regime progression — the methods that dominate in sequence
REGIME_METHODS = ["total_score", "plurality", "irv", "borda", "star"]
REGIME_LABELS  = {m: i for i, m in enumerate(REGIME_METHODS)}
REGIME_NAMES   = ["Total Score", "Plurality", "IRV", "Borda", "STAR"]

N_TERMINAL = 10


def load_data():
    """Load terminal mean fitness. Try npz first, fall back to raw files."""
    npz_path = "results/direct_results.npz"
    if os.path.exists(npz_path):
        print("Loading from consolidated npz ...", flush=True)
        d = np.load(npz_path)
        return d['mean']   # (nK, nA, nM)

    print("Loading from raw files (mmap) ...", flush=True)
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
            print(f"  K={K} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"Loaded in {time.time()-t0:.0f}s", flush=True)
    return mean


def compute_winner_map(mean):
    """For each (K, alpha), find rank-1 method."""
    nK, nA, nM = mean.shape
    winner = np.full((nK, nA), -1, dtype=int)
    for i in range(nK):
        for j in range(nA):
            vals = mean[i, j, :]
            if not np.all(np.isnan(vals)):
                winner[i, j] = np.nanargmax(vals)
    return winner


def compute_regime_labels(winner):
    """
    For each cell, assign an ordinal label based on which regime method wins.
    Cells where a non-regime method wins get the label of the closest
    regime method by rank.
    """
    nK, nA = winner.shape
    labels = np.full((nK, nA), np.nan)
    regime_indices = [METHODS.index(m) for m in REGIME_METHODS]

    for i in range(nK):
        for j in range(nA):
            w = winner[i, j]
            if w < 0:
                continue
            m = METHODS[w]
            if m in REGIME_LABELS:
                labels[i, j] = REGIME_LABELS[m]
            else:
                # Assign to closest regime method by interpolating
                # Use the regime method with highest fitness among regime methods
                # (i.e., which regime method would have won if we excluded non-regime ones)
                labels[i, j] = np.nan  # exclude from fit
    return labels


def compute_regime_labels_v2(mean):
    """
    For each cell, find the rank-1 method among ONLY the regime methods.
    This gives clean labels even when a non-regime method wins overall.
    """
    nK, nA, _ = mean.shape
    regime_indices = [METHODS.index(m) for m in REGIME_METHODS]
    labels = np.full((nK, nA), np.nan)

    for i in range(nK):
        for j in range(nA):
            regime_vals = np.array([mean[i, j, k] for k in regime_indices])
            if not np.all(np.isnan(regime_vals)):
                best = np.nanargmax(regime_vals)
                labels[i, j] = best  # 0=TS, 1=Plur, 2=IRV, 3=Borda, 4=STAR
    return labels


def build_features(K_arr, A_arr):
    """Build feature matrix for general quadratic: [1, K, α, K², α², Kα]."""
    return np.column_stack([
        np.ones_like(K_arr),
        K_arr,
        A_arr,
        K_arr ** 2,
        A_arr ** 2,
        K_arr * A_arr,
    ])


def fit_general_quadratic(K_arr, A_arr, labels):
    """OLS fit of ordinal labels to general quadratic features."""
    X = build_features(K_arr, A_arr)
    # OLS: beta = (X'X)^{-1} X'y
    beta = np.linalg.lstsq(X, labels, rcond=None)[0]
    pred = X @ beta
    residuals = labels - pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    return beta, pred, rmse


def fit_single_param(K_arr, A_arr, labels):
    """
    Fit C = K^a * [4*alpha*(1-alpha)]^b as a single complexity parameter,
    then fit ordinal labels as a function of C.

    Also try: C = K * (c0 + c1*alpha + c2*alpha^2)
    And:      C = K * alpha * (1-alpha)
    """
    results = {}

    # --- Form 1: C = K * alpha*(1-alpha) * scale  (1 param: just scale)
    # This is the user's intuition
    C1 = K_arr * 4 * A_arr * (1 - A_arr)
    # Fit labels = a + b*C1 via OLS
    X1 = np.column_stack([np.ones_like(C1), C1])
    b1 = np.linalg.lstsq(X1, labels, rcond=None)[0]
    pred1 = X1 @ b1
    rmse1 = np.sqrt(np.mean((labels - pred1)**2))
    results['C = K*4α(1-α)'] = (C1, pred1, rmse1, b1)

    # --- Form 2: C = K * (c0 + c1*α + c2*α²)  (3 params)
    def loss2(params):
        c0, c1, c2 = params
        C = K_arr * (c0 + c1 * A_arr + c2 * A_arr**2)
        X = np.column_stack([np.ones_like(C), C])
        b = np.linalg.lstsq(X, labels, rcond=None)[0]
        pred = X @ b
        return np.mean((labels - pred)**2)

    res2 = minimize(loss2, [1.0, -2.0, 2.0], method='Nelder-Mead')
    c0, c1, c2 = res2.x
    C2 = K_arr * (c0 + c1 * A_arr + c2 * A_arr**2)
    X2 = np.column_stack([np.ones_like(C2), C2])
    b2 = np.linalg.lstsq(X2, labels, rcond=None)[0]
    pred2 = X2 @ b2
    rmse2 = np.sqrt(np.mean((labels - pred2)**2))
    results['C = K*(c₀+c₁α+c₂α²)'] = (C2, pred2, rmse2,
                                          {'c0': c0, 'c1': c1, 'c2': c2, 'linear': b2})

    # --- Form 3: C = (K + k0) * (α - α0)^2 + K*k1   (3 params)
    def loss3(params):
        k0, a0, k1 = params
        C = (K_arr + k0) * (A_arr - a0)**2 + K_arr * k1
        X = np.column_stack([np.ones_like(C), C])
        b = np.linalg.lstsq(X, labels, rcond=None)[0]
        pred = X @ b
        return np.mean((labels - pred)**2)

    res3 = minimize(loss3, [0.0, 0.5, 0.1], method='Nelder-Mead')
    k0, a0, k1 = res3.x
    C3 = (K_arr + k0) * (A_arr - a0)**2 + K_arr * k1
    X3 = np.column_stack([np.ones_like(C3), C3])
    b3 = np.linalg.lstsq(X3, labels, rcond=None)[0]
    pred3 = X3 @ b3
    rmse3 = np.sqrt(np.mean((labels - pred3)**2))
    results['C = (K+k₀)(α-α₀)²+K·k₁'] = (C3, pred3, rmse3,
                                             {'k0': k0, 'a0': a0, 'k1': k1, 'linear': b3})

    # --- Form 4: C = K^a * [4α(1-α)]^b  (2 params, power law)
    def loss4(params):
        a, b = params
        C = np.power(K_arr, a) * np.power(4 * A_arr * (1 - A_arr) + 1e-10, b)
        X = np.column_stack([np.ones_like(C), C])
        beta = np.linalg.lstsq(X, labels, rcond=None)[0]
        pred = X @ beta
        return np.mean((labels - pred)**2)

    res4 = minimize(loss4, [1.0, 1.0], method='Nelder-Mead')
    a4, b4 = res4.x
    C4 = np.power(K_arr, a4) * np.power(4 * A_arr * (1 - A_arr) + 1e-10, b4)
    X4 = np.column_stack([np.ones_like(C4), C4])
    beta4 = np.linalg.lstsq(X4, labels, rcond=None)[0]
    pred4 = X4 @ beta4
    rmse4 = np.sqrt(np.mean((labels - pred4)**2))
    results['C = K^a·[4α(1-α)]^b'] = (C4, pred4, rmse4,
                                        {'a': a4, 'b': b4, 'linear': beta4})

    return results


def plot_winner_map(labels, fname="complexity_winner_map.pdf"):
    """Plot which regime method wins at each (K, alpha)."""
    nK, nA = labels.shape
    colors = ['#999999', '#ff7f00', '#4daf4a', '#377eb8', '#f781bf']  # TS, Plur, IRV, Borda, STAR

    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            if not np.isnan(labels[i, j]):
                from matplotlib.colors import to_rgb
                img[i, j] = to_rgb(colors[int(labels[i, j])])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)], rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title("Best method among regime progression\n(Total Score → Plurality → IRV → Borda → STAR)")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=n) for c, n in zip(colors, REGIME_NAMES)]
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def plot_iso_C(K_grid, A_grid, C_grid, labels, title, fname):
    """Plot iso-C contours over the winner map."""
    nK, nA = labels.shape
    colors = ['#999999', '#ff7f00', '#4daf4a', '#377eb8', '#f781bf']
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            if not np.isnan(labels[i, j]):
                img[i, j] = to_rgb(colors[int(labels[i, j])])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[-0.5, nA-0.5, -0.5, nK-0.5])

    # Overlay iso-C contours
    n_contours = 8
    C_min, C_max = np.nanmin(C_grid), np.nanmax(C_grid)
    levels = np.linspace(C_min, C_max, n_contours + 2)[1:-1]
    cs = ax.contour(C_grid, levels=levels, colors='white', linewidths=1.5,
                    extent=[-0.5, nA-0.5, -0.5, nK-0.5], origin='lower')
    ax.clabel(cs, fontsize=7, fmt='%.1f')

    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_FULL[i]:.2f}" for i in range(0, nA, 2)], rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(K_FULL[i]) for i in range(0, nK, 2)])
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("K", fontsize=12)
    ax.set_title(title, fontsize=12)

    handles = [Patch(facecolor=c, label=n) for c, n in zip(colors, REGIME_NAMES)]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def plot_C_vs_label(C, labels_flat, label_names, title, fname):
    """1D scatter: C value vs ordinal label, showing regime separation."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#999999', '#ff7f00', '#4daf4a', '#377eb8', '#f781bf']
    for regime_idx, (name, color) in enumerate(zip(label_names, colors)):
        mask = (labels_flat == regime_idx)
        if mask.sum() > 0:
            ax.scatter(C[mask], np.full(mask.sum(), regime_idx) + np.random.normal(0, 0.08, mask.sum()),
                       c=color, s=15, alpha=0.6, label=name, edgecolors='none')
    ax.set_xlabel("Complexity C(K, α)", fontsize=12)
    ax.set_ylabel("Regime", fontsize=12)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print("=== Complexity Parameter Fitting ===\n")

    # Load data
    mean = load_data()

    # Compute regime labels (rank-1 among regime methods only)
    labels = compute_regime_labels_v2(mean)
    print(f"\nRegime method counts:")
    for i, name in enumerate(REGIME_NAMES):
        print(f"  {name}: {np.sum(labels == i):.0f} cells")

    # Plot winner map
    print("\nPlotting winner map ...")
    plot_winner_map(labels)

    # Prepare flat arrays for fitting (exclude NaN cells)
    nK, nA = labels.shape
    K_grid = np.array([[K for _ in A_FULL] for K in K_FULL])
    A_grid = np.array([[a for a in A_FULL] for _ in K_FULL])
    valid = ~np.isnan(labels)
    K_flat = K_grid[valid].astype(float)
    A_flat = A_grid[valid].astype(float)
    L_flat = labels[valid]

    print(f"\nValid cells for fitting: {len(L_flat)}")

    # ── General quadratic fit ──
    print("\n── General Quadratic: C = β₀ + β₁K + β₂α + β₃K² + β₄α² + β₅Kα ──")
    beta, pred_gen, rmse_gen = fit_general_quadratic(K_flat, A_flat, L_flat)
    print(f"  β₀ (intercept) = {beta[0]:+.4f}")
    print(f"  β₁ (K)         = {beta[1]:+.4f}")
    print(f"  β₂ (α)         = {beta[2]:+.4f}")
    print(f"  β₃ (K²)        = {beta[3]:+.6f}")
    print(f"  β₄ (α²)        = {beta[4]:+.4f}")
    print(f"  β₅ (Kα)        = {beta[5]:+.4f}")
    print(f"  RMSE = {rmse_gen:.4f}")

    # Which terms matter most? Standardize by range
    K_range = K_flat.max() - K_flat.min()
    A_range = A_flat.max() - A_flat.min()
    contrib = np.abs(beta) * np.array([1, K_range, A_range, K_range**2, A_range**2, K_range*A_range])
    print(f"\n  Contribution (|β| × range):")
    names = ['intercept', 'K', 'α', 'K²', 'α²', 'Kα']
    for n, c in sorted(zip(names, contrib), key=lambda x: -x[1]):
        print(f"    {n:12s}: {c:.4f}")

    # General quadratic C values on grid
    C_gen_grid = np.full((nK, nA), np.nan)
    for i in range(nK):
        for j in range(nA):
            if valid[i, j]:
                x = np.array([1, K_FULL[i], A_FULL[j], K_FULL[i]**2, A_FULL[j]**2, K_FULL[i]*A_FULL[j]])
                C_gen_grid[i, j] = x @ beta

    # ── Single-parameter forms ──
    print("\n── Single-Parameter Forms ──")
    sp_results = fit_single_param(K_flat, A_flat, L_flat)
    for name, (C, pred, rmse, params) in sorted(sp_results.items(), key=lambda x: x[1][2]):
        print(f"\n  {name}:")
        if isinstance(params, dict):
            for k, v in params.items():
                if k != 'linear':
                    print(f"    {k} = {v:.4f}")
            if 'linear' in params:
                print(f"    linear map: {params['linear']}")
        else:
            print(f"    linear map: {params}")
        print(f"    RMSE = {rmse:.4f}")

    # ── Diagnostic plots ──
    print("\n── Generating plots ──")

    # General quadratic iso-C plot
    plot_iso_C(K_grid, A_grid, C_gen_grid, labels,
               f"General quadratic iso-C (RMSE={rmse_gen:.3f})",
               "complexity_isoC_general.pdf")

    # Best single-param iso-C plot
    best_name = min(sp_results, key=lambda k: sp_results[k][2])
    best_C, best_pred, best_rmse, best_params = sp_results[best_name]
    # Compute grid
    C_best_grid = np.full((nK, nA), np.nan)
    flat_idx = 0
    for i in range(nK):
        for j in range(nA):
            if valid[i, j]:
                C_best_grid[i, j] = best_C[flat_idx]
                flat_idx += 1

    plot_iso_C(K_grid, A_grid, C_best_grid, labels,
               f"Best single-param: {best_name} (RMSE={best_rmse:.3f})",
               "complexity_isoC_best.pdf")

    # 1D scatter: C vs regime for general quadratic
    C_gen_flat = np.full_like(L_flat, np.nan)
    flat_idx = 0
    for i in range(nK):
        for j in range(nA):
            if valid[i, j]:
                x = np.array([1, K_FULL[i], A_FULL[j], K_FULL[i]**2, A_FULL[j]**2, K_FULL[i]*A_FULL[j]])
                C_gen_flat[flat_idx] = x @ beta
                flat_idx += 1

    plot_C_vs_label(C_gen_flat, L_flat, REGIME_NAMES,
                    f"General quadratic C vs regime (RMSE={rmse_gen:.3f})",
                    "complexity_scatter_general.pdf")

    plot_C_vs_label(best_C, L_flat, REGIME_NAMES,
                    f"{best_name} vs regime (RMSE={best_rmse:.3f})",
                    "complexity_scatter_best.pdf")

    # For each single-param form, 1D scatter
    for name, (C, pred, rmse, params) in sp_results.items():
        safe_name = name.replace(' ', '_').replace('/', '_').replace('·', '_')
        safe_name = ''.join(c if c.isalnum() or c == '_' else '' for c in safe_name)
        plot_C_vs_label(C, L_flat, REGIME_NAMES,
                        f"{name} (RMSE={rmse:.3f})",
                        f"complexity_scatter_{safe_name}.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
