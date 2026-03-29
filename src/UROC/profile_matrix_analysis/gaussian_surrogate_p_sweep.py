#!/usr/bin/env python3
"""
Fit a Gaussian surrogate N(mu(K0), Sigma(K0)) from the raw NKalpha profile
matrices, then evaluate the one-shot welfare-optimal power-scoring parameter p
on synthetic score matrices drawn from that surrogate.

Also computes the same p-sweep directly on the empirical raw matrices for
comparison.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian surrogate p-sweep from raw profile matrices")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/raw_profile_matrix_analysis.npz",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/gaussian_surrogate_p_sweep",
    )
    p.add_argument("--num_synth_matrices", type=int, default=300)
    p.add_argument("--p_min", type=float, default=0.1)
    p.add_argument("--p_max", type=float, default=2.5)
    p.add_argument("--p_steps", type=int, default=49)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def power_scoring_winner(utility_matrix, p):
    n_voters, n_proposals = utility_matrix.shape
    ranks = np.argsort(np.argsort(-utility_matrix, axis=1), axis=1)
    if n_proposals > 1:
        scores = ((n_proposals - 1 - ranks) / (n_proposals - 1)) ** p
    else:
        scores = np.ones_like(ranks, dtype=float)
    return int(np.argmax(scores.sum(axis=0)))


def winner_welfare(U, p):
    w = power_scoring_winner(U, p)
    return float(np.mean(U[:, w]))


def participation_from_cov(cov):
    vals = np.linalg.eigvalsh(cov)
    vals = np.clip(vals, 0.0, None)
    s = np.sum(vals)
    if s <= 1e-15:
        return 0.0
    probs = vals / s
    return float(1.0 / np.sum(probs ** 2))


def fit_gaussian_params(matrices_by_k0):
    """
    matrices_by_k0: (n_k0, n_samples, n_voters, n_candidates)
    """
    n_k0 = matrices_by_k0.shape[0]
    n_candidates = matrices_by_k0.shape[-1]
    mus = np.zeros((n_k0, n_candidates), dtype=float)
    covs = np.zeros((n_k0, n_candidates, n_candidates), dtype=float)
    cov_parts = np.zeros(n_k0, dtype=float)
    for i in range(n_k0):
        X = matrices_by_k0[i].reshape(-1, n_candidates)
        mus[i] = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, bias=True)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-10, None)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        covs[i] = cov
        cov_parts[i] = participation_from_cov(cov)
    return mus, covs, cov_parts


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    matrices = d["matrices"].astype(float)
    k0_values = d["K0_values"].astype(int)
    n_k0, n_empirical, n_voters, n_candidates = matrices.shape

    p_grid = np.linspace(args.p_min, args.p_max, args.p_steps)
    rng = np.random.RandomState(args.seed)

    mus, covs, cov_parts = fit_gaussian_params(matrices)

    empirical_mean_welfare = np.zeros((n_k0, len(p_grid)), dtype=float)
    empirical_best_p = np.zeros(n_k0, dtype=float)

    gaussian_mean_welfare = np.zeros((n_k0, len(p_grid)), dtype=float)
    gaussian_best_p = np.zeros(n_k0, dtype=float)

    for i, K0 in enumerate(k0_values):
        # Empirical matrices
        for j, p in enumerate(p_grid):
            vals = [winner_welfare(matrices[i, s], p) for s in range(n_empirical)]
            empirical_mean_welfare[i, j] = float(np.mean(vals))
        empirical_best_p[i] = float(p_grid[int(np.argmax(empirical_mean_welfare[i]))])

        # Gaussian surrogate matrices
        synth = rng.multivariate_normal(
            mean=mus[i],
            cov=covs[i],
            size=(args.num_synth_matrices, n_voters),
            check_valid="warn",
        )
        synth = synth.reshape(args.num_synth_matrices, n_voters, n_candidates)
        for j, p in enumerate(p_grid):
            vals = [winner_welfare(synth[s], p) for s in range(args.num_synth_matrices)]
            gaussian_mean_welfare[i, j] = float(np.mean(vals))
        gaussian_best_p[i] = float(p_grid[int(np.argmax(gaussian_mean_welfare[i]))])
        print(f"K0={K0:2d}: empirical best p={empirical_best_p[i]:.3f}, gaussian best p={gaussian_best_p[i]:.3f}", flush=True)

    rows = []
    for i, K0 in enumerate(k0_values):
        rows.append({
            "K0": int(K0),
            "cov_participation": float(cov_parts[i]),
            "empirical_best_p": float(empirical_best_p[i]),
            "gaussian_best_p": float(gaussian_best_p[i]),
        })
    write_csv(
        outdir / "optimal_p_by_K0.csv",
        rows,
        ["K0", "cov_participation", "empirical_best_p", "gaussian_best_p"],
    )

    np.savez_compressed(
        outdir / "gaussian_surrogate_p_sweep.npz",
        k0_values=k0_values,
        p_grid=p_grid,
        mu=mus,
        cov=covs,
        cov_participation=cov_parts,
        empirical_mean_welfare=empirical_mean_welfare,
        empirical_best_p=empirical_best_p,
        gaussian_mean_welfare=gaussian_mean_welfare,
        gaussian_best_p=gaussian_best_p,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(k0_values, empirical_best_p, color="#377eb8", linewidth=2.2, label="Empirical raw-matrix best p")
    ax.plot(k0_values, gaussian_best_p, color="#e41a1c", linewidth=2.2, label="Gaussian surrogate best p")
    ax.set_xlabel("K0")
    ax.set_ylabel("best p")
    ax.set_title("Welfare-optimal p vs K0: empirical matrices vs Gaussian surrogate")
    ax.set_xticks(k0_values)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "optimal_p_vs_K0.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im0 = axes[0].imshow(
        empirical_mean_welfare.T, aspect="auto", origin="lower",
        extent=[k0_values[0], k0_values[-1], p_grid[0], p_grid[-1]], cmap="viridis"
    )
    axes[0].set_title("Empirical raw matrices")
    axes[0].set_xlabel("K0")
    axes[0].set_ylabel("p")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(
        gaussian_mean_welfare.T, aspect="auto", origin="lower",
        extent=[k0_values[0], k0_values[-1], p_grid[0], p_grid[-1]], cmap="viridis"
    )
    axes[1].set_title("Gaussian surrogate")
    axes[1].set_xlabel("K0")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.suptitle("Mean winner welfare across p")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "welfare_heatmaps.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {outdir / 'gaussian_surrogate_p_sweep.npz'}")
    print(f"saved {outdir / 'optimal_p_vs_K0.pdf'}")
    print(f"saved {outdir / 'welfare_heatmaps.pdf'}")


if __name__ == "__main__":
    main()
