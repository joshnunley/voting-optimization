#!/usr/bin/env python3
"""
Plot spectral-entropy curves across K0 for three related profile matrices.

Important note:
  For raw matrices, U, U^T U, and U U^T have the same nonzero spectrum, so
  their spectral entropies are identical. To get an informative three-curve
  comparison, this script plots:

  1. raw profile spectral entropy          : singular-value energy entropy of U
  2. candidate-centered spectral entropy   : eigenvalue entropy of (U-U_colmean)^T (U-U_colmean)
  3. voter-centered spectral entropy       : eigenvalue entropy of (U-U_rowmean) (U-U_rowmean)^T

Each K0 value is averaged across samples, then each curve is normalized to
[0, 1] across K0 for direct shape comparison.
"""

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot spectral entropy curves across complexity")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/raw_profile_matrix_analysis.npz",
    )
    p.add_argument(
        "--output",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/spectral_entropy/spectral_entropy_curves.pdf",
    )
    return p.parse_args()


def spectral_entropy_from_weights(w):
    w = np.asarray(w, dtype=float)
    w = w[w > 1e-15]
    if len(w) == 0:
        return 0.0
    p = w / np.sum(w)
    return float(-np.sum(p * np.log(p)))


def normalized_curve(y):
    y = np.asarray(y, dtype=float)
    lo = np.min(y)
    hi = np.max(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def entropy_raw_profile(U):
    svals = np.linalg.svd(U, compute_uv=False)
    return spectral_entropy_from_weights(np.square(svals))


def entropy_candidate_centered(U):
    X = U - U.mean(axis=0, keepdims=True)
    vals = np.linalg.eigvalsh(X.T @ X)
    vals = np.clip(vals, 0.0, None)
    return spectral_entropy_from_weights(vals)


def entropy_voter_centered(U):
    X = U - U.mean(axis=1, keepdims=True)
    vals = np.linalg.eigvalsh(X @ X.T)
    vals = np.clip(vals, 0.0, None)
    return spectral_entropy_from_weights(vals)


def main():
    args = parse_args()
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    matrices = d["matrices"].astype(float)
    k0_values = d["K0_values"].astype(int)

    n_k0, n_samples = matrices.shape[:2]
    H_raw = np.zeros((n_k0, n_samples), dtype=float)
    H_cand = np.zeros((n_k0, n_samples), dtype=float)
    H_voter = np.zeros((n_k0, n_samples), dtype=float)

    for i in range(n_k0):
        for j in range(n_samples):
            U = matrices[i, j]
            H_raw[i, j] = entropy_raw_profile(U)
            H_cand[i, j] = entropy_candidate_centered(U)
            H_voter[i, j] = entropy_voter_centered(U)

    raw_mean = H_raw.mean(axis=1)
    cand_mean = H_cand.mean(axis=1)
    voter_mean = H_voter.mean(axis=1)

    raw_norm = normalized_curve(raw_mean)
    cand_norm = normalized_curve(cand_mean)
    voter_norm = normalized_curve(voter_mean)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(k0_values, raw_norm, color="#377eb8", linewidth=2.2, label="raw U entropy")
    ax.plot(k0_values, cand_norm, color="#4daf4a", linewidth=2.2, label="candidate-centered entropy")
    ax.plot(k0_values, voter_norm, color="#e41a1c", linewidth=2.2, label="voter-centered entropy")
    ax.set_xlabel("K0")
    ax.set_ylabel("normalized spectral entropy")
    ax.set_title("Spectral entropy curves across complexity")
    ax.set_xticks(k0_values)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Mean raw-vs-candidate entropy abs diff:", float(np.mean(np.abs(raw_mean - cand_mean))))
    print("Mean raw-vs-voter entropy abs diff:", float(np.mean(np.abs(raw_mean - voter_mean))))
    print(f"saved {outpath}")


if __name__ == "__main__":
    main()
