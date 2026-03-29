#!/usr/bin/env python3
"""
Sample raw voter-by-candidate utility matrices U from the direct-democracy
NKalpha model along the fixed paper iso-complexity curves, then compute
matrix-level diagnostics for regime analysis.

The raw matrix U here is the proposal-fitness matrix for a single NKalpha draw:
    shape = (num_solutions, 2**vote_size)

For each K0 in 1..20 we:
1. Choose exact on-curve (K, alpha) points using the fixed paper formula.
2. If fewer than `samples_per_k0` unique exact points exist on the integer NK
   grid, reuse them with fresh random seeds until we reach `samples_per_k0`
   sampled matrices.
3. Generate one raw U matrix per sample.
4. Compute invariant matrix diagnostics and save both matrices and features.

This is meant to reveal complexity-dependent structure in the raw score
profiles, independent of any voting rule.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
UROC_DIR = THIS_DIR.parent
if str(UROC_DIR) not in sys.path:
    sys.path.insert(0, str(UROC_DIR))

from NKLandscape import NKLandscape
from sample_complexity_curve import PAPER_A, PAPER_B, sample_iso_complexity_points


def parse_args():
    p = argparse.ArgumentParser(description="Analyze raw NKalpha voter-candidate matrices by K0 regime")
    p.add_argument("--output_dir", type=str,
                   default=str(UROC_DIR / "results" / "profile_matrix_analysis" / "raw_u_regime_scan"),
                   help="Directory for outputs")
    p.add_argument("--samples_per_k0", type=int, default=10,
                   help="Number of raw U matrices to sample per K0")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--voting_portion", type=float, default=0.5)
    p.add_argument("--vote_size", type=int, default=2)
    p.add_argument("--num_solutions", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for reproducibility")
    p.add_argument("--K0_min", type=int, default=1)
    p.add_argument("--K0_max", type=int, default=20)
    return p.parse_args()


def decimal_to_binary(d: int, length: int) -> np.ndarray:
    bits = list(map(int, bin(d)[2:]))
    return np.array([0] * (length - len(bits)) + bits, dtype=int)


def build_initial_solutions(N: int, num_solutions: int, voting_portion: float, seed: int):
    rng = np.random.RandomState(seed)
    n_voting = int(voting_portion * N)
    indices = np.arange(N)
    rng.shuffle(indices)
    vote_indices = indices[:n_voting]
    non_vote_indices = indices[n_voting:]

    solutions = np.zeros((num_solutions, N), dtype=int)
    shared_bits = rng.randint(2, size=n_voting)
    solutions[:, vote_indices] = shared_bits
    solutions[:, non_vote_indices] = rng.randint(2, size=(num_solutions, N - n_voting))
    return solutions, vote_indices, non_vote_indices


def generate_raw_utility_matrix(
    N: int,
    K: int,
    alpha: float,
    voting_portion: float,
    vote_size: int,
    num_solutions: int,
    seed: int,
) -> np.ndarray:
    """
    Generate one raw voter-by-candidate utility matrix U from the direct
    democracy model before any voting rule is applied.
    """
    solutions, vote_indices, non_vote_indices = build_initial_solutions(
        N, num_solutions, voting_portion, seed
    )

    np.random.seed(seed + 1000)
    nk = NKLandscape(N, K)
    if K > 0:
        dep = NKLandscape.build_split_dependency_matrix(
            N, K, vote_indices, non_vote_indices, alpha
        )
        nk.set_dependency_matrix(dep)

    rng = np.random.RandomState(seed + 2000)
    proposal_indices = vote_indices.copy()
    rng.shuffle(proposal_indices)
    proposal_indices = proposal_indices[:vote_size]

    n_prop = 2 ** vote_size
    utility = np.zeros((num_solutions, n_prop), dtype=float)
    for pr in range(n_prop):
        proposed = solutions.copy()
        proposed[:, proposal_indices] = decimal_to_binary(pr, vote_size)
        utility[:, pr] = nk.calculate_fitness_batch(proposed)
    return utility


def safe_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = np.sum(x)
    if total <= 0:
        return np.full_like(x, 1.0 / len(x))
    return x / total


def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def effective_rank_from_svals(svals: np.ndarray) -> float:
    energy = safe_prob(np.square(svals))
    return float(np.exp(shannon_entropy(energy)))


def pairwise_margin_matrix(U: np.ndarray) -> np.ndarray:
    return (U[:, :, None] > U[:, None, :]).sum(axis=0) - (U[:, :, None] < U[:, None, :]).sum(axis=0)


def condorcet_winner_exists(M: np.ndarray) -> float:
    n = M.shape[0]
    for a in range(n):
        ok = True
        for b in range(n):
            if a == b:
                continue
            if M[a, b] <= 0:
                ok = False
                break
        if ok:
            return 1.0
    return 0.0


def condorcet_cycle_exists(M: np.ndarray) -> float:
    n = M.shape[0]
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                if M[a, b] > 0 and M[b, c] > 0 and M[c, a] > 0:
                    return 1.0
    return 0.0


def centered_candidate_gram(U: np.ndarray) -> np.ndarray:
    X = U - U.mean(axis=0, keepdims=True)
    return X.T @ X / max(1, U.shape[0])


def centered_voter_gram_eigvals(U: np.ndarray) -> np.ndarray:
    X = U - U.mean(axis=1, keepdims=True)
    gram_small = X.T @ X / max(1, U.shape[0])
    vals = np.linalg.eigvalsh(gram_small)
    vals = np.clip(vals, 0.0, None)
    return np.sort(vals)[::-1]


def compute_feature_dict(U: np.ndarray) -> Dict[str, float]:
    n_voters, n_candidates = U.shape
    sorted_rows = np.sort(U, axis=1)[:, ::-1]
    row_gaps = sorted_rows[:, :-1] - sorted_rows[:, 1:]
    top_choices = np.argmax(U, axis=1)
    first_place_counts = np.bincount(top_choices, minlength=n_candidates).astype(float)
    first_place_share = first_place_counts / float(n_voters)
    sorted_first_share = np.sort(first_place_share)[::-1]

    col_means = U.mean(axis=0)
    sorted_col_means = np.sort(col_means)[::-1]
    col_stds = U.std(axis=0)
    sorted_col_stds = np.sort(col_stds)[::-1]

    svals = np.linalg.svd(U, compute_uv=False)
    s_share = safe_prob(np.square(svals))
    cand_gram = centered_candidate_gram(U)
    cand_eigs = np.linalg.eigvalsh(cand_gram)
    cand_eigs = np.clip(np.sort(cand_eigs)[::-1], 0.0, None)
    cand_share = safe_prob(cand_eigs)

    voter_eigs = centered_voter_gram_eigvals(U)
    voter_share = safe_prob(voter_eigs)

    M = pairwise_margin_matrix(U)
    upper = np.abs(M[np.triu_indices(n_candidates, k=1)])
    norm_margin = upper / float(n_voters)

    feature_dict = {
        "u_mean": float(np.mean(U)),
        "u_std": float(np.std(U)),
        "u_range_mean_row": float(np.mean(sorted_rows[:, 0] - sorted_rows[:, -1])),
        "row_gap_1_mean": float(np.mean(row_gaps[:, 0])),
        "row_gap_2_mean": float(np.mean(row_gaps[:, 1])),
        "row_gap_3_mean": float(np.mean(row_gaps[:, 2])),
        "row_gap_1_std": float(np.std(row_gaps[:, 0])),
        "row_gap_2_std": float(np.std(row_gaps[:, 1])),
        "row_gap_3_std": float(np.std(row_gaps[:, 2])),
        "row_top_gap_share": float(np.mean(row_gaps[:, 0] / np.maximum(sorted_rows[:, 0] - sorted_rows[:, -1], 1e-12))),
        "first_place_entropy": float(shannon_entropy(first_place_share)),
        "first_place_top_share": float(sorted_first_share[0]),
        "first_place_second_share": float(sorted_first_share[1]),
        "col_mean_top_gap": float(sorted_col_means[0] - sorted_col_means[1]),
        "col_mean_std": float(np.std(col_means)),
        "col_std_mean": float(np.mean(col_stds)),
        "col_std_top_gap": float(sorted_col_stds[0] - sorted_col_stds[1]),
        "sv_energy_1": float(s_share[0]),
        "sv_energy_2": float(s_share[1] if len(s_share) > 1 else 0.0),
        "sv_energy_3": float(s_share[2] if len(s_share) > 2 else 0.0),
        "sv_energy_4": float(s_share[3] if len(s_share) > 3 else 0.0),
        "effective_rank": float(effective_rank_from_svals(svals)),
        "cand_eig_1": float(cand_share[0]),
        "cand_eig_2": float(cand_share[1] if len(cand_share) > 1 else 0.0),
        "cand_effective_rank": float(np.exp(shannon_entropy(cand_share))),
        "voter_eig_1": float(voter_share[0]),
        "voter_eig_2": float(voter_share[1] if len(voter_share) > 1 else 0.0),
        "voter_effective_rank": float(np.exp(shannon_entropy(voter_share))),
        "pairwise_margin_mean_abs": float(np.mean(norm_margin)),
        "pairwise_margin_min_abs": float(np.min(norm_margin)),
        "pairwise_margin_std_abs": float(np.std(norm_margin)),
        "condorcet_winner_exists": float(condorcet_winner_exists(M)),
        "condorcet_cycle_exists": float(condorcet_cycle_exists(M)),
    }
    return feature_dict


def write_feature_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return (X - mu) / sigma, mu, sigma


def pca_projection(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    Xz, _, _ = standardize(X)
    _, svals, vt = np.linalg.svd(Xz, full_matrices=False)
    coords = Xz @ vt[:n_components].T
    var = np.square(svals)
    var_ratio = var / np.sum(var)
    return coords, var_ratio[:n_components]


def make_pca_plot(coords: np.ndarray, K0_arr: np.ndarray, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=K0_arr, cmap="viridis", s=45, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("K0")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_feature_summary_plot(
    feature_names: Sequence[str],
    features: np.ndarray,
    K0_values: Sequence[int],
    outpath: Path,
) -> None:
    n_features = len(feature_names)
    ncols = 4
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for j, feat in enumerate(feature_names):
        ax = axes_flat[j]
        means = features[:, :, j].mean(axis=1)
        stds = features[:, :, j].std(axis=1)
        ax.plot(K0_values, means, color="#377eb8", linewidth=1.6)
        ax.fill_between(K0_values, means - stds, means + stds, color="#377eb8", alpha=0.2)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n_features, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Raw profile-matrix feature summaries by K0", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_representative_matrix_plot(
    matrices: np.ndarray,
    K_vals: np.ndarray,
    alpha_vals: np.ndarray,
    K0_values: Sequence[int],
    outpath: Path,
) -> None:
    picks = []
    for K0 in [min(K0_values), int(np.median(K0_values)), max(K0_values)]:
        idx = list(K0_values).index(K0)
        picks.append((idx, 0))

    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 4))
    if len(picks) == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, picks):
        im = ax.imshow(matrices[i, j], aspect="auto", cmap="coolwarm")
        ax.set_title(f"K0={K0_values[i]}, K={int(K_vals[i, j])}, a={alpha_vals[i, j]:.3f}", fontsize=9)
        ax.set_xlabel("candidate")
        ax.set_ylabel("voter")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Representative raw utility matrices U", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    K0_values = list(range(args.K0_min, args.K0_max + 1))
    n_k0 = len(K0_values)
    n_candidates = 2 ** args.vote_size

    matrices = np.zeros((n_k0, args.samples_per_k0, args.num_solutions, n_candidates), dtype=float)
    K_used = np.zeros((n_k0, args.samples_per_k0), dtype=int)
    alpha_used = np.zeros((n_k0, args.samples_per_k0), dtype=float)
    arc_used = np.zeros((n_k0, args.samples_per_k0), dtype=float)
    reused_point_index = np.zeros((n_k0, args.samples_per_k0), dtype=int)
    unique_points_available = np.zeros(n_k0, dtype=int)

    feature_rows: List[Dict[str, float]] = []
    feature_names: List[str] = []

    for i, K0 in enumerate(K0_values):
        base_points = sample_iso_complexity_points(K0, num_points=args.samples_per_k0, integer_K=True)
        if not base_points:
            raise RuntimeError(f"No valid exact on-curve integer-K points found for K0={K0}")
        unique_points_available[i] = len(base_points)

        for j in range(args.samples_per_k0):
            pidx = j % len(base_points)
            point = base_points[pidx]
            seed = args.seed + i * 100000 + j * 1000
            U = generate_raw_utility_matrix(
                N=args.N,
                K=int(round(point.K)),
                alpha=float(point.alpha),
                voting_portion=args.voting_portion,
                vote_size=args.vote_size,
                num_solutions=args.num_solutions,
                seed=seed,
            )
            matrices[i, j] = U
            K_used[i, j] = int(round(point.K))
            alpha_used[i, j] = float(point.alpha)
            arc_used[i, j] = float(point.arc_fraction)
            reused_point_index[i, j] = pidx

            feats = compute_feature_dict(U)
            if not feature_names:
                feature_names = list(feats.keys())
            row = {
                "K0": int(K0),
                "sample_idx": int(j),
                "point_idx": int(pidx),
                "seed": int(seed),
                "K": int(round(point.K)),
                "alpha": float(point.alpha),
                "arc_fraction": float(point.arc_fraction),
                "unique_points_available": int(len(base_points)),
            }
            row.update(feats)
            feature_rows.append(row)
        print(f"K0={K0:2d}: generated {args.samples_per_k0} matrices from {len(base_points)} exact curve points", flush=True)

    feature_matrix = np.array([[row[name] for name in feature_names] for row in feature_rows], dtype=float)
    features_by_k0 = feature_matrix.reshape(n_k0, args.samples_per_k0, len(feature_names))
    K0_per_sample = np.array([row["K0"] for row in feature_rows], dtype=int)

    coords, var_ratio = pca_projection(feature_matrix, n_components=2)

    np.savez_compressed(
        outdir / "raw_profile_matrix_analysis.npz",
        matrices=matrices,
        K0_values=np.array(K0_values, dtype=int),
        K_used=K_used,
        alpha_used=alpha_used,
        arc_used=arc_used,
        reused_point_index=reused_point_index,
        unique_points_available=unique_points_available,
        feature_names=np.array(feature_names, dtype=object),
        feature_matrix=feature_matrix,
        features_by_k0=features_by_k0,
        K0_per_sample=K0_per_sample,
        pca_coords=coords,
        pca_var_ratio=var_ratio,
        N=np.array(args.N),
        voting_portion=np.array(args.voting_portion),
        vote_size=np.array(args.vote_size),
        num_solutions=np.array(args.num_solutions),
        samples_per_k0=np.array(args.samples_per_k0),
        curve_a=np.array(PAPER_A),
        curve_b=np.array(PAPER_B),
    )

    csv_fieldnames = [
        "K0", "sample_idx", "point_idx", "seed", "K", "alpha", "arc_fraction",
        "unique_points_available"
    ] + feature_names
    write_feature_csv(outdir / "feature_table.csv", feature_rows, csv_fieldnames)

    make_pca_plot(
        coords,
        K0_per_sample,
        figdir / "pca_by_K0.pdf",
        title=f"Raw U feature PCA by K0 (var={var_ratio[0]:.2f}, {var_ratio[1]:.2f})",
    )
    make_feature_summary_plot(
        feature_names,
        features_by_k0,
        K0_values,
        figdir / "feature_summary_by_K0.pdf",
    )
    make_representative_matrix_plot(
        matrices,
        K_used,
        alpha_used,
        K0_values,
        figdir / "representative_matrices.pdf",
    )

    print(f"saved {outdir / 'raw_profile_matrix_analysis.npz'}", flush=True)
    print(f"saved {outdir / 'feature_table.csv'}", flush=True)
    print(f"saved {figdir / 'pca_by_K0.pdf'}", flush=True)
    print(f"saved {figdir / 'feature_summary_by_K0.pdf'}", flush=True)
    print(f"saved {figdir / 'representative_matrices.pdf'}", flush=True)


if __name__ == "__main__":
    main()
