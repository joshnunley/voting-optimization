#!/usr/bin/env python3
"""
Run a single configuration of the generalized Borda family:
  Score = (1-λ)·normalized_utility + λ·borda_points
  With probability ρ, do top-2 pairwise runoff after scoring.

Usage:
  python run_borda_family.py --lam 0.5 --rho 0.3 --K 10 --alpha 0.5 \
    --runs 100 --output_dir results/borda_family/...
"""

import argparse
import os
import numpy as np
import json
from NKLandscape import NKLandscape

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, required=True, help="Cardinal vs ordinal weight [0,1]")
    p.add_argument("--rho", type=float, required=True, help="Runoff probability [0,1]")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--voting_portion", type=float, default=0.5)
    p.add_argument("--vote_size", type=int, default=2)
    p.add_argument("--num_solutions", type=int, default=100)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def generalized_borda_vote(utility_matrix, current_fitnesses, lam, rho):
    """
    Generalized Borda family vote.
    utility_matrix: (n_voters, n_proposals)
    Returns: winning proposal index
    """
    n_voters, n_proposals = utility_matrix.shape

    # Borda points per voter
    ranks = np.argsort(np.argsort(utility_matrix, axis=1), axis=1)  # 0=worst, n-1=best
    borda = ranks.astype(float)

    # Normalize utility to [0,1] per voter for mixing
    u_min = utility_matrix.min(axis=1, keepdims=True)
    u_max = utility_matrix.max(axis=1, keepdims=True)
    u_range = u_max - u_min
    u_range[u_range < 1e-10] = 1.0
    u_norm = (utility_matrix - u_min) / u_range

    # Normalize borda to [0,1]
    if n_proposals > 1:
        b_norm = borda / (n_proposals - 1)
    else:
        b_norm = np.ones_like(borda)

    # Combined score
    combined = (1.0 - lam) * u_norm + lam * b_norm
    total_scores = combined.sum(axis=0)

    # Runoff decision
    if rho > 0 and np.random.random() < rho and n_proposals >= 2:
        # Top-2 by combined score
        top2 = np.argsort(total_scores)[-2:]
        # Pairwise comparison using raw utilities
        prefer_a = np.sum(utility_matrix[:, top2[1]] > utility_matrix[:, top2[0]])
        prefer_b = np.sum(utility_matrix[:, top2[0]] > utility_matrix[:, top2[1]])
        return int(top2[1]) if prefer_a >= prefer_b else int(top2[0])
    else:
        return int(np.argmax(total_scores))


def decimal_to_binary(decimal, length):
    binary = list(map(int, bin(decimal)[2:]))
    return np.array([0] * (length - len(binary)) + binary, dtype=int)


def run_single(nk, initial_solutions, vote_indices, vote_size, lam, rho, iterations):
    solutions = initial_solutions.copy()
    num_solutions = solutions.shape[0]
    N = solutions.shape[1]

    mean_history = []
    var_history = []

    for _ in range(iterations):
        # Generate vote indices
        np.random.shuffle(vote_indices)
        prop_indices = vote_indices[:vote_size]

        # Calculate proposal fitnesses
        n_proposals = 2 ** vote_size
        fitnesses = np.zeros((num_solutions, n_proposals))
        for p in range(n_proposals):
            proposed = solutions.copy()
            proposed[:, prop_indices] = decimal_to_binary(p, vote_size)
            fitnesses[:, p] = nk.calculate_fitness_batch(proposed)

        current_fitnesses = nk.calculate_fitness_batch(solutions)

        # Vote
        winner_idx = generalized_borda_vote(fitnesses, current_fitnesses, lam, rho)
        winner = decimal_to_binary(winner_idx, vote_size)

        # Update
        solutions[:, prop_indices] = winner
        solutions = np.unique(solutions, axis=0)
        num_solutions = solutions.shape[0]

        f = nk.calculate_fitness_batch(solutions)
        mean_history.append(float(np.mean(f)))
        var_history.append(float(np.var(f)))

    return np.array(mean_history), np.array(var_history)


def main():
    args = parse_args()
    if args.iterations is None:
        args.iterations = 150 + 50 * args.K

    os.makedirs(args.output_dir, exist_ok=True)

    N = args.N
    n_vote = int(args.voting_portion * N)

    all_mean = []
    all_var = []

    for run in range(args.runs):
        rng = np.random.RandomState(args.seed + run * 10000)
        idx = np.arange(N)
        rng.shuffle(idx)
        vote_idx = idx[:n_vote]
        non_vote_idx = idx[n_vote:]

        solutions = np.zeros((args.num_solutions, N), dtype=int)
        voting_bits = rng.randint(2, size=n_vote)
        solutions[:, vote_idx] = voting_bits
        solutions[:, non_vote_idx] = rng.randint(2, size=(args.num_solutions, N - n_vote))

        np.random.seed(args.seed + run * 10000 + 1000)
        nk = NKLandscape(N, args.K)
        if args.K > 0:
            dep = NKLandscape.build_split_dependency_matrix(
                N, args.K, vote_idx, non_vote_idx, args.alpha)
            nk.set_dependency_matrix(dep)
        np.random.seed(args.seed + run * 10000 + 2000)

        mh, vh = run_single(nk, solutions, vote_idx.copy(), args.vote_size,
                            args.lam, args.rho, args.iterations)
        all_mean.append(mh)
        all_var.append(vh)

    np.save(os.path.join(args.output_dir, "mean_history.npy"), np.array(all_mean))
    np.save(os.path.join(args.output_dir, "variance_history.npy"), np.array(all_var))

    metadata = {
        "lam": args.lam, "rho": args.rho,
        "K": args.K, "alpha": args.alpha,
        "N": args.N, "voting_portion": args.voting_portion,
        "vote_size": args.vote_size, "num_solutions": args.num_solutions,
        "runs": args.runs, "iterations": args.iterations, "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
