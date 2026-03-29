#!/usr/bin/env python3
"""
Generalized positional scoring rule with power-law exponent p.
Score for rank k (0=best) out of n proposals: s(k) = ((n-1-k)/(n-1))^p

p=1: Borda.  p→∞: Plurality.  p→0: Antiplurality.

Usage:
  python run_scoring_power.py --p 1.0 --K 10 --alpha 0.5 --runs 100 --output_dir ...
"""

import argparse, os, json
import numpy as np
from NKLandscape import NKLandscape


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("--p", type=float, required=True)
    a.add_argument("--K", type=int, required=True)
    a.add_argument("--alpha", type=float, required=True)
    a.add_argument("--N", type=int, default=50)
    a.add_argument("--voting_portion", type=float, default=0.5)
    a.add_argument("--vote_size", type=int, default=2)
    a.add_argument("--num_solutions", type=int, default=100)
    a.add_argument("--runs", type=int, default=100)
    a.add_argument("--iterations", type=int, default=None)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--output_dir", type=str, required=True)
    return a.parse_args()


def decimal_to_binary(d, length):
    b = list(map(int, bin(d)[2:]))
    return np.array([0] * (length - len(b)) + b, dtype=int)


def power_scoring_vote(utility_matrix, p):
    """Positional scoring: s(k) = ((n-1-k)/(n-1))^p, sum across voters."""
    n_voters, n_proposals = utility_matrix.shape
    # Rank each voter's proposals: 0=best, n-1=worst
    ranks = np.argsort(np.argsort(-utility_matrix, axis=1), axis=1)
    # Scoring
    if n_proposals > 1:
        scores = ((n_proposals - 1 - ranks) / (n_proposals - 1)) ** p
    else:
        scores = np.ones_like(ranks, dtype=float)
    return int(np.argmax(scores.sum(axis=0)))


def run_single(nk, solutions, vote_indices, vote_size, p, iterations):
    solutions = solutions.copy()
    num_solutions = solutions.shape[0]
    mean_hist, var_hist = [], []

    for _ in range(iterations):
        np.random.shuffle(vote_indices)
        prop_idx = vote_indices[:vote_size]
        n_prop = 2 ** vote_size

        fitnesses = np.zeros((num_solutions, n_prop))
        for pr in range(n_prop):
            proposed = solutions.copy()
            proposed[:, prop_idx] = decimal_to_binary(pr, vote_size)
            fitnesses[:, pr] = nk.calculate_fitness_batch(proposed)

        winner_idx = power_scoring_vote(fitnesses, p)
        solutions[:, prop_idx] = decimal_to_binary(winner_idx, vote_size)
        solutions = np.unique(solutions, axis=0)
        num_solutions = solutions.shape[0]

        f = nk.calculate_fitness_batch(solutions)
        mean_hist.append(float(np.mean(f)))
        var_hist.append(float(np.var(f)))

    return np.array(mean_hist), np.array(var_hist)


def main():
    args = parse_args()
    if args.iterations is None:
        args.iterations = 150 + 50 * args.K

    os.makedirs(args.output_dir, exist_ok=True)
    N = args.N
    n_vote = int(args.voting_portion * N)

    all_mean, all_var = [], []
    for run in range(args.runs):
        rng = np.random.RandomState(args.seed + run * 10000)
        idx = np.arange(N); rng.shuffle(idx)
        vote_idx = idx[:n_vote]
        non_vote_idx = idx[n_vote:]

        sol = np.zeros((args.num_solutions, N), dtype=int)
        sol[:, vote_idx] = rng.randint(2, size=n_vote)
        sol[:, non_vote_idx] = rng.randint(2, size=(args.num_solutions, N - n_vote))

        np.random.seed(args.seed + run * 10000 + 1000)
        nk = NKLandscape(N, args.K)
        if args.K > 0:
            dep = NKLandscape.build_split_dependency_matrix(
                N, args.K, vote_idx, non_vote_idx, args.alpha)
            nk.set_dependency_matrix(dep)
        np.random.seed(args.seed + run * 10000 + 2000)

        mh, vh = run_single(nk, sol, vote_idx.copy(), args.vote_size, args.p, args.iterations)
        all_mean.append(mh)
        all_var.append(vh)

    np.save(os.path.join(args.output_dir, "mean_history.npy"), np.array(all_mean))
    np.save(os.path.join(args.output_dir, "variance_history.npy"), np.array(all_var))
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
