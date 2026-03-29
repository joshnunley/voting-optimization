#!/usr/bin/env python3
"""
Generate sparse sample data for a candidate voting method along the fixed
paper iso-complexity curves.

For each integer K0 in 1..20:
  - sample a configurable number of interior points along the published
    iso-complexity curve using `sample_complexity_curve.py`
  - run the candidate method at those (K, alpha) points
  - save the sampled terminal mean/variance data into a single .npz file

This file is intentionally just the data-generation stage. Ranking against the
8-method baseline and plotting are handled separately.
"""

import argparse
import os
import numpy as np

from NKLandscape import NKLandscape
from VoteModel import VoteModel
from sample_complexity_curve import PAPER_A, PAPER_B, sample_all_K0


TERMINAL_WINDOW = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_npz", required=True,
                   help="Path to output .npz file")
    p.add_argument("--vote_type", default=None,
                   help="Candidate method name passed to VoteModel")
    p.add_argument("--scoring_power_p", type=float, default=None,
                   help="If set, run the generalized positional power rule with this exponent p")
    p.add_argument("--num_points", type=int, default=4,
                   help="Interior samples per K0 (default: 4)")
    p.add_argument("--runs", type=int, default=100,
                   help="Number of stochastic runs per sampled point")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--voting_portion", type=float, default=0.5)
    p.add_argument("--vote_size", type=int, default=2)
    p.add_argument("--num_solutions", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations_base", type=int, default=150,
                   help="Default iteration schedule intercept")
    p.add_argument("--iterations_slope", type=int, default=50,
                   help="Default iteration schedule slope in K")
    return p.parse_args()


def build_initial_solutions(N, num_solutions, voting_portion, seed):
    rng = np.random.RandomState(seed)
    n_voting = int(voting_portion * N)
    indices = np.arange(N)
    rng.shuffle(indices)
    vote_indices = indices[:n_voting]
    non_vote_indices = indices[n_voting:]

    solutions = np.zeros((num_solutions, N), dtype=int)
    voting_bits = rng.randint(2, size=n_voting)
    solutions[:, vote_indices] = voting_bits
    non_voting_bits = rng.randint(2, size=(num_solutions, N - n_voting))
    solutions[:, non_vote_indices] = non_voting_bits
    return solutions, vote_indices, non_vote_indices


def decimal_to_binary(decimal, length):
    binary = list(map(int, bin(decimal)[2:]))
    return np.array([0] * (length - len(binary)) + binary, dtype=int)


def power_scoring_vote(utility_matrix, p):
    """
    Generalized positional scoring rule:
      score(rank k) = ((n-1-k)/(n-1))^p, with k=0 best.
    p=1 is Borda, p→∞ approaches plurality, p<1 flattens Borda.
    """
    n_voters, n_proposals = utility_matrix.shape
    ranks = np.argsort(np.argsort(-utility_matrix, axis=1), axis=1)
    if n_proposals > 1:
        scores = ((n_proposals - 1 - ranks) / float(n_proposals - 1)) ** p
    else:
        scores = np.ones_like(ranks, dtype=float)
    return int(np.argmax(scores.sum(axis=0)))


def run_power_scoring_point(args, nk, initial_solutions, vote_idx, p, iterations):
    solutions = initial_solutions.copy()
    num_solutions = solutions.shape[0]
    mean_history = np.zeros(iterations)
    variance_history = np.zeros(iterations)

    for it in range(iterations):
        np.random.shuffle(vote_idx)
        prop_idx = vote_idx[:args.vote_size]
        n_prop = 2 ** args.vote_size

        fitnesses = np.zeros((num_solutions, n_prop))
        for pr in range(n_prop):
            proposed = solutions.copy()
            proposed[:, prop_idx] = decimal_to_binary(pr, args.vote_size)
            fitnesses[:, pr] = nk.calculate_fitness_batch(proposed)

        winner_idx = power_scoring_vote(fitnesses, p)
        solutions[:, prop_idx] = decimal_to_binary(winner_idx, args.vote_size)
        solutions = np.unique(solutions, axis=0)
        num_solutions = solutions.shape[0]

        f = nk.calculate_fitness_batch(solutions)
        mean_history[it] = np.mean(f)
        variance_history[it] = np.var(f)

    return mean_history, variance_history


def run_point(args, K, alpha):
    iterations = args.iterations_base + args.iterations_slope * int(K)
    mean_history = np.zeros((args.runs, iterations))
    variance_history = np.zeros((args.runs, iterations))

    for run in range(args.runs):
        solutions, vote_idx, non_vote_idx = build_initial_solutions(
            args.N,
            args.num_solutions,
            args.voting_portion,
            args.seed + run * 10000 + int(K) * 1000000 + int(round(alpha * 100)) * 1000,
        )

        np.random.seed(args.seed + run * 10000 + int(K) * 1000000 + int(round(alpha * 100)) * 1000 + 100)
        nk = NKLandscape(args.N, int(K))
        dep = NKLandscape.build_split_dependency_matrix(
            args.N, int(K), vote_idx, non_vote_idx, float(alpha)
        )
        nk.set_dependency_matrix(dep)

        np.random.seed(args.seed + run * 10000 + int(K) * 1000000 + int(round(alpha * 100)) * 1000 + 200)
        if args.scoring_power_p is not None:
            mh, vh = run_power_scoring_point(
                args, nk, solutions, vote_idx.copy(), args.scoring_power_p, iterations
            )
            mean_history[run] = mh
            variance_history[run] = vh
        else:
            vote = VoteModel(
                nk,
                solutions=solutions,
                possible_vote_indices=vote_idx,
                vote_size=args.vote_size,
                vote_type=args.vote_type,
            )

            for it in range(iterations):
                if it > 0:
                    vote.step()
                fitnesses = vote.get_fitnesses()
                mean_history[run, it] = np.mean(fitnesses)
                variance_history[run, it] = np.var(fitnesses)

    terminal_mean_per_run = mean_history[:, -TERMINAL_WINDOW:].mean(axis=1)
    terminal_var_per_run = variance_history[:, -TERMINAL_WINDOW:].mean(axis=1)

    return {
        "iterations": iterations,
        "terminal_mean": float(np.mean(terminal_mean_per_run)),
        "terminal_mean_se": float(np.std(terminal_mean_per_run) / np.sqrt(args.runs)),
        "terminal_variance": float(np.mean(terminal_var_per_run)),
        "terminal_variance_se": float(np.std(terminal_var_per_run) / np.sqrt(args.runs)),
    }


def main():
    args = parse_args()
    if args.scoring_power_p is None and args.vote_type is None:
        raise ValueError("Provide either --vote_type or --scoring_power_p")
    if args.scoring_power_p is not None and args.vote_type is not None:
        raise ValueError("Use only one of --vote_type or --scoring_power_p")

    outdir = os.path.dirname(args.output_npz)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    sampled = sample_all_K0(num_points=args.num_points, integer_K=True)
    K0_values = np.arange(1, 21, dtype=int)
    max_points = max(len(pts) for pts in sampled)

    sampled_K = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_alpha = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_arc_fraction = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_terminal_mean = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_terminal_mean_se = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_terminal_variance = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_terminal_variance_se = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_iterations = np.full((len(K0_values), max_points), np.nan, dtype=float)
    sampled_counts = np.zeros(len(K0_values), dtype=int)
    vote_type_name = (
        "scoring_power_p{:.2f}".format(args.scoring_power_p)
        if args.scoring_power_p is not None else args.vote_type
    )

    for i, K0 in enumerate(K0_values):
        pts = sampled[i]
        sampled_counts[i] = len(pts)
        print("K0={} -> {} points".format(K0, len(pts)), flush=True)
        for j, pt in enumerate(pts):
            sampled_K[i, j] = pt.K
            sampled_alpha[i, j] = pt.alpha
            sampled_arc_fraction[i, j] = pt.arc_fraction
            stats = run_point(args, int(round(pt.K)), float(pt.alpha))
            sampled_terminal_mean[i, j] = stats["terminal_mean"]
            sampled_terminal_mean_se[i, j] = stats["terminal_mean_se"]
            sampled_terminal_variance[i, j] = stats["terminal_variance"]
            sampled_terminal_variance_se[i, j] = stats["terminal_variance_se"]
            sampled_iterations[i, j] = stats["iterations"]
            print(
                "  K={:2d} alpha={:.4f} mean={:.4f} se={:.4f}".format(
                    int(round(pt.K)), pt.alpha, stats["terminal_mean"], stats["terminal_mean_se"]
                ),
                flush=True,
            )

    np.savez_compressed(
        args.output_npz,
        vote_type=np.array(vote_type_name),
        K0=K0_values,
        num_points_requested=np.array(args.num_points),
        runs=np.array(args.runs),
        curve_a=np.array(PAPER_A),
        curve_b=np.array(PAPER_B),
        scoring_power_p=(np.array(args.scoring_power_p)
                         if args.scoring_power_p is not None else np.array(np.nan)),
        terminal_window=np.array(TERMINAL_WINDOW),
        iterations_base=np.array(args.iterations_base),
        iterations_slope=np.array(args.iterations_slope),
        sampled_counts=sampled_counts,
        sampled_K=sampled_K,
        sampled_alpha=sampled_alpha,
        sampled_arc_fraction=sampled_arc_fraction,
        sampled_terminal_mean=sampled_terminal_mean,
        sampled_terminal_mean_se=sampled_terminal_mean_se,
        sampled_terminal_variance=sampled_terminal_variance,
        sampled_terminal_variance_se=sampled_terminal_variance_se,
        sampled_iterations=sampled_iterations,
    )
    print("saved {}".format(args.output_npz), flush=True)


if __name__ == "__main__":
    main()
