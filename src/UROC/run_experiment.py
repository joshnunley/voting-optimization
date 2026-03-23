#!/usr/bin/env python3
"""
Single-job experiment runner for voting optimization on NK landscapes.

Each invocation runs one (K, alpha, vote_type) combination across all runs,
saving full fitness distributions and summary statistics.

Usage:
    python run_experiment.py --output_dir results/sweep \
        --N 50 --K 10 --alpha 0.5 --vote_type plurality \
        --runs 50 --iterations 100 --num_solutions 100

For representative democracy:
    python run_experiment.py --output_dir results/rep \
        --N 50 --K 10 --alpha 0.5 --vote_type plurality \
        --num_candidates 5 --selection_temperature 1.0 \
        --runs 50 --iterations 100 --num_solutions 100
"""

import argparse
import os
import sys
import json
import numpy as np
from NKLandscape import NKLandscape
from VoteModel import VoteModel


def parse_args():
    p = argparse.ArgumentParser(description="Run voting experiment on NK landscape")
    p.add_argument("--output_dir", required=True, help="Output directory for results")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Cross-dependency fraction (0=independent, 1=fully coupled)")
    p.add_argument("--voting_portion", type=float, default=0.5)
    p.add_argument("--vote_type", type=str, required=True,
                   choices=VoteModel.VOTE_TYPES)
    p.add_argument("--vote_size", type=int, default=2)
    p.add_argument("--num_solutions", type=int, default=100)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for reproducibility")
    # Representative democracy
    p.add_argument("--num_candidates", type=int, default=None,
                   help="Number of candidates (None = direct democracy)")
    p.add_argument("--selection_temperature", type=float, default=None,
                   help="Softmax temperature for candidate selection")
    return p.parse_args()


def build_initial_solutions(N, num_solutions, voting_portion, seed):
    """Build initial solutions: shared voting bits + individual non-voting bits."""
    rng = np.random.RandomState(seed)

    n_voting = int(voting_portion * N)
    indices = np.arange(N)
    rng.shuffle(indices)
    vote_indices = indices[:n_voting]
    non_vote_indices = indices[n_voting:]

    solutions = np.zeros((num_solutions, N), dtype=int)
    # Shared voting bits: same for all individuals
    voting_bits = rng.randint(2, size=n_voting)
    solutions[:, vote_indices] = voting_bits
    # Individual non-voting bits: unique per individual
    non_voting_bits = rng.randint(2, size=(num_solutions, N - n_voting))
    solutions[:, non_vote_indices] = non_voting_bits

    return solutions, vote_indices, non_vote_indices


def run_single(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Build initial state (deterministic from base seed)
    initial_solutions, vote_indices, non_vote_indices = build_initial_solutions(
        args.N, args.num_solutions, args.voting_portion, args.seed
    )

    # Build NK landscape with cross-dependency
    np.random.seed(args.seed + 1000)
    nk = NKLandscape(args.N, args.K)

    if args.alpha > 0 and args.K > 0:
        dep_matrix = NKLandscape.build_split_dependency_matrix(
            args.N, args.K, vote_indices, non_vote_indices, args.alpha
        )
        nk.set_dependency_matrix(dep_matrix)

    # Storage: summary stats + full fitness distributions
    mean_history = np.zeros((args.runs, args.iterations))
    variance_history = np.zeros((args.runs, args.iterations))
    min_history = np.zeros((args.runs, args.iterations))
    max_history = np.zeros((args.runs, args.iterations))
    # Full distributions: (runs, iterations, num_solutions) - may shrink as solutions merge
    # Store as list of arrays since num_solutions can change
    fitness_distributions = []

    for run in range(args.runs):
        # Each run gets a unique seed for the voting stochasticity,
        # but same landscape and initial solutions
        np.random.seed(args.seed + run * 10000 + 2000)

        vote = VoteModel(
            nk,
            solutions=initial_solutions,
            possible_vote_indices=vote_indices,
            vote_size=args.vote_size,
            vote_type=args.vote_type,
            num_candidates=args.num_candidates,
            selection_temperature=args.selection_temperature,
        )

        run_distributions = []

        for it in range(args.iterations):
            if it > 0:
                vote.step()

            fitnesses = vote.get_fitnesses()
            mean_history[run, it] = np.mean(fitnesses)
            variance_history[run, it] = np.var(fitnesses)
            min_history[run, it] = np.min(fitnesses)
            max_history[run, it] = np.max(fitnesses)
            run_distributions.append(fitnesses.copy())

        fitness_distributions.append(run_distributions)
        print(f"{args.vote_type} K={args.K} a={args.alpha} run={run}/{args.runs}",
              flush=True)

    # Save results
    np.save(os.path.join(args.output_dir, "mean_history.npy"), mean_history)
    np.save(os.path.join(args.output_dir, "variance_history.npy"), variance_history)
    np.save(os.path.join(args.output_dir, "min_history.npy"), min_history)
    np.save(os.path.join(args.output_dir, "max_history.npy"), max_history)

    # Save distributions as object array (variable-length per iteration)
    np.save(os.path.join(args.output_dir, "fitness_distributions.npy"),
            np.array(fitness_distributions, dtype=object),
            allow_pickle=True)

    # Save initial state and metadata
    np.save(os.path.join(args.output_dir, "initial_solutions.npy"), initial_solutions)
    np.save(os.path.join(args.output_dir, "fitness_mapping.npy"), nk.get_fitness_mapping())
    np.save(os.path.join(args.output_dir, "dependency_matrix.npy"), nk.get_dependency_matrix())

    metadata = {
        "N": args.N, "K": args.K, "alpha": args.alpha,
        "voting_portion": args.voting_portion,
        "vote_type": args.vote_type, "vote_size": args.vote_size,
        "num_solutions": args.num_solutions,
        "runs": args.runs, "iterations": args.iterations,
        "seed": args.seed,
        "num_candidates": args.num_candidates,
        "selection_temperature": args.selection_temperature,
        "democracy": "representative" if args.num_candidates else "direct",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    run_single(args)
