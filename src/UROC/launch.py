#!/usr/bin/env python3
"""
Generate shell commands for dispatching all experiment jobs.

Usage:
    python launch.py > jobs.sh
    # Then dispatch jobs.sh lines across your cluster (e.g., GNU parallel, SLURM array, etc.)

    # Or run directly with GNU parallel:
    python launch.py | parallel -j 100

    # Or with a SLURM array:
    python launch.py > jobs.txt
    # Submit with: sbatch --array=1-$(wc -l < jobs.txt) run_array.sh
"""

import itertools

# ---- Experiment 1: K x alpha sweep (direct democracy) ----

K_VALUES = [0, 1, 5, 10, 15, 20]
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
VOTE_TYPES = [
    "plurality", "approval", "total_score",
    "borda", "irv", "star", "minimax", "random_dictator",
]

N = 50
VOTING_PORTION = 0.5
NUM_SOLUTIONS = 300
RUNS = 50
ITERATIONS = 300
VOTE_SIZE = 2
SEED = 0
PYTHON = "python"  # Change to your python path if needed

BASE_DIR = "results"

jobs = []

# Experiment 1: Direct democracy K x alpha sweep
for K, alpha, vote_type in itertools.product(K_VALUES, ALPHA_VALUES, VOTE_TYPES):
    # Skip alpha > 0 when K = 0 (no dependencies to cross)
    if K == 0 and alpha > 0:
        continue
    out = f"{BASE_DIR}/direct/K{K}_a{alpha:.2f}/{vote_type}"
    cmd = (
        f"{PYTHON} run_experiment.py"
        f" --output_dir {out}"
        f" --N {N} --K {K} --alpha {alpha}"
        f" --voting_portion {VOTING_PORTION}"
        f" --vote_type {vote_type}"
        f" --vote_size {VOTE_SIZE}"
        f" --num_solutions {NUM_SOLUTIONS}"
        f" --runs {RUNS} --iterations {ITERATIONS}"
        f" --seed {SEED}"
    )
    jobs.append(cmd)

# Experiment 2: Representative democracy
REP_K_VALUES = [5, 10, 15]
REP_ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
NUM_CANDIDATES_VALUES = [3, 5]
TEMPERATURE_VALUES = [None, 1.0, 0.1]

for K, alpha, vote_type, n_cand, temp in itertools.product(
    REP_K_VALUES, REP_ALPHA_VALUES, VOTE_TYPES,
    NUM_CANDIDATES_VALUES, TEMPERATURE_VALUES,
):
    temp_str = f"_t{temp}" if temp is not None else "_tuniform"
    out = f"{BASE_DIR}/representative/K{K}_a{alpha:.2f}/c{n_cand}{temp_str}/{vote_type}"
    cmd = (
        f"{PYTHON} run_experiment.py"
        f" --output_dir {out}"
        f" --N {N} --K {K} --alpha {alpha}"
        f" --voting_portion {VOTING_PORTION}"
        f" --vote_type {vote_type}"
        f" --vote_size {VOTE_SIZE}"
        f" --num_solutions {NUM_SOLUTIONS}"
        f" --runs {RUNS} --iterations {ITERATIONS}"
        f" --seed {SEED}"
        f" --num_candidates {n_cand}"
    )
    if temp is not None:
        cmd += f" --selection_temperature {temp}"
    jobs.append(cmd)

print(f"# Total jobs: {len(jobs)}", flush=True)
for job in jobs:
    print(job)
