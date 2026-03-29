#!/usr/bin/env python3
"""
Generate jobs_rep_grid.txt for the representative democracy sweep.

Sweep:
  beta    in {0.0, 0.5, 1.0}
  p_self  in {0.0, 0.5, 1.0}
  K       in {1, 5, 10, 20}           -- outer loop for stride-based K-balancing
  alpha   in {0.0, 0.1, ..., 1.0}    -- 11 levels
  vote_type: all 8 methods

Fixed:
  N=50, voting_portion=0.5, num_solutions=100
  num_candidates=5, selection_temperature=None (uniform)
  vote_size=4 (16 proposals), runs=500, seed=0
  iterations = 150 + 50*K
"""

import itertools

BETAS     = [0.0, 0.5, 1.0]
PSELFS    = [0.0, 0.5, 1.0]
KS        = [1, 5, 10, 20]          # outer loop → K-balanced under striding
ALPHAS    = [round(a * 0.1, 1) for a in range(11)]  # 0.0 … 1.0
VOTE_TYPES = [
    "plurality", "approval", "total_score",
    "borda", "irv", "star", "minimax", "random_dictator",
]

BASE_OUT = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
            "/src/UROC/results/rep")
N             = 50
VOTING_PORTION = 0.5
NUM_SOLUTIONS  = 100
NUM_CANDIDATES = 5
VOTE_SIZE      = 4
RUNS           = 500
SEED           = 0

lines = []
# K is the outer loop so consecutive jobs cycle across K values → stride-balanced
for K in KS:
    iters = 150 + 50 * K
    for beta in BETAS:
        for p_self in PSELFS:
            for alpha in ALPHAS:
                for vt in VOTE_TYPES:
                    out = (f"{BASE_OUT}/K{K}_a{alpha:.2f}"
                           f"/b{beta:.1f}_p{p_self:.1f}/{vt}")
                    cmd = (
                        f"python run_experiment.py"
                        f" --output_dir {out}"
                        f" --N {N}"
                        f" --K {K}"
                        f" --alpha {alpha:.2f}"
                        f" --voting_portion {VOTING_PORTION}"
                        f" --vote_type {vt}"
                        f" --vote_size {VOTE_SIZE}"
                        f" --num_solutions {NUM_SOLUTIONS}"
                        f" --runs {RUNS}"
                        f" --iterations {iters}"
                        f" --seed {SEED}"
                        f" --num_candidates {NUM_CANDIDATES}"
                        f" --beta {beta:.1f}"
                        f" --p_self {p_self:.1f}"
                    )
                    lines.append(cmd)

outfile = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
           "/jobs_rep_grid.txt")
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written {len(lines)} jobs to {outfile}")

# Sanity check
n_expected = len(KS) * len(BETAS) * len(PSELFS) * len(ALPHAS) * len(VOTE_TYPES)
assert len(lines) == n_expected, f"Expected {n_expected}, got {len(lines)}"
print(f"Check: {len(KS)} K × {len(BETAS)} beta × {len(PSELFS)} p_self"
      f" × {len(ALPHAS)} alpha × {len(VOTE_TYPES)} vote_types = {n_expected} ✓")
