#!/usr/bin/env python3
"""Generate jobs for missing K values in the rep grid."""

import os

BETAS     = [0.0, 0.5, 1.0]
PSELFS    = [0.0, 0.5, 1.0]
EXISTING_KS = {1, 5, 10, 20}
ALL_KS    = set(range(1, 21))
NEW_KS    = sorted(ALL_KS - EXISTING_KS)  # [2,3,4,6,7,8,9,11,12,13,14,15,16,17,18,19]
ALPHAS    = [round(a * 0.1, 1) for a in range(11)]
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
for K in NEW_KS:
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
           "/jobs_rep_fill.txt")
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written {len(lines)} jobs to {outfile}")
print(f"  {len(NEW_KS)} new K values: {NEW_KS}")
print(f"  {len(BETAS)} β × {len(PSELFS)} p_self × {len(ALPHAS)} α × {len(VOTE_TYPES)} methods")

# Verify none already exist
existing = 0
for line in lines:
    d = line.split("--output_dir ")[1].split(" --")[0]
    if os.path.exists(os.path.join(d, "metadata.json")):
        existing += 1
if existing:
    print(f"  WARNING: {existing} already exist")
else:
    print(f"  Confirmed: 0 already exist")
