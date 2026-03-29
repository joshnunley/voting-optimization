#!/usr/bin/env python3
"""Generate fine-grained scoring power sweep over full direct democracy grid."""

PS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
KS = list(range(1, 21))
ALPHAS = [round(a * 0.05, 2) for a in range(21)]
RUNS = 1000
BASE = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
        "/src/UROC/results/scoring_fine")

lines = []
# K as outer loop for stride-based K-balancing
for K in KS:
    iters = 150 + 50 * K
    for alpha in ALPHAS:
        for p in PS:
            out = f"{BASE}/K{K}_a{alpha:.2f}/p{p:.2f}"
            cmd = (f"python run_scoring_power.py --p {p}"
                   f" --K {K} --alpha {alpha:.2f}"
                   f" --runs {RUNS} --iterations {iters}"
                   f" --seed 0 --output_dir {out}")
            lines.append(cmd)

outfile = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
           "/jobs_scoring_fine.txt")
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written {len(lines)} jobs")
print(f"  {len(KS)} K × {len(ALPHAS)} α × {len(PS)} p = {len(lines)}")
