#!/usr/bin/env python3
PS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
KS = [1, 5, 10, 15, 20]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RUNS = 100
BASE = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
        "/src/UROC/results/scoring_power")

lines = []
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
           "/jobs_scoring_power.txt")
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Written {len(lines)} jobs")
