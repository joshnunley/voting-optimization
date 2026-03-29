#!/usr/bin/env python3
PS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
      2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
KS = list(range(1, 21))
ALPHAS = [round(a * 0.05, 2) for a in range(21)]
RUNS = 1000
BASE = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
        "/results/scoring_fine")

lines = []
for K in KS:
    iters = 150 + 50 * K
    for alpha in ALPHAS:
        for p in PS:
            out = f"{BASE}/K{K}_a{alpha:.2f}/p{p:.2f}"
            cmd = (f"python run_scoring_power.py --p {p:.2f}"
                   f" --K {K} --alpha {alpha:.2f}"
                   f" --N 50 --voting_portion 0.5 --vote_size 2"
                   f" --num_solutions 100"
                   f" --runs {RUNS} --iterations {iters}"
                   f" --seed 0 --output_dir {out}")
            lines.append(cmd)

outfile = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
           "/jobs_scoring_ext.txt")
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Written {len(lines)} jobs")
