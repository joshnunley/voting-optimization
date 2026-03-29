#!/usr/bin/env python3
"""Generate jobs for the Borda family sweep."""

LAMS = [round(l * 0.1, 1) for l in range(11)]    # 0.0 .. 1.0
RHOS = [round(r * 0.1, 1) for r in range(11)]    # 0.0 .. 1.0
KS = [1, 5, 10, 15, 20]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

BASE = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
        "/src/UROC/results/borda_family")
RUNS = 100

lines = []
for K in KS:
    iters = 150 + 50 * K
    for alpha in ALPHAS:
        for lam in LAMS:
            for rho in RHOS:
                out = f"{BASE}/K{K}_a{alpha:.2f}/l{lam:.1f}_r{rho:.1f}"
                cmd = (f"python run_borda_family.py"
                       f" --lam {lam:.1f} --rho {rho:.1f}"
                       f" --K {K} --alpha {alpha:.2f}"
                       f" --runs {RUNS} --iterations {iters}"
                       f" --seed 0 --output_dir {out}")
                lines.append(cmd)

outfile = ("/geode2/home/u040/joshnunl/BigRed200/voting-optimization"
           "/jobs_borda_family.txt")
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")

n = len(lines)
print(f"Written {n} jobs to {outfile}")
print(f"  {len(KS)} K × {len(ALPHAS)} α × {len(LAMS)} λ × {len(RHOS)} ρ = {n}")
