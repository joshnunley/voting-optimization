#!/bin/bash
#SBATCH --job-name=vote_probe
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/probe_results/probe_%j.out

module load python/3.12.11

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

OUTBASE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/probe_results

echo "Starting probes at $(date)"

# K=1, alpha=0 (easy landscape, no coupling)
{ time python3 run_experiment.py \
    --output_dir $OUTBASE/K1_a0/plurality \
    --N 50 --K 1 --alpha 0.0 --voting_portion 0.5 \
    --vote_type plurality --vote_size 2 \
    --num_solutions 300 --runs 50 --iterations 100 --seed 0
} 2>&1 | sed 's/^/[K1_a0_plurality] /' &

# K=10, alpha=0.5 (moderate)
{ time python3 run_experiment.py \
    --output_dir $OUTBASE/K10_a0.5/irv \
    --N 50 --K 10 --alpha 0.5 --voting_portion 0.5 \
    --vote_type irv --vote_size 2 \
    --num_solutions 300 --runs 50 --iterations 100 --seed 0
} 2>&1 | sed 's/^/[K10_a0.5_irv] /' &

# K=20, alpha=1.0 (hardest landscape, full coupling)
{ time python3 run_experiment.py \
    --output_dir $OUTBASE/K20_a1/minimax \
    --N 50 --K 20 --alpha 1.0 --voting_portion 0.5 \
    --vote_type minimax --vote_size 2 \
    --num_solutions 300 --runs 50 --iterations 100 --seed 0
} 2>&1 | sed 's/^/[K20_a1_minimax] /' &

# Representative democracy (K=15, alpha=0.5, 5 candidates)
{ time python3 run_experiment.py \
    --output_dir $OUTBASE/K15_rep/borda \
    --N 50 --K 15 --alpha 0.5 --voting_portion 0.5 \
    --vote_type borda --vote_size 2 \
    --num_solutions 300 --runs 50 --iterations 100 --seed 0 \
    --num_candidates 5
} 2>&1 | sed 's/^/[K15_rep_borda] /' &

wait
echo "All probes done at $(date)"
