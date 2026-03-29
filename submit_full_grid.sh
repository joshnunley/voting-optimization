#!/bin/bash
#SBATCH --job-name=vote_full
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/full_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/full_%A_%a.err
#SBATCH --array=0-335

# Stride-based assignment: task i runs jobs i, i+336, i+336*2, ..., i+336*9
# (one experiment from each K-tier per task → load balanced to within ±3.7%)
# Worst-case task: ~7000 total iterations across 300 runs ≈ 2.2h

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_full_grid.txt
STRIDE=336

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

for i in 0 1 2 3 4 5 6 7 8 9; do
    LINE=$(( SLURM_ARRAY_TASK_ID + i * STRIDE + 1 ))
    CMD=$(sed -n "${LINE}p" "$JOBSFILE")
    if [ -n "$CMD" ]; then
        eval "$CMD"
    fi
done
