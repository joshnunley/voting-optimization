#!/bin/bash
#SBATCH --job-name=vote_rep
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/rep_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/rep_%A_%a.err
#SBATCH --array=0-299

# 3168 jobs across 300 tasks via stride-based assignment.
# K is the outer loop in the jobs file (all K=1 first, then K=5, K=10, K=20),
# so stride-300 naturally distributes each K tier across all tasks.
# Each task runs at most 11 jobs; worst-case wall time ~4h for K=20-heavy tasks.

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_rep_grid.txt
TOTAL_JOBS=3168
STRIDE=300

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

JOB_NUM=0
while true; do
    LINE=$(( SLURM_ARRAY_TASK_ID + JOB_NUM * STRIDE + 1 ))
    if [ "$LINE" -gt "$TOTAL_JOBS" ]; then
        break
    fi
    CMD=$(sed -n "${LINE}p" "$JOBSFILE")
    if [ -n "$CMD" ]; then
        eval "$CMD"
    fi
    JOB_NUM=$(( JOB_NUM + 1 ))
done
