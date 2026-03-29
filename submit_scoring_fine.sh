#!/bin/bash
#SBATCH --job-name=score_fin
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sf_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sf_%A_%a.err
#SBATCH --array=0-419

# 4200 jobs across 420 tasks via stride (10 jobs each).
# K is outer loop so stride-balanced across K tiers.

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_scoring_fine.txt
TOTAL_JOBS=4200
STRIDE=420

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
