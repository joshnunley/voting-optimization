#!/bin/bash
#SBATCH --job-name=rep_fill
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/rf_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/rf_%A_%a.err
#SBATCH --array=0-499

# 12672 jobs across 500 tasks via stride.
# Each task runs ~25 jobs. K is outer loop so stride-balanced.

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_rep_fill.txt
TOTAL_JOBS=12672
STRIDE=500

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
