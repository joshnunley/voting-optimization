#!/bin/bash
#SBATCH --job-name=rep_s035
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/rep_s035_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/rep_s035_%A_%a.err
#SBATCH --array=0-197

# 1980 scoring_p035 rep jobs, 198 tasks, 10 jobs each

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_rep_scoring035.txt
TOTAL_JOBS=1980
STRIDE=198

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
