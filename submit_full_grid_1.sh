#!/bin/bash
#SBATCH --job-name=vote_full_1
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/full1_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/full1_%A_%a.err
#SBATCH --array=0-499

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_full_grid.txt
TOTAL_JOBS=4200
TOTAL_TASKS=1000
TASK_ID=${SLURM_ARRAY_TASK_ID}   # 0-499

START=$(( TASK_ID * TOTAL_JOBS / TOTAL_TASKS + 1 ))
NEXT=$(( (TASK_ID + 1) * TOTAL_JOBS / TOTAL_TASKS + 1 ))

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

for LINE in $(seq $START $((NEXT - 1))); do
    CMD=$(sed -n "${LINE}p" "$JOBSFILE")
    if [ -n "$CMD" ]; then
        eval "$CMD"
    fi
done
