#!/bin/bash
#SBATCH --job-name=vote_fix_a0
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/fix_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/fix_%A_%a.err
#SBATCH --array=0-183

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_alpha0_fix.txt
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
CMD=$(sed -n "${LINE}p" "$JOBSFILE")

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC
eval "$CMD"
