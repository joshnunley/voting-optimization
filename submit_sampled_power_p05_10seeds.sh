#!/bin/bash
#SBATCH --job-name=sp05_samp
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sp05_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sp05_%A_%a.err
#SBATCH --array=0-9

module load python/3.12.11

JOBSFILE=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/jobs_sampled_power_p05_10seeds.txt
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

CMD=$(sed -n "${LINE}p" "$JOBSFILE")
if [ -n "$CMD" ]; then
    eval "$CMD"
fi
