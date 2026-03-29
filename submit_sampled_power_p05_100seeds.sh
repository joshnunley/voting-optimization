#!/bin/bash
#SBATCH --job-name=sp05_s100
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sp05s100_%A_%a.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sp05s100_%A_%a.err
#SBATCH --array=0-99

module load python/3.12.11

SEED=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

python generate_sampled_curve_data.py \
  --scoring_power_p 0.5 \
  --num_points 4 \
  --runs 1 \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --output_npz "results/sampled_curves/scoring_power_p0.50_100seeds/seed_${SEED}.npz"
