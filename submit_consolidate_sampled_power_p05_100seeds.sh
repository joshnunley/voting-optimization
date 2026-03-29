#!/bin/bash
#SBATCH --job-name=sp05c100
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sp05c100_%j.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/sp05c100_%j.err

module load python/3.12.11
cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC

python consolidate_sampled_curve_data.py \
  --input_glob "results/sampled_curves/scoring_power_p0.50_100seeds/seed_*.npz" \
  --output_npz "results/sampled_curves/scoring_power_p0.50_100seeds.npz"
