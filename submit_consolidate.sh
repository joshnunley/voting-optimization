#!/bin/bash
#SBATCH --job-name=consolidate
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/consolidate_%j.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/consolidate_%j.err

module load python/3.12.11
cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC
python3 consolidate_results.py
