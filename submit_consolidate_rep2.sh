#!/bin/bash
#SBATCH --job-name=cons_rep
#SBATCH --partition=general
#SBATCH -A r01998
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/cons_rep2_%j.out
#SBATCH --error=/geode2/home/u040/joshnunl/BigRed200/voting-optimization/logs/cons_rep2_%j.err

module load python/3.12.11
cd /geode2/home/u040/joshnunl/BigRed200/voting-optimization/src/UROC
python3 consolidate_results.py rep
