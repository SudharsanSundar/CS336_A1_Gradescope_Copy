#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=analysis_%j.out
#SBATCH --error=analysis_%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python3 cs336_basics/analysis.py