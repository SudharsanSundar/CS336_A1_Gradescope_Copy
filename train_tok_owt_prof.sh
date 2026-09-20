#!/bin/bash
#SBATCH --job-name=train_tok_owt_prof
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=train_tok_owt_prof_%j.out
#SBATCH --error=train_tok_owt_prof_%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

kernprof -l cs336_basics/train_tok_profiled.py