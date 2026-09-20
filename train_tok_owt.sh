#!/bin/bash
#SBATCH --job-name=train_tok_owt
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=train_tok_owt_%j.out
#SBATCH --error=train_tok_owt_%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python3 cs336_basics/train_tok_profiled.py ../../../data/owt_train_2G.txt 32000 '<|endoftext|>'