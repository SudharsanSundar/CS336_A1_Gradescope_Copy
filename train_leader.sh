#!/bin/bash
#SBATCH --job-name=train_leader
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=train_leader_%j.out
#SBATCH --error=train_leader_%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

CUDA_LAUNCH_BLOCKING=1
python3 cs336_basics/train_transformer.py leaderboard.json