#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=train_model_%j.out
#SBATCH --error=train_model_%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python3 cs336_basics/train_transformer.py sweep2.json