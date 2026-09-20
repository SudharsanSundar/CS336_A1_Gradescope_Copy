#!/bin/bash
#SBATCH --job-name=test_tok
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=test_tok_%j.out
#SBATCH --error=test_tok_%j.err

# Optional: activate a conda environment to use for this job
# eval "$(conda shell.bash hook)"
# conda activate <environment name>

pytest tests/test_tokenizer.py
