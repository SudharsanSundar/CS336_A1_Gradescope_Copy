#!/bin/bash
#SBATCH --job-name=train_tok_ts
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=train_tok_ts_%j.out
#SBATCH --error=train_tok_ts_%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python3 cs336_basics/tokenizer_fast.py ../../../data/TinyStoriesV2-GPT4-train.txt 10000 '<|endoftext|>'