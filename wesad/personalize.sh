#!/usr/bin/env bash
#
#SBATCH --job-name=personalize
#SBATCH --partition=gpu-long
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/personalize.txt
#SBATCH --error=data/logs/personalize.err

module load miniconda/4.11.0
module load cuda/11.3.1
conda activate pytorch_env
python3 -m main personalize