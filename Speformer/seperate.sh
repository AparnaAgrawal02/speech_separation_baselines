#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH -w gnode081
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python sepformer_whamr.py
