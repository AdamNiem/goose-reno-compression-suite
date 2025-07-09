#!/bin/bash

#SBATCH --job-name train_RENO
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 64gb
#SBATCH --time 14:00:00

cd /home/aniemcz/gooseReno

pixi run wandb login

# Activate pixi environment and run training
pixi run python RENO/train.py \
  --training_data='/scratch/aniemcz/goose-reno/train/**/*.bin' \
  --model_save_folder='/scratch/aniemcz/renoGooseModels/goose'