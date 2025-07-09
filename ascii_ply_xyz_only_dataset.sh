#!/bin/bash

#SBATCH --job-name processGooseDataset
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node v100:0
#SBATCH --mem 64gb
#SBATCH --time 1:00:00

cd /home/aniemcz/gooseReno

# Activate pixi environment and run training
pixi run python create_ascii_ply_xyz_only_dataset_parallel.py \
  --input_root /scratch/aniemcz/goose-pointcept/lidar \
  --output_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar