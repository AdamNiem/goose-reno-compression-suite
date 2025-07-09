#!/bin/bash

#SBATCH --job-name compressGooseWithReno
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 256gb
#SBATCH --time 2:00:00

cd /home/aniemcz/gooseReno

# Activate pixi environment and run training
pixi run python reno_compress_goose_dataset.py \
  --data_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar \
  --ckpt ./RENO/model/Goose/ckpt.pt \
  --quant_levels 8 \
  --output_root /scratch/aniemcz/goose-pointcept/reno_decompressed_lidar
  
#v100 seems to no longer be allowed in batch job?