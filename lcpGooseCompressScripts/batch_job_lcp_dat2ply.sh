#!/bin/bash

#SBATCH --job-name convertdat2ply
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno/lcpGooseCompressScripts
  
pixi run python dat2ply_parallel.py \
  --input_root /scratch/aniemcz/goose-pointcept/lcp_decompressed_lidar \
  --output_root /scratch/aniemcz/goose-pointcept/lcp_ply_decompressed_lidar \
  --num_workers 4