#!/bin/bash

#SBATCH --job-name compressGooseWithLCP
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 24:00:00

cd /home/aniemcz/gooseReno/lcpGooseCompressScripts

pixi run python lcp_compress_goose_dataset_parallel.py \
  --data_root /scratch/aniemcz/goose-pointcept/dat_xyz_only_lidar \
  --ply_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar \
  --output_root /scratch/aniemcz/goose-pointcept/lcp_compression_results \
  --workers 4