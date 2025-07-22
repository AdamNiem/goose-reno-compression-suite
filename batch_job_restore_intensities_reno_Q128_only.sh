#!/bin/bash

#SBATCH --job-name restore_intensities_goose_decomp_reno_q128
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno

pixi run python restore_intensity_feature_dataset_parallel2.py   --ply_root /scratch/aniemcz/goose-pointcept/reno_decompressed_lidar_Q128_only/Q_128   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_lidar_Q128_only/Q_128   --num_workers 4
