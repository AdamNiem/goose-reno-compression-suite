#!/bin/bash

#SBATCH --job-name restore_intensities_goose_decomp_lcp_ply
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno

pixi run python restore_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/lcp_ply_decompressed_lidar/EB_0.01   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root goose-dataset/lcp_bin_decompressed_lidar/EB_0.01   --num_workers 4

pixi run python restore_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/lcp_ply_decompressed_lidar/EB_0.085901831   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root goose-dataset/lcp_bin_decompressed_lidar/EB_0.085901831   --num_workers 4

pixi run python restore_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/lcp_ply_decompressed_lidar/EB_0.1   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root goose-dataset/lcp_bin_decompressed_lidar/EB_0.1   --num_workers 4

pixi run python restore_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/lcp_ply_decompressed_lidar/EB_0.2364   --orig_bin_root goose-dataset/lidar  --no_threshold   --out_bin_root goose-dataset/lcp_bin_decompressed_lidar/EB_0.2364   --num_workers 4

pixi run python restore_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/lcp_ply_decompressed_lidar/EB_0.689   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root goose-dataset/lcp_bin_decompressed_lidar/EB_0.689   --num_workers 4



#Made the threshold a high value since I dont really mind what it is since its only a sanity check thing and has not real effect on the restoration