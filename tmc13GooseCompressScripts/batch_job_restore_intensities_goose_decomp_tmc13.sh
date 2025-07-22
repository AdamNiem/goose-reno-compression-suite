#!/bin/bash

#SBATCH --job-name restore_intensities_goose_decomp_tmc13_dequant
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno

pixi run python restore_quantized_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/tmc13_decompressed_lidar/Q_0.0001   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root goose-dataset/tmc13_bin_dequantized_decompressed_lidar/Q_0.0001   --num_workers 4

pixi run python restore_quantized_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/tmc13_decompressed_lidar/Q_0.00521332   --orig_bin_root goose-dataset/lidar   --no_threshold   --out_bin_root goose-dataset/tmc13_bin_dequantized_decompressed_lidar/Q_0.00521332   --num_workers 4

pixi run python restore_quantized_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/tmc13_decompressed_lidar/Q_0.0668   --orig_bin_root goose-dataset/lidar  --no_threshold   --out_bin_root goose-dataset/tmc13_bin_dequantized_decompressed_lidar/Q_0.0668   --num_workers 4

#Made the threshold a high value since I dont really mind what it is since its only a sanity check thing and has not real effect on the restoration