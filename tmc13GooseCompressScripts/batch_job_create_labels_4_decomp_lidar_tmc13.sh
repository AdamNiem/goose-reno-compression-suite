#!/bin/bash

#SBATCH --job-name create_labels_4_decompressed_lidar_tmc13
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno

Q_lvl="Q_0.0668"
pixi run python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold
  
Q_lvl="Q_0.00521332"
pixi run python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold
  
Q_lvl="Q_0.0001"
pixi run python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold