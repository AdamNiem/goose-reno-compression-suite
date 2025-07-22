#!/bin/bash

#SBATCH --job-name create_labels_4_decompressed_lidar_reno_q128
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno

Q_lvl="Q_128"
pixi run python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_lidar_Q128_only/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold
