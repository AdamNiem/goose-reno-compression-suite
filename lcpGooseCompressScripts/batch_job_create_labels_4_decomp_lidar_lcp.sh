#!/bin/bash

#SBATCH --job-name create_labels_4_decompressed_lidar_lcp
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

cd /home/aniemcz/gooseReno

Q_lvls=(
  "EB_0.01"
  "EB_0.085901831"
  "EB_0.1"
  "EB_0.2364"
  "EB_0.689"
)

for Q_lvl in "${Q_lvls[@]}"; do
    pixi run python create_labels_4_decompressed_lidar.py \
      --decomp_bin_root /scratch/aniemcz/goose-pointcept/lcp_bin_decompressed_lidar/${Q_lvl} \
      --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
      --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
      --out_label_root /scratch/aniemcz/goose-pointcept/lcp_bin_decompressed_labels_challenge/${Q_lvl} \
      --no_threshold
done