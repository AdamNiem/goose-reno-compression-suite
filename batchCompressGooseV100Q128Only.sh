#!/bin/bash

#SBATCH --job-name compressGooseWithRenoQ128
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 64gb
#SBATCH --time 14:00:00

cd /home/aniemcz/gooseReno

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/val   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results_Q128_only/val

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/valEx   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results_Q128_only/valEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/trainEx   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results_Q128_only/trainEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/train   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results_Q128_only/train
