#!/bin/bash

#SBATCH --job-name compressGooseWithTMC13
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node a100:0
#SBATCH --mem 16gb
#SBATCH --time 24:00:00

cd /home/aniemcz/gooseReno/tmc13GooseCompressScripts

pixi run python tmc13_compress_goose_dataset_parallel.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/valEx   --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_compression_results/valEx --workers 4

pixi run python tmc13_compress_goose_dataset_parallel.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/val   --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_compression_results/val --workers 4

pixi run python tmc13_compress_goose_dataset_parallel.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/trainEx    --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_compression_results/trainEx --workers 4

pixi run python tmc13_compress_goose_dataset_parallel.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/train  --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_compression_results/train --workers 4