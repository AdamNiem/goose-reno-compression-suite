pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/val   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results/val

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/val   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results/valEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/trainEx   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results/trainEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/train   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 128   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_compression_results/train
